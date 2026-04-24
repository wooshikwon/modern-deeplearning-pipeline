"""Checkpoint I/O — 학습 state를 파일 시스템에 쓰고 복원한다.

spec-training-restructure U3에서 신설. Trainer / RLTrainer의 save/resume/export 로직을
compute 레이어(trainer 본체)에서 분리하여 이 I/O 레이어에 단일화한다.

책임:
- ``save_checkpoint``: state dict를 ckpt_dir에 직렬화
- ``load_checkpoint``: ckpt_dir에서 state dict를 복원 (순수 함수, side-effect 없음)
- ``gather_fsdp_state_dict``: FSDP 모델의 full state dict를 all-rank 협력으로 수집
- ``export_model_artifact``: policy / SFT 모델을 MLflow artifact로 등록
- ``find_best_checkpoint``: best/latest symlink로 최적 체크포인트 경로 조회

외부(``mdp/cli/``, ``mdp/serving/``)에서 직접 import하지 않는다 —
``_``-prefix 파일은 ``training/`` 네임스페이스의 private 구현.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Save ────────────────────────────────────────────────────────────────────


def save_checkpoint(state: dict, ckpt_dir: Path) -> None:
    """state dict를 ckpt_dir에 직렬화한다.

    state 구조는 trainer가 ``_checkpoint_state()``로 반환한 dict와 동일하다.
    이 함수는 I/O만 담당하며 state의 의미는 caller(trainer)가 결정한다.

    :param state: ``_checkpoint_state()``가 반환한 직렬화 대상 dict.
    :param ckpt_dir: 저장할 체크포인트 디렉토리 (없으면 자동 생성).
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # trainer_state.json — 에포크·스텝 등 scalar 상태
    trainer_state = state.get("trainer_state")
    if trainer_state is not None:
        import json
        (ckpt_dir / "trainer_state.json").write_text(json.dumps(trainer_state))

    # model weights — 단일 모델 (Trainer) 또는 복수 모델 (RLTrainer)
    models: dict[str, Any] = state.get("models", {})
    for name, model_state in models.items():
        model_dir = ckpt_dir if name == "" else ckpt_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        _save_model_state(model_state, model_dir)

    # optimizer / scheduler / scaler — 단일 (Trainer) 또는 per-model (RLTrainer)
    optimizers: dict[str, Any] = state.get("optimizers", {})
    for name, opt_sd in optimizers.items():
        target_dir = ckpt_dir if name == "" else ckpt_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        torch.save(opt_sd, target_dir / "optimizer.pt")

    schedulers: dict[str, Any] = state.get("schedulers", {})
    for name, sched_sd in schedulers.items():
        target_dir = ckpt_dir if name == "" else ckpt_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        torch.save(sched_sd, target_dir / "scheduler.pt")

    # scaler — AMP GradScaler
    scaler_sd = state.get("scaler")
    if scaler_sd is not None:
        torch.save(scaler_sd, ckpt_dir / "scaler.pt")

    # recipe.yaml — 재현용 snapshot
    recipe_dict = state.get("recipe_dict")
    if recipe_dict is not None:
        import yaml
        (ckpt_dir / "recipe.yaml").write_text(yaml.dump(recipe_dict, allow_unicode=True))


def _save_model_state(model_state: dict, model_dir: Path) -> None:
    """단일 모델의 저장 방식(adapter / safetensors / pt)을 결정하여 기록한다.

    model_state 키:
    - ``"save_pretrained_dir"``: PEFT / HF save_pretrained 호출 경로
    - ``"safetensors"``: safetensors bytes dict  → ``model.safetensors``
    - ``"state_dict_pt"``: torch state dict → ``model.pt``
    """
    if "save_pretrained_dir" in model_state:
        # save_pretrained_dir에 임시로 저장된 파일들을 model_dir로 이동하거나,
        # 실제 호출은 trainer가 strategy/PEFT를 알아야 하므로 여기서는
        # "_strategy_save" 키를 통해 처리한다.
        pass

    if "safetensors" in model_state:
        from safetensors.torch import save_file
        save_file(model_state["safetensors"], model_dir / "model.safetensors")
    elif "state_dict_pt" in model_state:
        torch.save(model_state["state_dict_pt"], model_dir / "model.pt")


# ── Load ────────────────────────────────────────────────────────────────────


def load_checkpoint(ckpt_dir: Path) -> dict:
    """ckpt_dir에서 학습 상태를 복원하여 dict로 반환한다.

    이 함수는 파일 읽기만 수행하는 순수 함수다. 실제 복원 (모델에 state_dict 주입,
    global_step 재설정 등)은 caller인 trainer가 ``_load_checkpoint_state(state)``로
    처리한다.

    반환 dict 구조:
    - ``"ckpt_dir"``: 체크포인트 디렉토리 경로 (Path) — 모델/optimizer 로드 경로 계산용
    - ``"trainer_state"``: trainer_state.json 내용 (dict 또는 None)
    - ``"scaler"``: GradScaler state_dict (또는 None)

    :param ckpt_dir: 복원할 체크포인트 디렉토리.
    :returns: 위 구조의 dict. trainer가 ``_load_checkpoint_state``로 소비한다.
    :raises FileNotFoundError: ckpt_dir가 존재하지 않으면 raise.
    """
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"체크포인트 경로가 존재하지 않습니다: {ckpt_dir}")

    state: dict[str, Any] = {"ckpt_dir": ckpt_dir}

    # trainer_state.json
    state_path = ckpt_dir / "trainer_state.json"
    if state_path.exists():
        import json
        state["trainer_state"] = json.loads(state_path.read_text())
    else:
        state["trainer_state"] = None

    # scaler
    scaler_path = ckpt_dir / "scaler.pt"
    if scaler_path.exists():
        state["scaler"] = torch.load(scaler_path, map_location="cpu", weights_only=True)
    else:
        state["scaler"] = None

    return state


# ── FSDP state_dict ─────────────────────────────────────────────────────────


def gather_fsdp_state_dict(model: "nn.Module", is_main_process: bool) -> "dict | None":
    """FSDP 모델의 full state dict를 all-rank 협력으로 수집한다.

    모든 rank가 반드시 호출해야 한다 (NCCL all-gather가 내부에서 실행됨).
    ``rank0_only=True``이므로 실제 weight는 rank 0에만 채워지고, 나머지는 빈 dict.
    FSDP가 아닌 경우 None을 반환한다.

    .. warning::
        NCCL collective이 포함되므로 반드시 모든 rank에서 호출해야 한다.
        한 rank만 호출하면 all-gather가 블로킹 상태로 deadlock된다.

    :param model: DDP/FSDP로 래핑된 (또는 일반) nn.Module.
    :param is_main_process: rank 0 여부 (``int(os.environ.get("RANK", "0")) == 0``).
    :returns: rank 0이면 full state dict, 그 외 rank이면 None. FSDP 아니면 None.
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if not isinstance(model, FSDP):
            return None
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    except Exception as e:
        logger.warning("FSDP state dict cooperative gather failed: %s", e)
        return None

    # NCCL collective — outside try/except so a raise here propagates to all ranks
    # instead of one rank silently returning None while others block in all-gather.
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        # All ranks participate in NCCL all-gather here.
        # rank0_only=True → result populated on rank 0 only; others get {}.
        state_dict = model.state_dict()
    return state_dict if is_main_process else None


# ── Export ──────────────────────────────────────────────────────────────────


def export_model_artifact(
    model: "nn.Module",
    settings: Any,
    *,
    policy_state_dict: "dict | None" = None,
) -> None:
    """Policy / SFT 모델을 MLflow artifact로 등록한다.

    LoRA면 adapter만, full finetuning이면 전체 모델을 저장한다.
    merge는 수행하지 않는다 — merge는 ``mdp export`` / ``mdp serve`` 시점에 on-demand.

    :param model: 등록할 모델 (DDP/FSDP 래퍼 포함 가능).
    :param settings: Settings 객체 (recipe, tokenizer 정보 추출용).
    :param policy_state_dict: FSDP cooperative gather로 수집한 full state dict.
                              제공되면 모델에서 직접 state_dict를 읽지 않는다.
    """
    import tempfile

    import mlflow

    try:
        import yaml

        target = getattr(model, "module", model)
        has_adapter = hasattr(target, "peft_config")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            if policy_state_dict is not None:
                # FSDP path: use pre-gathered full state dict to avoid NCCL on rank 0 only.
                if has_adapter:
                    # Extract adapter-only weights via PEFT helper (respects state_dict arg).
                    from peft import get_peft_model_state_dict
                    adapter_names = list(target.peft_config.keys())
                    adapter_name = adapter_names[0] if adapter_names else "default"
                    adapter_sd = get_peft_model_state_dict(
                        target,
                        state_dict=policy_state_dict,
                        adapter_name=adapter_name,
                    )
                    from safetensors.torch import save_file
                    save_file(adapter_sd, str(output_dir / "adapter_model.safetensors"))
                    # Save adapter config for each adapter name.
                    peft_config = target.peft_config  # dict[adapter_name, PeftConfigMixin]
                    for _adapter_name, cfg in peft_config.items():
                        cfg.save_pretrained(str(output_dir))
                else:
                    from safetensors.torch import save_file
                    save_file(policy_state_dict, str(output_dir / "model.safetensors"))
            elif has_adapter:
                target.save_pretrained(output_dir)
            elif hasattr(target, "save_pretrained"):
                target.save_pretrained(output_dir)
            else:
                from safetensors.torch import save_file
                save_file(target.state_dict(), str(output_dir / "model.safetensors"))

            # tokenizer — collator _component_의 init_args에서 추출
            recipe = settings.recipe
            tokenizer_name = (
                recipe.data.collator.get("tokenizer")
                if isinstance(recipe.data.collator, dict)
                else None
            )
            if tokenizer_name:
                try:
                    from transformers import AutoTokenizer
                    AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(output_dir)
                except Exception:
                    pass

            # recipe.yaml
            recipe_dict = recipe.model_dump(mode="json")
            (output_dir / "recipe.yaml").write_text(yaml.dump(recipe_dict, allow_unicode=True))

            mlflow.log_artifacts(tmp, "model")
            logger.info("모델을 MLflow artifact로 등록: model/")
    except Exception as e:
        logger.warning(f"모델 artifact 저장 실패: {e}")


def export_sft_model_artifact(
    model: "nn.Module",
    settings: Any,
    checkpoint_dir: Path,
) -> None:
    """SFT Trainer 전용 모델 artifact 등록.

    LoRA면 adapter만, full finetuning이면 전체 모델을 저장한다.
    tokenizer와 recipe.yaml도 함께 등록한다.

    :param model: 등록할 모델 (DDP/FSDP 래퍼 포함 가능).
    :param settings: Settings 객체 (recipe, tokenizer 정보 추출용).
    :param checkpoint_dir: recipe.yaml 소스 디렉토리.
    """
    import shutil
    import tempfile

    import mlflow

    try:
        recipe = settings.recipe
        target = getattr(model, "module", model)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            # 모델 가중치: PEFT면 adapter만, 아니면 전체
            has_adapter = hasattr(target, "save_pretrained") and hasattr(target, "peft_config")
            if has_adapter:
                target.save_pretrained(output_dir)
            elif hasattr(target, "save_pretrained"):
                target.save_pretrained(output_dir)
            else:
                from safetensors.torch import save_file
                save_file(target.state_dict(), output_dir / "model.safetensors")

            # tokenizer 저장
            tokenizer_name = (
                recipe.data.collator.get("tokenizer")
                if isinstance(recipe.data.collator, dict)
                else None
            )
            if tokenizer_name:
                try:
                    from transformers import AutoTokenizer
                    AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(output_dir)
                except Exception as e:
                    logger.warning(f"토크나이저 저장 실패 (무시): {e}")

            # recipe.yaml 복사
            recipe_src = checkpoint_dir / "recipe.yaml"
            if recipe_src.exists():
                shutil.copy(recipe_src, output_dir / "recipe.yaml")

            mlflow.log_artifacts(tmp, "model")
            logger.info("모델 artifact를 MLflow에 등록: model/")

    except Exception as e:
        logger.warning(f"모델 artifact 등록 실패 (학습 결과는 유효합니다): {e}")


# ── Find ────────────────────────────────────────────────────────────────────


def find_best_checkpoint(checkpoint_dir: Path) -> "Path | None":
    """``best`` 또는 ``latest`` symlink가 가리키는 체크포인트 디렉토리를 반환한다.

    :param checkpoint_dir: ModelCheckpoint가 저장하는 최상위 디렉토리.
    :returns: best → latest 순서로 symlink를 해석한 절대 경로, 없으면 None.
    """
    for name in ("best", "latest"):
        link = checkpoint_dir / name
        if link.exists():
            return link.resolve()
    return None
