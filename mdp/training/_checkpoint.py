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

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.optim import Optimizer
    from mdp.training.strategies.base import StrategyCheckpointCapability

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
CHECKPOINT_LAYOUT_VERSION = 2
ModelRecordFormat = Literal[
    "peft_adapter",
    "safetensors",
    "safetensors_full_state_dict",
    "hf_pretrained",
    "torch_state_dict",
    "pretrained_dir",
]


# ── Manifest model ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelRecord:
    """Manifest entry for one model slot in a checkpoint."""

    role: str
    format: ModelRecordFormat
    path: str
    trainable: bool
    optimizer: str | None = None
    scheduler: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": self.role,
            "format": self.format,
            "path": self.path,
            "trainable": self.trainable,
        }
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRecord":
        model_format = str(data["format"])
        valid_formats = set(ModelRecordFormat.__args__)
        if model_format not in valid_formats:
            raise ValueError(
                f"지원하지 않는 checkpoint model format: {model_format}"
            )
        return cls(
            role=str(data["role"]),
            format=model_format,  # type: ignore[arg-type]
            path=str(data["path"]),
            trainable=bool(data["trainable"]),
            optimizer=data.get("optimizer"),
            scheduler=data.get("scheduler"),
        )


@dataclass(frozen=True)
class CheckpointManifest:
    """JSON-serializable checkpoint layout description."""

    layout_version: int = CHECKPOINT_LAYOUT_VERSION
    kind: Literal["sft", "rl"] = "sft"
    saved_at: Literal["step", "validation", "train_end", "manual"] = "manual"
    global_step: int = 0
    trainer_state_file: str | None = "trainer_state.json"
    recipe_file: str | None = None
    config_file: str | None = None
    models: dict[str, ModelRecord] = field(default_factory=dict)
    scaler: str | None = None
    epoch: int | None = None
    step_in_epoch: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "layout_version": self.layout_version,
            "kind": self.kind,
            "saved_at": self.saved_at,
            "global_step": self.global_step,
            "trainer_state_file": self.trainer_state_file,
            "recipe_file": self.recipe_file,
            "config_file": self.config_file,
            "models": {
                name: record.to_dict()
                for name, record in self.models.items()
            },
            "scaler": self.scaler,
        }
        if self.epoch is not None:
            data["epoch"] = self.epoch
        if self.step_in_epoch is not None:
            data["step_in_epoch"] = self.step_in_epoch
        if self.metrics:
            data["metrics"] = self.metrics
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointManifest":
        layout_version = int(data["layout_version"])
        if layout_version != CHECKPOINT_LAYOUT_VERSION:
            raise ValueError(
                f"지원하지 않는 checkpoint layout_version: {layout_version}"
            )
        models = {
            name: ModelRecord.from_dict(record)
            for name, record in data.get("models", {}).items()
        }
        return cls(
            layout_version=layout_version,
            kind=data["kind"],
            saved_at=data["saved_at"],
            global_step=int(data["global_step"]),
            trainer_state_file=data.get("trainer_state_file"),
            recipe_file=data.get("recipe_file"),
            config_file=data.get("config_file"),
            models=models,
            scaler=data.get("scaler"),
            epoch=data.get("epoch"),
            step_in_epoch=data.get("step_in_epoch"),
            metrics={
                str(key): float(value)
                for key, value in data.get("metrics", {}).items()
            },
        )

    def write(self, ckpt_dir: Path) -> Path:
        path = ckpt_dir / MANIFEST_FILE
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def read(cls, ckpt_dir: Path) -> "CheckpointManifest":
        return cls.from_dict(json.loads((ckpt_dir / MANIFEST_FILE).read_text()))


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Checkpoint load result shared by manager users and legacy wrappers."""

    ckpt_dir: Path
    trainer_state: dict[str, Any] | None
    scaler: Any | None
    manifest: CheckpointManifest | None = None
    legacy: bool = False

    def to_legacy_state(self) -> dict[str, Any]:
        return {
            "ckpt_dir": self.ckpt_dir,
            "trainer_state": self.trainer_state,
            "scaler": self.scaler,
        }


@dataclass(frozen=True)
class CheckpointContext:
    kind: Literal["sft", "rl"]
    ckpt_dir: Path
    global_step: int
    epoch: int | None = None
    step_in_epoch: int | None = None
    saved_at: Literal["step", "validation", "train_end", "manual"] = "manual"
    metrics: dict[str, float] = field(default_factory=dict)
    recipe_dict: dict[str, Any] | None = None
    is_main_process: bool = True


@dataclass(frozen=True)
class ModelSlot:
    name: str
    role: Literal["policy", "reference", "reward", "critic", "value", "model"]
    model: "nn.Module"
    trainable: bool
    optimizer: "Optimizer | None" = None
    scheduler: Any | None = None


class CheckpointManager:
    """Owns manifest-aware checkpoint save/load entry points."""

    def save(
        self,
        context: CheckpointContext,
        slots: list[ModelSlot],
        *,
        strategy: Any | None = None,
        scaler: Any | None = None,
    ) -> Path:
        ckpt_dir = context.ckpt_dir
        capability = (
            _strategy_checkpoint_capability(strategy)
            if strategy is not None else None
        )
        if strategy is not None and not context.is_main_process:
            if capability is not None and capability.requires_all_ranks_for_save:
                for slot in slots:
                    model_dir = ckpt_dir if slot.name == "" else ckpt_dir / slot.name
                    strategy.save_checkpoint(
                        slot.model,
                        str(model_dir / "model.safetensors"),
                    )
            return ckpt_dir
        if strategy is None and not context.is_main_process:
            return ckpt_dir

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer_state = {
            "epoch": context.epoch,
            "global_step": context.global_step,
            "step_in_epoch": context.step_in_epoch,
            "metrics": context.metrics,
        }
        (ckpt_dir / "trainer_state.json").write_text(json.dumps(trainer_state))

        recipe_file = None
        if context.recipe_dict is not None:
            import yaml

            recipe_file = "recipe.yaml"
            recipe_path = ckpt_dir / recipe_file
            if not recipe_path.exists():
                recipe_path.write_text(
                    yaml.dump(context.recipe_dict, allow_unicode=True)
                )

        models: dict[str, ModelRecord] = {}
        for slot in slots:
            model_dir = ckpt_dir if slot.name == "" else ckpt_dir / slot.name
            model_dir.mkdir(parents=True, exist_ok=True)
            target = getattr(slot.model, "module", slot.model)
            if strategy is not None:
                strategy.save_checkpoint(slot.model, str(model_dir / "model.safetensors"))
                model_path = _relative_to_ckpt(model_dir / "model.safetensors", ckpt_dir)
                model_format = (
                    capability.weight_format
                    if capability is not None else "safetensors"
                )
            elif hasattr(target, "save_pretrained"):
                target.save_pretrained(model_dir)
                model_path, model_format = _detect_model_file(model_dir, ckpt_dir)
            else:
                torch.save(target.state_dict(), model_dir / "model.pt")
                model_path = _relative_to_ckpt(model_dir / "model.pt", ckpt_dir)
                model_format = "torch_state_dict"

            optimizer_path = None
            if slot.optimizer is not None:
                optimizer_path = _relative_to_ckpt(model_dir / "optimizer.pt", ckpt_dir)
                torch.save(slot.optimizer.state_dict(), ckpt_dir / optimizer_path)

            scheduler_path = None
            if slot.scheduler is not None:
                scheduler_path = _relative_to_ckpt(model_dir / "scheduler.pt", ckpt_dir)
                torch.save(slot.scheduler.state_dict(), ckpt_dir / scheduler_path)

            models[slot.name or "model"] = ModelRecord(
                role=slot.role,
                format=model_format,
                path=model_path,
                trainable=slot.trainable,
                optimizer=optimizer_path,
                scheduler=scheduler_path,
            )

        scaler_path = None
        if scaler is not None:
            scaler_state = scaler.state_dict() if hasattr(scaler, "state_dict") else scaler
            scaler_path = "scaler.pt"
            torch.save(scaler_state, ckpt_dir / scaler_path)

        manifest = CheckpointManifest(
            kind=context.kind,
            saved_at=context.saved_at,
            global_step=context.global_step,
            trainer_state_file="trainer_state.json",
            recipe_file=recipe_file,
            models=models,
            scaler=scaler_path,
            epoch=context.epoch,
            step_in_epoch=context.step_in_epoch,
            metrics=context.metrics,
        )
        manifest.write(ckpt_dir)
        return ckpt_dir

    def load(self, ckpt_dir: Path) -> LoadedCheckpoint:
        return self.restore(ckpt_dir)

    def restore(
        self,
        ckpt_dir: Path,
        slots: list[ModelSlot] | None = None,
        *,
        strategy: Any | None = None,
        scaler: Any | None = None,
    ) -> LoadedCheckpoint:
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"체크포인트 경로가 존재하지 않습니다: {ckpt_dir}")
        if not (ckpt_dir / MANIFEST_FILE).exists():
            return _load_legacy_checkpoint(ckpt_dir)

        manifest = CheckpointManifest.read(ckpt_dir)
        if slots is not None:
            _restore_manifest_slots(ckpt_dir, manifest, slots, strategy=strategy)
        trainer_state = _read_json_file(ckpt_dir, manifest.trainer_state_file)
        scaler_state = _load_torch_file(ckpt_dir, manifest.scaler)
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        return LoadedCheckpoint(
            ckpt_dir=ckpt_dir,
            trainer_state=trainer_state,
            scaler=scaler_state,
            manifest=manifest,
            legacy=False,
        )


def _strategy_checkpoint_capability(strategy: Any) -> "StrategyCheckpointCapability":
    capability = getattr(strategy, "checkpoint_capability", None)
    if capability is None:
        raise ValueError(
            f"{type(strategy).__name__} does not declare checkpoint capability"
        )
    if not capability.supports_managed_checkpoint:
        reason = capability.unsupported_reason or type(strategy).__name__
        raise ValueError(
            "CheckpointManager does not support this strategy checkpoint path: "
            f"{reason}. DeepSpeed ZeRO checkpoints require a separate "
            "engine-contract implementation and are not restored as DDP/FSDP "
            "checkpoints."
        )
    return capability


def _relative_to_ckpt(path: Path, ckpt_dir: Path) -> str:
    return path.relative_to(ckpt_dir).as_posix()


def _detect_model_file(model_dir: Path, ckpt_dir: Path) -> tuple[str, str]:
    candidates = [
        ("adapter_model.safetensors", "peft_adapter"),
        ("model.safetensors", "safetensors"),
        ("pytorch_model.bin", "hf_pretrained"),
        ("model.pt", "torch_state_dict"),
    ]
    for filename, model_format in candidates:
        path = model_dir / filename
        if path.exists():
            return _relative_to_ckpt(path, ckpt_dir), model_format
    return _relative_to_ckpt(model_dir, ckpt_dir), "pretrained_dir"


def _record_for_slot(
    manifest: CheckpointManifest,
    slot: ModelSlot,
) -> ModelRecord | None:
    candidates = [slot.name]
    if slot.name == "":
        candidates.append("model")
    candidates.append(slot.role)
    for key in candidates:
        if key and key in manifest.models:
            return manifest.models[key]
    for record in manifest.models.values():
        if record.role == slot.role:
            return record
    return None


def _restore_manifest_slots(
    ckpt_dir: Path,
    manifest: CheckpointManifest,
    slots: list[ModelSlot],
    *,
    strategy: Any | None,
) -> None:
    for slot in slots:
        record = _record_for_slot(manifest, slot)
        if record is None:
            logger.warning(
                "Resume: manifest에 slot name=%s role=%s record 없음, 건너뜀",
                slot.name,
                slot.role,
            )
            continue
        model_path = ckpt_dir / record.path
        model_dir = model_path.parent
        if strategy is not None and record.format in {
            "safetensors",
            "safetensors_full_state_dict",
        }:
            strategy.load_checkpoint(slot.model, str(model_path))
        else:
            _load_model_record(slot.model, model_dir, model_path, record)
        if slot.optimizer is not None and record.optimizer is not None:
            slot.optimizer.load_state_dict(
                torch.load(ckpt_dir / record.optimizer, map_location="cpu", weights_only=True)
            )
        if slot.scheduler is not None and record.scheduler is not None:
            slot.scheduler.load_state_dict(
                torch.load(ckpt_dir / record.scheduler, map_location="cpu", weights_only=True)
            )


def _load_model_record(
    model: "nn.Module",
    model_dir: Path,
    model_path: Path,
    record: ModelRecord,
) -> None:
    target = getattr(model, "module", model)
    if record.format == "peft_adapter":
        if hasattr(target, "load_adapter"):
            from mdp.serving.model_loader import _get_adapter_name

            adapter_name = _get_adapter_name(model_dir)
            target.load_adapter(str(model_dir), adapter_name=adapter_name)
        else:
            logger.warning(
                "adapter checkpoint found but model has no load_adapter method: %s",
                model_dir,
            )
    elif record.format in {"safetensors", "safetensors_full_state_dict"}:
        from safetensors.torch import load_file

        target.load_state_dict(load_file(model_path), strict=False)
    elif record.format == "hf_pretrained":
        target.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True),
            strict=False,
        )
    elif record.format == "torch_state_dict":
        target.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True),
            strict=False,
        )
    elif record.format == "pretrained_dir":
        logger.warning(
            "pretrained_dir checkpoint record cannot be injected as state_dict: %s",
            model_path,
        )


def _read_json_file(ckpt_dir: Path, relative_path: str | None) -> dict[str, Any] | None:
    if relative_path is None:
        return None
    path = ckpt_dir / relative_path
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_torch_file(ckpt_dir: Path, relative_path: str | None) -> Any | None:
    if relative_path is None:
        return None
    path = ckpt_dir / relative_path
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


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


def load_checkpoint(
    ckpt_dir: Path,
    slots: list[ModelSlot] | None = None,
    *,
    strategy: Any | None = None,
    scaler: Any | None = None,
) -> dict:
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
    return CheckpointManager().restore(
        ckpt_dir,
        slots,
        strategy=strategy,
        scaler=scaler,
    ).to_legacy_state()


def _load_legacy_checkpoint(ckpt_dir: Path) -> LoadedCheckpoint:
    trainer_state = _read_json_file(ckpt_dir, "trainer_state.json")
    scaler = _load_torch_file(ckpt_dir, "scaler.pt")
    return LoadedCheckpoint(
        ckpt_dir=ckpt_dir,
        trainer_state=trainer_state,
        scaler=scaler,
        manifest=None,
        legacy=True,
    )


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
    import mlflow
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            _write_serving_model_artifact(
                model,
                settings,
                output_dir,
                policy_state_dict=policy_state_dict,
            )
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
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            _write_serving_model_artifact(model, settings, output_dir)

            # recipe.yaml 복사
            recipe_src = checkpoint_dir / "recipe.yaml"
            if recipe_src.exists():
                shutil.copy(recipe_src, output_dir / "recipe.yaml")

            mlflow.log_artifacts(tmp, "model")
            logger.info("모델 artifact를 MLflow에 등록: model/")

    except Exception as e:
        logger.warning(f"모델 artifact 등록 실패 (학습 결과는 유효합니다): {e}")


def _write_serving_model_artifact(
    model: "nn.Module",
    settings: Any,
    output_dir: Path,
    *,
    policy_state_dict: "dict | None" = None,
) -> None:
    """Write the flat serving artifact layout consumed by ``reconstruct_model``."""
    import yaml

    target = getattr(model, "module", model)
    has_adapter = hasattr(target, "peft_config")

    if policy_state_dict is not None:
        if has_adapter:
            from peft import get_peft_model_state_dict
            from safetensors.torch import save_file

            adapter_names = list(target.peft_config.keys())
            adapter_name = adapter_names[0] if adapter_names else "default"
            adapter_sd = get_peft_model_state_dict(
                target,
                state_dict=policy_state_dict,
                adapter_name=adapter_name,
            )
            save_file(adapter_sd, str(output_dir / "adapter_model.safetensors"))
            for _adapter_name, cfg in target.peft_config.items():
                cfg.save_pretrained(str(output_dir))
        else:
            from safetensors.torch import save_file

            save_file(policy_state_dict, str(output_dir / "model.safetensors"))
    elif has_adapter or hasattr(target, "save_pretrained"):
        target.save_pretrained(output_dir)
    else:
        from safetensors.torch import save_file

        save_file(target.state_dict(), output_dir / "model.safetensors")

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
        except Exception as e:
            logger.warning(f"토크나이저 저장 실패 (무시): {e}")

    recipe_dict = recipe.model_dump(mode="json")
    (output_dir / "recipe.yaml").write_text(
        yaml.dump(recipe_dict, allow_unicode=True)
    )


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
