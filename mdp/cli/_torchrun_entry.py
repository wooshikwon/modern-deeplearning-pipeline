"""torchrun 엔트리포인트 -- 분산 학습 워커가 실행하는 스크립트."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any


def _init_distributed_if_torchrun(settings) -> None:
    """torchrun 워커로 실행 중이면 process group을 초기화한다.

    분산 환경 초기화는 프로세스 생명주기의 문제이지 Trainer 생명주기의
    문제가 아니다. ``Factory.create_dataloaders()``가 ``DistributedSampler``를
    만들 때 ``dist.get_world_size()``를 호출하는데, 이 시점에 process group이
    이미 초기화되어 있어야 한다.

    기존에는 FSDPStrategy/Trainer 내부에서 ``init_process_group()``을 호출했으나,
    그 호출은 create_dataloaders 이후에 발생하므로 DataLoader 생성이 실패했다.
    이 함수는 entry point에서 먼저 초기화를 수행하고, 기존 trainer/strategy의
    중복 호출은 ``is_initialized()`` 가드로 no-op이 된다.

    torchrun이 아닌 단일 프로세스 실행(예: 라이브러리 import)에서는
    RANK/WORLD_SIZE 환경변수가 없으므로 조기 return한다.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return

    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        return

    # backend: Config.compute.distributed.backend → fallback nccl(cuda)/gloo(cpu)
    dist_cfg = settings.config.compute.distributed
    backend = None
    if isinstance(dist_cfg, dict):
        backend = dist_cfg.get("backend")
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend)


def _is_main_process() -> bool:
    """rank-0 여부를 반환한다. torchrun 환경이 아니면 항상 True."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return int(os.environ.get("RANK", "0")) == 0


def _print_callbacks_log(
    cb_instances: list,
    settings,
) -> None:
    """실행 시작 시 적용된 callbacks와 training.* 자동 추가 항목을 출력한다.

    is_json_mode()이면 생략. rank-0에서만 출력.
    """
    from mdp.cli.output import is_json_mode

    if is_json_mode() or not _is_main_process():
        return

    from mdp.callbacks.base import BaseInterventionCallback

    lines = []
    for cb in cb_instances:
        tag = "[Int]" if isinstance(cb, BaseInterventionCallback) else "[Obs]"
        cb_name = type(cb).__name__
        # 대표 속성 표시 (monitor 있으면)
        attrs = []
        if hasattr(cb, "monitor"):
            attrs.append(f"monitor={cb.monitor}")
        detail = f" ({', '.join(attrs)})" if attrs else ""
        lines.append(f"  {tag} {cb_name}{detail}")

    # training.* 자동 추가 항목
    auto_lines = []
    training = settings.recipe.training
    if training.early_stopping is not None:
        es = training.early_stopping
        auto_lines.append(
            f"  EarlyStopping (monitor={es.monitor}, patience={es.patience})"
        )
    if training.ema is not None:
        ema = training.ema
        auto_lines.append(f"  EMA (decay={ema.decay})")

    if lines or auto_lines:
        if lines:
            print("Applied callbacks:")
            for line in lines:
                print(line)
        if auto_lines:
            print("Auto-promoted from training.*:")
            for line in auto_lines:
                print(line)


def _resolve_cb_configs(cb_configs: list[dict[str, Any]]) -> list:
    """cb_configs list[dict]를 실제 callback 인스턴스 리스트로 resolve한다."""
    from mdp.settings.resolver import ComponentResolver
    from mdp.training._common import create_callbacks

    resolver = ComponentResolver()
    return create_callbacks(cb_configs, resolver)


def run_training(settings, cb_configs: list[dict] | None = None) -> dict:
    """Settings 객체를 받아 Factory -> Trainer -> train() 파이프라인을 실행한다.

    RL recipe(settings.recipe.rl이 존재)이면 RLTrainer로, 아니면 SFT Trainer로 디스패치한다.

    Args:
        settings: 학습 Settings 객체.
        cb_configs: CLI --callbacks에서 로드된 list[dict[str, Any]]. None이면 빈 리스트로 처리.
    """
    # Liger-Kernel monkey-patch는 HF `from_pretrained`(Factory.create_model 내부)
    # 이전에 수행되어야 효과가 있다. 단일 GPU 실행 경로와 torchrun worker subprocess
    # 양쪽에서 이 함수를 공통으로 거치므로 여기에 배치한다. Idempotent.
    # 상세: mdp/_liger_patch.py, spec-algorithm-hidden-states-support §U2.
    from mdp._liger_patch import apply_liger_patches
    apply_liger_patches()

    if getattr(settings.recipe, "rl", None) is not None:
        return run_rl_training(settings, cb_configs=cb_configs)

    from mdp.factory.factory import Factory
    from mdp.training.trainer import Trainer

    # cb_configs를 실제 callback 인스턴스로 resolve (torchrun worker 프로세스 내부에서 수행)
    callbacks = _resolve_cb_configs(cb_configs) if cb_configs else []

    # 실행 시작 로그 (rank-0, non-JSON 모드만)
    _print_callbacks_log(callbacks, settings)

    factory = Factory(settings)
    model = factory.create_model()
    dataloaders = factory.create_dataloaders()

    trainer = Trainer(
        settings=settings,
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders.get("val"),
        callbacks=callbacks if callbacks else None,
    )
    return trainer.train()


def run_rl_training(settings, cb_configs: list[dict] | None = None) -> dict:
    """Settings 객체를 받아 Factory -> RLTrainer -> train() 파이프라인을 실행한다."""
    from mdp.factory.factory import Factory
    from mdp.training.rl_trainer import RLTrainer

    # cb_configs를 실제 callback 인스턴스로 resolve
    callbacks = _resolve_cb_configs(cb_configs) if cb_configs else []

    # 실행 시작 로그 (rank-0, non-JSON 모드만)
    _print_callbacks_log(callbacks, settings)

    factory = Factory(settings)
    models = factory.create_models()
    dataloaders = factory.create_dataloaders()

    trainer = RLTrainer(
        settings=settings,
        models=models,
        train_loader=dataloaders["train"],
        val_loader=dataloaders.get("val"),
        callbacks=callbacks if callbacks else None,
    )
    return trainer.train()


def main() -> None:
    # PYTORCH_CUDA_ALLOC_CONF must be set before any CUDA allocation (including
    # dist.init_process_group(nccl)). expandable_segments:True lets the caching
    # allocator grow existing memory segments instead of creating new fixed-size
    # blocks, eliminating fragmentation-driven OOM in FSDP training.
    # setdefault preserves any value the caller may have already exported.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # 워커 프로세스의 기본 로깅 레벨은 WARNING이라 logger.info()가 묻힌다.
    # INFO로 설정하여 GC/FSDP 진단 메시지가 로그에 남도록 한다.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(description="MDP torchrun worker")
    parser.add_argument(
        "--settings-path", required=True, help="Settings JSON 파일 경로"
    )
    parser.add_argument(
        "--result-path", default=None, help="학습 결과를 저장할 JSON 파일 경로 (rank-0 전용)"
    )
    args = parser.parse_args()

    from mdp.settings.schema import Settings

    with open(args.settings_path) as f:
        raw = json.load(f)

    # __cb_configs는 Settings 스키마 밖 임시 필드로 전달됨 (extra="forbid" 우회)
    cb_configs: list[dict] | None = raw.pop("__cb_configs", None)

    settings = Settings(**raw)

    # DistributedSampler가 dataloader 생성 시 분산 통신을 요구하므로,
    # create_dataloaders 이전에 process group을 초기화한다.
    _init_distributed_if_torchrun(settings)

    # Liger monkey-patch는 run_training() 내부에서 수행된다 (Factory.create_model
    # 호출 이전). 여기서는 dist init만 보장.
    result = run_training(settings, cb_configs=cb_configs)

    # rank-0만 결과를 저장한다
    if args.result_path and result:
        try:
            import torch.distributed as dist
            is_main = not dist.is_initialized() or dist.get_rank() == 0
        except Exception:
            is_main = True
        if is_main:
            with open(args.result_path, "w") as f:
                json.dump(result, f, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
