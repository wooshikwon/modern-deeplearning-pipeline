"""torchrun 엔트리포인트 -- 분산 학습 워커가 실행하는 스크립트."""

from __future__ import annotations

import argparse
import json
import os


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


def run_training(settings) -> dict:
    """Settings 객체를 받아 Factory -> Trainer -> train() 파이프라인을 실행한다.

    RL recipe(settings.recipe.rl이 존재)이면 RLTrainer로, 아니면 SFT Trainer로 디스패치한다.
    """
    if getattr(settings.recipe, "rl", None) is not None:
        return run_rl_training(settings)

    from mdp.factory.factory import Factory
    from mdp.training.trainer import Trainer

    factory = Factory(settings)
    model = factory.create_model()
    dataloaders = factory.create_dataloaders()

    trainer = Trainer(
        settings=settings,
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders.get("val"),
    )
    return trainer.train()


def run_rl_training(settings) -> dict:
    """Settings 객체를 받아 Factory -> RLTrainer -> train() 파이프라인을 실행한다."""
    from mdp.factory.factory import Factory
    from mdp.training.rl_trainer import RLTrainer

    factory = Factory(settings)
    models = factory.create_models()
    dataloaders = factory.create_dataloaders()

    trainer = RLTrainer(
        settings=settings,
        models=models,
        train_loader=dataloaders["train"],
        val_loader=dataloaders.get("val"),
    )
    return trainer.train()


def main() -> None:
    # PYTORCH_CUDA_ALLOC_CONF must be set before any CUDA allocation (including
    # dist.init_process_group(nccl)). expandable_segments:True lets the caching
    # allocator grow existing memory segments instead of creating new fixed-size
    # blocks, eliminating fragmentation-driven OOM in FSDP training.
    # setdefault preserves any value the caller may have already exported.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    settings = Settings(**raw)

    # DistributedSampler가 dataloader 생성 시 분산 통신을 요구하므로,
    # create_dataloaders 이전에 process group을 초기화한다.
    _init_distributed_if_torchrun(settings)

    result = run_training(settings)

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
