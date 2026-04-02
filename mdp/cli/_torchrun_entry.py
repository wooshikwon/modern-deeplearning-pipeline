"""torchrun 엔트리포인트 -- 분산 학습 워커가 실행하는 스크립트."""

from __future__ import annotations

import argparse
import json


def run_training(settings) -> dict:
    """Settings 객체를 받아 Factory -> Trainer -> train() 파이프라인을 실행한다."""
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


def main() -> None:
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
