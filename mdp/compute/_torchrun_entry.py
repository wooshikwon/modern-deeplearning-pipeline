"""torchrun 엔트리포인트 -- 분산 학습 워커가 실행하는 스크립트."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="MDP torchrun worker")
    parser.add_argument(
        "--settings-path", required=True, help="Settings JSON 파일 경로"
    )
    args = parser.parse_args()

    from mdp.factory.factory import Factory
    from mdp.settings.schema import Settings
    from mdp.training.trainer import Trainer

    with open(args.settings_path) as f:
        raw = json.load(f)
    settings = Settings(**raw)

    factory = Factory(settings)
    model = factory.create_model()
    dataloaders = factory.create_dataloaders()

    trainer = Trainer(
        settings=settings,
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders.get("val"),
    )
    trainer.train()


if __name__ == "__main__":
    main()
