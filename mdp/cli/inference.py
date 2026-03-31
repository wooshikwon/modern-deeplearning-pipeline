"""mdp inference -- 배치 추론을 실행한다."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)


def run_inference(
    recipe_path: str,
    config_path: str,
    checkpoint_path: str | None = None,
) -> None:
    """Recipe + Config + (선택) 체크포인트로 배치 추론을 실행한다.

    1. SettingsFactory로 Settings 조립
    2. Factory로 모델 + 데이터로더 생성
    3. 체크포인트가 지정되면 모델에 가중치 로드
    4. run_batch_inference로 추론 실행 & 결과 저장
    """
    import torch

    from mdp.factory.factory import Factory
    from mdp.serving.inference import run_batch_inference
    from mdp.settings.factory import SettingsFactory

    typer.echo(f"Recipe: {recipe_path}")
    typer.echo(f"Config: {config_path}")

    try:
        settings = SettingsFactory().for_inference(recipe_path, config_path)
    except Exception as e:
        typer.echo(f"[error] Settings 로딩 실패: {e}", err=True)
        raise typer.Exit(code=1)

    try:
        factory = Factory(settings)
        model = factory.create_model()

        if model is None:
            typer.echo("[error] 모델 생성에 실패했습니다.", err=True)
            raise typer.Exit(code=1)

        # 체크포인트 로드
        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.exists():
                typer.echo(f"[error] 체크포인트를 찾을 수 없습니다: {ckpt_path}", err=True)
                raise typer.Exit(code=1)

            typer.echo(f"체크포인트 로드: {ckpt_path}")
            state_dict = torch.load(
                ckpt_path, map_location="cpu", weights_only=True
            )
            # state_dict가 체크포인트 디렉토리 내 model.pt인 경우 대응
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict)

        dataloaders = factory.create_dataloaders()
        test_loader = dataloaders.get("test", dataloaders.get("val", dataloaders.get("train")))

        if test_loader is None:
            typer.echo("[error] 추론용 데이터로더가 없습니다.", err=True)
            raise typer.Exit(code=1)

        output_dir = Path(settings.config.storage.output_dir)
        output_path = output_dir / f"{settings.recipe.name}_predictions"

        typer.echo("배치 추론 시작...")
        result_path = run_batch_inference(
            model=model,
            dataloader=test_loader,
            output_path=output_path,
            task=settings.recipe.task,
        )
        typer.echo(f"추론 완료. 결과: {result_path}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"[error] 추론 실패: {e}", err=True)
        logger.exception("추론 실패 상세")
        raise typer.Exit(code=1)
