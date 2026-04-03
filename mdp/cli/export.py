"""mdp export -- adapter/checkpoint에서 서빙 가능한 전체 모델을 내보낸다."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import typer

from mdp.cli.output import build_error, build_result, emit_result, is_json_mode

logger = logging.getLogger(__name__)


def run_export(
    run_id: str | None = None,
    checkpoint: str | None = None,
    output: str = "./exported-model",
) -> None:
    """adapter artifact 또는 checkpoint에서 merge된 전체 모델을 내보낸다.

    --run-id: MLflow run의 model artifact에서 내보내기.
    --checkpoint: 로컬 checkpoint 디렉토리에서 내보내기.
    """
    if run_id and checkpoint:
        msg = "--run-id와 --checkpoint는 동시에 지정할 수 없습니다."
        if is_json_mode():
            emit_result(build_error(command="export", error_type="ValidationError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    if not run_id and not checkpoint:
        msg = "--run-id 또는 --checkpoint 중 하나를 지정하세요."
        if is_json_mode():
            emit_result(build_error(command="export", error_type="ValidationError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    try:
        from mdp.serving.model_loader import reconstruct_model

        # 소스 디렉토리 결정
        if run_id:
            import mlflow

            if not is_json_mode():
                typer.echo(f"MLflow run: {run_id}")
            source_dir = Path(mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="model",
            ))
        else:
            source_dir = Path(checkpoint)
            if not source_dir.exists():
                raise FileNotFoundError(f"checkpoint 경로를 찾을 수 없습니다: {checkpoint}")

        if not is_json_mode():
            typer.echo(f"소스: {source_dir}")

        # 모델 재구성 + merge
        model, settings = reconstruct_model(source_dir, merge=True)
        recipe = settings.recipe
        target = getattr(model, "module", model)

        # 출력 디렉토리 생성
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 모델 저장
        if hasattr(target, "save_pretrained"):
            target.save_pretrained(output_dir)
        else:
            from safetensors.torch import save_file
            save_file(target.state_dict(), output_dir / "model.safetensors")

        # tokenizer 저장
        tokenizer_config = recipe.data.tokenizer
        if tokenizer_config:
            pretrained = tokenizer_config.get("pretrained") if isinstance(tokenizer_config, dict) else getattr(tokenizer_config, "pretrained", None)
            if pretrained:
                try:
                    from transformers import AutoTokenizer
                    AutoTokenizer.from_pretrained(pretrained).save_pretrained(output_dir)
                except Exception as e:
                    logger.warning(f"토크나이저 저장 실패 (무시): {e}")

        # recipe.yaml 복사
        recipe_src = source_dir / "recipe.yaml"
        if recipe_src.exists():
            shutil.copy(recipe_src, output_dir / "recipe.yaml")

        if not is_json_mode():
            typer.echo(f"내보내기 완료: {output_dir}")
        else:
            emit_result(build_result(
                command="export",
                output_dir=str(output_dir),
                model_class=type(target).__name__,
                merged=True,
            ))

    except typer.Exit:
        raise
    except Exception as e:
        if is_json_mode():
            emit_result(build_error(command="export", error_type="RuntimeError", message=str(e)))
            raise typer.Exit(code=1)
        typer.echo(f"[error] {e}", err=True)
        logger.exception("Export 실패 상세")
        raise typer.Exit(code=1)
