"""mdp export -- adapter/checkpoint에서 서빙 가능한 전체 모델을 내보낸다."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import typer

from mdp.cli.output import build_error, build_result, emit_result, is_json_mode, resolve_model_source

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
    try:
        source_dir = resolve_model_source(run_id, checkpoint, "export")
    except typer.BadParameter as e:
        msg = str(e)
        if is_json_mode():
            emit_result(build_error(command="export", error_type="ValidationError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    try:
        from mdp.serving.model_loader import reconstruct_model

        if not run_id and not source_dir.exists():
            raise FileNotFoundError(f"checkpoint 경로를 찾을 수 없습니다: {checkpoint}")

        if not is_json_mode() and run_id:
            typer.echo(f"MLflow run: {run_id}")

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
        # BaseModel.export()가 있으면 우선 위임 (backbone+head 분리 저장 등 커스텀 구조 지원).
        # 순수 HF 모델(save_pretrained)과 generic fallback은 이후 처리.
        if hasattr(target, "export"):
            target.export(output_dir)
        elif hasattr(target, "save_pretrained"):
            target.save_pretrained(output_dir)
        else:
            from safetensors.torch import save_file
            save_file(target.state_dict(), output_dir / "model.safetensors")

        # tokenizer 저장 — collator _component_의 init_args에서 추출
        tokenizer_name = recipe.data.collator.get("tokenizer") if isinstance(recipe.data.collator, dict) else None
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer
                AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(output_dir)
            except Exception as e:
                logger.warning(f"토크나이저 저장 실패 (무시): {e}")

        # recipe.yaml 복사
        recipe_src = source_dir / "recipe.yaml"
        if recipe_src.exists():
            shutil.copy(recipe_src, output_dir / "recipe.yaml")

        if not is_json_mode():
            typer.echo(f"내보내기 완료: {output_dir}")
        else:
            from mdp.cli.schemas import ExportResult
            result = ExportResult(
                output_dir=str(output_dir),
                model_class=type(target).__name__,
                merged=True,
            )
            emit_result(build_result(
                command="export", **result.model_dump(exclude_none=True),
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
