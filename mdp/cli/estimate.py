"""mdp estimate -- GPU 메모리 사용량을 추정한다."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_estimate(recipe_path: str) -> None:
    """Recipe YAML을 파싱하여 GPU 메모리 사용량을 추정한다."""
    from rich.console import Console
    from rich.table import Table

    import typer

    from mdp.cli.output import build_error, build_result, emit_result, is_json_mode
    from mdp.cli.schemas import EstimateResult
    from mdp.settings.loader import SettingsLoader
    from mdp.utils.estimator import MemoryEstimator

    try:
        settings = SettingsLoader().load_estimation_settings(recipe_path)
    except Exception as e:
        if is_json_mode():
            emit_result(build_error(
                command="estimate",
                error_type="ValidationError",
                message=str(e),
            ))
            raise typer.Exit(code=1)
        typer.echo(f"[error] Settings 로딩 실패: {e}", err=True)
        raise typer.Exit(code=1)

    raw = MemoryEstimator().estimate(settings)

    if is_json_mode():
        result = EstimateResult(**raw)
        emit_result(build_result(
            command="estimate", **result.model_dump(exclude_none=True),
        ))
        return

    # Rich Table 출력
    console = Console()
    table = Table(
        title="GPU Memory Estimation",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("항목", style="bold")
    table.add_column("값", justify="right")

    display_rows: list[tuple[str, str]] = [
        ("model_mem_gb", "Model Weights"),
        ("gradient_mem_gb", "Gradients"),
        ("optimizer_mem_gb", "Optimizer States"),
        ("activation_mem_gb", "Activations (est.)"),
        ("total_mem_gb", "Total Estimated"),
    ]

    for key, label in display_rows:
        value = raw.get(key)
        if value is not None:
            table.add_row(label, f"{value:.2f} GB")

    table.add_row("", "")
    table.add_row(
        "Suggested GPUs",
        f"{raw.get('suggested_gpus', 'N/A')}",
    )
    table.add_row(
        "Suggested Strategy",
        f"{raw.get('suggested_strategy', 'N/A')}",
    )

    console.print(table)
