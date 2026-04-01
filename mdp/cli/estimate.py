"""mdp estimate -- GPU 메모리 사용량을 추정한다."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_estimate(recipe_path: str) -> None:
    """Recipe YAML을 파싱하여 GPU 메모리 사용량을 추정한다.

    SettingsFactory.for_estimation()으로 Recipe만 로딩하여 Settings를 조립한다.
    JSON 모드와 Rich Table 출력을 모두 지원한다.
    """
    from rich.console import Console
    from rich.table import Table

    import typer

    from mdp.cli.output import build_error, build_result, emit_result, is_json_mode
    from mdp.utils.estimator import MemoryEstimator
    from mdp.settings.factory import SettingsFactory

    try:
        settings = SettingsFactory().for_estimation(recipe_path)
    except Exception as e:
        if is_json_mode():
            emit_result(build_error(
                command="estimate",
                error_type="settings_error",
                message=str(e),
            ))
            raise typer.Exit(code=1)
        typer.echo(f"[error] Settings 로딩 실패: {e}", err=True)
        raise typer.Exit(code=1)

    result = MemoryEstimator().estimate(settings)

    if is_json_mode():
        emit_result(build_result(command="estimate", **result))
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

    # 표시 순서와 라벨 매핑
    display_rows: list[tuple[str, str]] = [
        ("model_mem_gb", "Model Weights"),
        ("gradient_mem_gb", "Gradients"),
        ("optimizer_mem_gb", "Optimizer States"),
        ("activation_mem_gb", "Activations (est.)"),
        ("total_mem_gb", "Total Estimated"),
    ]

    for key, label in display_rows:
        value = result.get(key)
        if value is not None:
            table.add_row(label, f"{value:.2f} GB")

    # GPU 추천
    table.add_row("", "")  # 구분선
    table.add_row(
        "Suggested GPUs",
        f"{result.get('suggested_gpus', 'N/A')}",
    )
    table.add_row(
        "Suggested Strategy",
        f"{result.get('suggested_strategy', 'N/A')}",
    )

    console.print(table)
