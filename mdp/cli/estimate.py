"""mdp estimate -- GPU 메모리 사용량을 추정한다."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def run_estimate(recipe_path: str) -> None:
    """Recipe YAML을 파싱하여 GPU 메모리 사용량을 추정, Rich Table로 출력한다.

    MemoryEstimator는 Settings 객체를 요구하므로, 최소한의 Config를 임시 생성하여
    SettingsFactory.for_training()으로 Settings를 조립한다.
    """
    from rich.console import Console
    from rich.table import Table

    import typer

    from mdp.compute.estimator import MemoryEstimator
    from mdp.settings.factory import SettingsFactory

    # 최소한의 Config 생성 (estimate에는 compute target만 필요)
    minimal_config = {"compute": {"target": "local"}}
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp:
            yaml.dump(minimal_config, tmp)
            tmp_config_path = tmp.name

        settings = SettingsFactory().for_training(recipe_path, tmp_config_path)
    except Exception as e:
        typer.echo(f"[error] Settings 로딩 실패: {e}", err=True)
        raise typer.Exit(code=1)
    finally:
        Path(tmp_config_path).unlink(missing_ok=True)

    result = MemoryEstimator().estimate(settings)

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
