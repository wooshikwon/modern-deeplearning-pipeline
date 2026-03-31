"""mdp list -- 카탈로그를 조회한다."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import yaml


def _list_models() -> None:
    """models/catalog/ 하위 YAML을 Rich Table로 출력한다."""
    from rich.console import Console
    from rich.table import Table

    catalog_dir = Path(__file__).parent.parent / "models" / "catalog"
    console = Console()
    table = Table(title="Model Catalog", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Family")
    table.add_column("Tasks")
    table.add_column("Params (M)", justify="right")
    table.add_column("FP32 (GB)", justify="right")
    table.add_column("FP16 (GB)", justify="right")

    if not catalog_dir.exists():
        console.print("[yellow]카탈로그 디렉토리가 없습니다.[/yellow]")
        return

    for yaml_path in sorted(catalog_dir.rglob("*.yaml")):
        try:
            data = yaml.safe_load(yaml_path.read_text())
            if not data:
                continue
            name = data.get("name", yaml_path.stem)
            family = data.get("family", "-")
            tasks = ", ".join(data.get("supported_tasks", []))
            memory = data.get("memory", {})
            params = str(memory.get("params_m", "-"))
            fp32 = str(memory.get("fp32_gb", "-"))
            fp16 = str(memory.get("fp16_gb", "-"))
            table.add_row(name, family, tasks, params, fp32, fp16)
        except Exception:
            continue

    console.print(table)


def _list_tasks() -> None:
    """지원하는 태스크 목록을 출력한다."""
    from rich.console import Console
    from rich.table import Table

    tasks = [
        ("image_classification", "이미지 분류"),
        ("object_detection", "객체 탐지"),
        ("feature_extraction", "특징 추출"),
        ("causal_lm", "인과적 언어 모델링"),
        ("text_generation", "텍스트 생성"),
        ("seq2seq", "시퀀스-투-시퀀스"),
        ("text_classification", "텍스트 분류"),
        ("token_classification", "토큰 분류"),
    ]

    console = Console()
    table = Table(title="Supported Tasks", show_header=True, header_style="bold cyan")
    table.add_column("Task", style="bold")
    table.add_column("설명")

    for task_name, description in tasks:
        table.add_row(task_name, description)

    console.print(table)


def _list_callbacks() -> None:
    """등록된 콜백 목록을 출력한다."""
    from rich.console import Console
    from rich.table import Table

    from mdp.training.callbacks import __all__ as callback_names

    console = Console()
    table = Table(title="Available Callbacks", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold")

    for name in sorted(callback_names):
        table.add_row(name)

    console.print(table)


def _list_strategies() -> None:
    """지원하는 분산 전략 목록을 출력한다."""
    from rich.console import Console
    from rich.table import Table

    strategies = [
        ("ddp", "DistributedDataParallel", "멀티 GPU 데이터 병렬"),
        ("fsdp", "FullyShardedDataParallel", "메모리 효율적 모델 병렬"),
        ("deepspeed", "DeepSpeed ZeRO", "ZeRO Stage 1/2 최적화"),
        ("deepspeed_zero3", "DeepSpeed ZeRO-3", "ZeRO Stage 3 전체 분할"),
    ]

    console = Console()
    table = Table(title="Distributed Strategies", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Strategy")
    table.add_column("설명")

    for name, strategy, description in strategies:
        table.add_row(name, strategy, description)

    console.print(table)


def _list_jobs() -> None:
    """JobManager에서 작업 목록을 조회하여 Rich Table로 출력한다."""
    from rich.console import Console
    from rich.table import Table

    from mdp.compute.job_manager import JobManager

    console = Console()

    try:
        manager = JobManager()
        jobs = manager.list_jobs()
        manager.close()
    except Exception as e:
        console.print(f"[red]작업 목록 조회 실패: {e}[/red]")
        return

    table = Table(title="Jobs", show_header=True, header_style="bold cyan")
    table.add_column("Job ID", style="bold", max_width=16)
    table.add_column("Executor")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Error", max_width=40)

    if not jobs:
        console.print("[dim]등록된 작업이 없습니다.[/dim]")
        return

    for job in jobs:
        created = datetime.datetime.fromtimestamp(job.created_at).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        status_style = {
            "running": "yellow",
            "completed": "green",
            "failed": "red",
            "stopped": "dim",
        }.get(job.status, "")
        status_text = f"[{status_style}]{job.status}[/{status_style}]" if status_style else job.status

        table.add_row(
            job.job_id[:16],
            job.executor,
            status_text,
            created,
            job.error or "",
        )

    console.print(table)


_TARGET_DISPATCH = {
    "models": _list_models,
    "tasks": _list_tasks,
    "callbacks": _list_callbacks,
    "strategies": _list_strategies,
    "jobs": _list_jobs,
}

_VALID_TARGETS = list(_TARGET_DISPATCH.keys())


def run_list(target: str) -> None:
    """target에 해당하는 카탈로그를 조회한다.

    Parameters
    ----------
    target:
        ``models`` | ``tasks`` | ``callbacks`` | ``strategies`` | ``jobs``
    """
    import typer

    handler = _TARGET_DISPATCH.get(target)
    if handler is None:
        typer.echo(
            f"[error] 알 수 없는 target: '{target}'. "
            f"사용 가능: {_VALID_TARGETS}",
            err=True,
        )
        raise typer.Exit(code=1)

    handler()
