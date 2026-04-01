"""mdp list -- 카탈로그를 조회한다."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import yaml


def _load_catalog_by_task() -> dict[str, list[dict[str, Any]]]:
    """catalog YAMLs를 task별로 그룹화하여 반환한다."""
    catalog_dir = Path(__file__).parent.parent / "models" / "catalog"
    result: dict[str, list[dict[str, Any]]] = {}

    if not catalog_dir.exists():
        return result

    for yaml_path in sorted(catalog_dir.rglob("*.yaml")):
        try:
            data = yaml.safe_load(yaml_path.read_text())
            if not data:
                continue
            # task 폴더명
            task_folder = yaml_path.parent.name
            if task_folder not in result:
                result[task_folder] = []
            result[task_folder].append(data)

            # supported_tasks에서 추가 task에도 등록
            for task in data.get("supported_tasks", []):
                if task != task_folder:
                    if task not in result:
                        result[task] = []
                    result[task].append(data)
        except Exception:
            continue

    return result


def _list_tasks() -> None:
    """TASK_PRESETS에서 동적으로 태스크 목록을 출력한다."""
    from rich.console import Console
    from rich.table import Table

    from mdp.task_taxonomy import TASK_PRESETS

    _DESCRIPTIONS = {
        "image_classification": "이미지를 보고 클래스 판별",
        "object_detection": "이미지에서 물체 위치+클래스 탐지",
        "semantic_segmentation": "이미지 픽셀별 클래스 분류",
        "text_classification": "텍스트를 보고 클래스 판별",
        "token_classification": "토큰별 태그 판별",
        "text_generation": "텍스트 생성",
        "seq2seq": "입력 텍스트를 다른 텍스트로 변환",
        "image_generation": "이미지 생성",
        "feature_extraction": "임베딩 추출",
    }

    catalog_by_task = _load_catalog_by_task()

    console = Console()
    table = Table(title="Supported Tasks", show_header=True, header_style="bold cyan")
    table.add_column("Task", style="bold", no_wrap=True)
    table.add_column("설명")
    table.add_column("Head")
    table.add_column("호환 모델", max_width=50)

    for task_name, preset in TASK_PRESETS.items():
        models = catalog_by_task.get(task_name, [])
        models_str = ", ".join(m.get("name", "?") for m in models[:5])
        if len(models) > 5:
            models_str += f" (+{len(models) - 5})"
        table.add_row(
            task_name,
            _DESCRIPTIONS.get(task_name, ""),
            preset.default_head or "-",
            models_str or "-",
        )

    console.print(table)


def _list_models(task_filter: str | None = None) -> None:
    """models/catalog/ 하위 YAML을 Rich Table로 출력한다."""
    from rich.console import Console
    from rich.table import Table

    catalog_dir = Path(__file__).parent.parent / "models" / "catalog"
    console = Console()
    table = Table(title="Model Catalog", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Family")
    table.add_column("Tasks")
    table.add_column("Head Builtin")
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

            supported_tasks = data.get("supported_tasks", [])
            if task_filter and task_filter not in supported_tasks:
                continue

            name = data.get("name", yaml_path.stem)
            family = data.get("family", "-")
            tasks = ", ".join(supported_tasks)
            head_builtin = str(data.get("head_builtin", "-"))
            memory = data.get("memory", {})
            params = str(memory.get("params_m", "-"))
            fp32 = str(memory.get("fp32_gb", "-"))
            fp16 = str(memory.get("fp16_gb", "-"))
            table.add_row(name, family, tasks, head_builtin, params, fp32, fp16)
        except Exception:
            continue

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

    from mdp.utils.job_manager import JobManager

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
    "tasks": _list_tasks,
    "callbacks": _list_callbacks,
    "strategies": _list_strategies,
    "jobs": _list_jobs,
}

_VALID_TARGETS = ["models"] + list(_TARGET_DISPATCH.keys())


def run_list(target: str, task_filter: str | None = None) -> None:
    """target에 해당하는 카탈로그를 조회한다.

    Parameters
    ----------
    target:
        ``models`` | ``tasks`` | ``callbacks`` | ``strategies`` | ``jobs``
    task_filter:
        models 조회 시 task로 필터링.
    """
    import typer

    if target == "models":
        _list_models(task_filter=task_filter)
        return

    handler = _TARGET_DISPATCH.get(target)
    if handler is None:
        typer.echo(
            f"[error] 알 수 없는 target: '{target}'. "
            f"사용 가능: {_VALID_TARGETS}",
            err=True,
        )
        raise typer.Exit(code=1)

    handler()
