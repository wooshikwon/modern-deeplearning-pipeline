"""mdp list -- 카탈로그를 조회한다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mdp.cli.output import build_result, emit_result, is_json_mode


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
            task_folder = yaml_path.parent.name
            if task_folder not in result:
                result[task_folder] = []
            result[task_folder].append(data)

            for task in data.get("supported_tasks", []):
                if task != task_folder:
                    if task not in result:
                        result[task] = []
                    result[task].append(data)
        except Exception:
            continue

    return result


def _load_models(task_filter: str | None = None) -> list[dict[str, Any]]:
    """카탈로그 모델을 로드한다."""
    catalog_dir = Path(__file__).parent.parent / "models" / "catalog"
    models: list[dict[str, Any]] = []

    if not catalog_dir.exists():
        return models

    for yaml_path in sorted(catalog_dir.rglob("*.yaml")):
        try:
            data = yaml.safe_load(yaml_path.read_text())
            if not data:
                continue
            supported_tasks = data.get("supported_tasks", [])
            if task_filter and task_filter not in supported_tasks:
                continue
            models.append(data)
        except Exception:
            continue

    return models


# ── tasks ──


_TASK_DESCRIPTIONS = {
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


def _list_tasks() -> None:
    from mdp.task_taxonomy import TASK_PRESETS

    catalog_by_task = _load_catalog_by_task()

    if is_json_mode():
        tasks = []
        for task_name, preset in TASK_PRESETS.items():
            models = catalog_by_task.get(task_name, [])
            tasks.append({
                "name": task_name,
                "description": _TASK_DESCRIPTIONS.get(task_name, ""),
                "default_head": preset.default_head,
                "default_metric": preset.default_metric,
                "compatible_models": [m.get("name", "?") for m in models],
            })
        emit_result(build_result(command="list", what="tasks", tasks=tasks))
        return

    from rich.console import Console
    from rich.table import Table

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
            _TASK_DESCRIPTIONS.get(task_name, ""),
            preset.default_head or "-",
            models_str or "-",
        )

    console.print(table)


# ── models ──


def _list_models(task_filter: str | None = None) -> None:
    models = _load_models(task_filter)

    if is_json_mode():
        emit_result(build_result(command="list", what="models", models=models))
        return

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Model Catalog", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Family")
    table.add_column("Tasks")
    table.add_column("Head Builtin")
    table.add_column("Params (M)", justify="right")
    table.add_column("FP32 (GB)", justify="right")
    table.add_column("FP16 (GB)", justify="right")

    if not models:
        console.print("[yellow]카탈로그가 비어 있습니다.[/yellow]")
        return

    for data in models:
        supported_tasks = data.get("supported_tasks", [])
        name = data.get("name", "?")
        family = data.get("family", "-")
        tasks = ", ".join(supported_tasks)
        head_builtin = str(data.get("head_builtin", "-"))
        memory = data.get("memory", {})
        params = str(memory.get("params_m", "-"))
        fp32 = str(memory.get("fp32_gb", "-"))
        fp16 = str(memory.get("fp16_gb", "-"))
        table.add_row(name, family, tasks, head_builtin, params, fp32, fp16)

    console.print(table)


# ── callbacks ──


def _list_callbacks() -> None:
    from mdp.training.callbacks import __all__ as callback_names

    names = sorted(callback_names)

    if is_json_mode():
        emit_result(build_result(command="list", what="callbacks", callbacks=names))
        return

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Available Callbacks", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold")

    for name in names:
        table.add_row(name)

    console.print(table)


# ── strategies ──


_STRATEGIES = [
    {"name": "ddp", "strategy": "DistributedDataParallel", "description": "멀티 GPU 데이터 병렬"},
    {"name": "fsdp", "strategy": "FullyShardedDataParallel", "description": "메모리 효율적 모델 병렬"},
    {"name": "deepspeed_zero2", "strategy": "DeepSpeed ZeRO-2", "description": "ZeRO Stage 2 최적화"},
    {"name": "deepspeed_zero3", "strategy": "DeepSpeed ZeRO-3", "description": "ZeRO Stage 3 전체 분할"},
]


def _list_strategies() -> None:
    if is_json_mode():
        emit_result(build_result(command="list", what="strategies", strategies=_STRATEGIES))
        return

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Distributed Strategies", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Strategy")
    table.add_column("설명")

    for s in _STRATEGIES:
        table.add_row(s["name"], s["strategy"], s["description"])

    console.print(table)


# ── dispatch ──


_TARGET_DISPATCH = {
    "tasks": _list_tasks,
    "callbacks": _list_callbacks,
    "strategies": _list_strategies,
}

_VALID_TARGETS = ["models"] + list(_TARGET_DISPATCH.keys())


def run_list(target: str, task_filter: str | None = None) -> None:
    """target에 해당하는 카탈로그를 조회한다."""
    import typer

    if target == "models":
        _list_models(task_filter=task_filter)
        return

    handler = _TARGET_DISPATCH.get(target)
    if handler is None:
        msg = f"알 수 없는 target: '{target}'. 사용 가능: {_VALID_TARGETS}"
        if is_json_mode():
            from mdp.cli.output import build_error
            emit_result(build_error(command="list", error_type="ValueError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    handler()
