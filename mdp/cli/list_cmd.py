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
        from mdp.cli.schemas import ListTasksResult
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
        result = ListTasksResult(tasks=tasks)
        emit_result(build_result(command="list", **result.model_dump()))
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
        from mdp.cli.schemas import ListModelsResult
        result = ListModelsResult(models=models)
        emit_result(build_result(command="list", **result.model_dump()))
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


def _load_aliases_by_category() -> dict[str, dict[str, str]]:
    """aliases.yaml을 카테고리별 dict로 로드한다 ({category: {alias: class_path}})."""
    aliases_path = Path(__file__).parent.parent / "aliases.yaml"
    if not aliases_path.exists():
        return {}
    return yaml.safe_load(aliases_path.read_text()) or {}


def _classify(class_path: str) -> str:
    """class_path를 import해 콜백 타입을 분류한다.

    Returns
    -------
    "[Int]"   : BaseInterventionCallback 서브클래스 (출력을 바꾸는 개입)
    "[Obs]"   : BaseInferenceCallback 직계 서브클래스 (읽기 전용 관측)
    "[Train]" : BaseCallback 서브클래스 (학습 콜백)
    "[?]"     : import 실패 또는 분류 불가
    """
    try:
        import importlib
        mod_name, cls_name = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)
        from mdp.callbacks.base import BaseInterventionCallback, BaseInferenceCallback
        if issubclass(cls, BaseInterventionCallback):
            return "[Int]"
        if issubclass(cls, BaseInferenceCallback):
            return "[Obs]"
        return "[Train]"
    except Exception:
        return "[?]"


def _classify_to_type_str(class_path: str) -> str:
    """JSON 모드용 type 문자열을 반환한다."""
    label = _classify(class_path)
    return {
        "[Int]": "intervention",
        "[Obs]": "observational",
        "[Train]": "training",
    }.get(label, "unknown")


def _list_callbacks() -> None:
    """Recipe에 `_component_:`로 쓸 수 있는 콜백 alias 목록을 보여준다.

    aliases.yaml의 callback 카테고리가 권위 있는 소스 — agent는 여기 표시된
    alias를 그대로 Recipe에 사용해야 한다 (클래스명을 직접 쓰면 ValueError).
    Type 컬럼: [Int] = intervention (출력 수정), [Obs] = observational (읽기 전용),
    [Train] = training callback.
    """
    aliases = _load_aliases_by_category().get("callback", {})
    items = sorted(aliases.items())  # [(alias, class_path), ...]

    if is_json_mode():
        from mdp.cli.schemas import ListCallbacksResult
        # 풀 정보로 표시 — 단순 리스트가 아닌 alias→class_path 매핑, type 포함
        result = ListCallbacksResult(callbacks=[a for a, _ in items])
        emit_result(build_result(
            command="list",
            **result.model_dump(),
            callback_aliases=[
                {
                    "alias": a,
                    "class_path": p,
                    "class_name": p.rsplit(".", 1)[-1],
                    "type": _classify_to_type_str(p),
                }
                for a, p in items
            ],
        ))
        return

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Available Callbacks", show_header=True, header_style="bold cyan")
    table.add_column("Alias (use in _component_)", style="bold")
    table.add_column("Type", no_wrap=True)
    table.add_column("Class")
    table.add_column("Module")

    for alias, class_path in items:
        module, _, cls_name = class_path.rpartition(".")
        type_label = _classify(class_path)
        table.add_row(alias, type_label, cls_name, module)

    console.print(table)


# ── strategies ──


_STRATEGY_DESCRIPTIONS = {
    "ddp": "멀티 GPU 데이터 병렬",
    "fsdp": "메모리 효율적 모델 병렬",
    "deepspeed": "UNSUPPORTED: fail-fast until DeepSpeed engine-contract is implemented",
    "deepspeed_zero2": "UNSUPPORTED: fail-fast until DeepSpeed engine-contract is implemented",
    "deepspeed_zero3": "UNSUPPORTED: fail-fast until DeepSpeed engine-contract is implemented",
    "auto": "GPU 수에 따라 ddp/fsdp 자동 선택",
    # CamelCase 변형도 동일한 설명으로 처리
    "DDPStrategy": "멀티 GPU 데이터 병렬",
    "FSDPStrategy": "메모리 효율적 모델 병렬",
    "DeepSpeedStrategy": "UNSUPPORTED: fail-fast until DeepSpeed engine-contract is implemented",
    "DeepSpeedZeRO2": "UNSUPPORTED: fail-fast until DeepSpeed engine-contract is implemented",
    "DeepSpeedZeRO3": "UNSUPPORTED: fail-fast until DeepSpeed engine-contract is implemented",
}


def _list_strategies() -> None:
    """Recipe/Config의 distributed.strategy 또는 _component_에 쓸 수 있는 alias 목록.

    aliases.yaml의 strategy 카테고리가 권위 있는 소스. 소문자 단축형(ddp, fsdp 등)과
    `auto`를 포함한다.
    """
    aliases = _load_aliases_by_category().get("strategy", {})
    items = sorted(aliases.items())

    if is_json_mode():
        from mdp.cli.schemas import ListStrategiesResult
        strategies = [
            {
                "name": alias,
                "strategy": class_path.rsplit(".", 1)[-1],
                "description": _STRATEGY_DESCRIPTIONS.get(alias, ""),
            }
            for alias, class_path in items
        ]
        result = ListStrategiesResult(strategies=strategies)
        emit_result(build_result(command="list", **result.model_dump()))
        return

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Distributed Strategies", show_header=True, header_style="bold cyan")
    table.add_column("Alias", style="bold")
    table.add_column("Strategy")
    table.add_column("설명")

    for alias, class_path in items:
        cls_name = class_path.rsplit(".", 1)[-1]
        table.add_row(alias, cls_name, _STRATEGY_DESCRIPTIONS.get(alias, ""))

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
