"""mdp init -- 새 MDP 프로젝트 디렉토리를 생성한다."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

import typer
import yaml

from mdp.cli.output import build_error, build_result, emit_result, is_json_mode


def _find_catalog_entry(model_name: str) -> dict[str, Any] | None:
    """catalog YAMLs에서 이름으로 모델을 검색한다."""
    catalog_dir = Path(__file__).parent.parent / "models" / "catalog"
    if not catalog_dir.exists():
        return None

    for yaml_path in catalog_dir.rglob("*.yaml"):
        try:
            data = yaml.safe_load(yaml_path.read_text())
            if not data:
                continue
            if data.get("name") == model_name:
                data["_task_folder"] = yaml_path.parent.name
                return data
        except Exception:
            continue
    return None


def _list_models_for_task(task: str) -> list[dict[str, Any]]:
    """특정 task 폴더에 있는 모델 목록을 반환한다."""
    catalog_dir = Path(__file__).parent.parent / "models" / "catalog"
    models: list[dict[str, Any]] = []

    if not catalog_dir.exists():
        return models

    # task 폴더에서 직접 검색
    task_dir = catalog_dir / task
    if task_dir.is_dir():
        for yaml_path in sorted(task_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(yaml_path.read_text())
                if data:
                    models.append(data)
            except Exception:
                continue

    # 다른 폴더에서 supported_tasks에 포함된 모델도 검색
    for yaml_path in sorted(catalog_dir.rglob("*.yaml")):
        if yaml_path.parent.name == task:
            continue  # 이미 수집됨
        try:
            data = yaml.safe_load(yaml_path.read_text())
            if not data:
                continue
            if task in data.get("supported_tasks", []):
                if data not in models:
                    models.append(data)
        except Exception:
            continue

    return models


_TASK_DEFAULT_LABEL_STRATEGY: dict[str, str] = {
    "image_classification": "copy",
    "object_detection": "unlabeled",
    "semantic_segmentation": "unlabeled",
    "text_classification": "copy",
    "token_classification": "align",
    "text_generation": "causal",
    "seq2seq": "seq2seq",
    "image_generation": "unlabeled",
    "feature_extraction": "unlabeled",
}


def _build_recipe_from_catalog(
    task: str, catalog: dict[str, Any], project_name: str,
) -> str:
    """TASK_PRESETS + catalog 데이터를 조합하여 Recipe YAML을 생성한다."""
    import yaml as _yaml
    from mdp.task_taxonomy import TASK_PRESETS

    preset = TASK_PRESETS.get(task)
    name = catalog.get("name", "model")
    class_path = catalog.get("class_path", "???")
    head_builtin = catalog.get("head_builtin", False)
    pretrained = catalog.get("pretrained_sources", [""])[0]
    input_spec = catalog.get("input_spec", {})

    # pretrained URI에서 tokenizer pretrained 파생
    tokenizer_pretrained = pretrained.replace("hf://", "").replace("timm://", "")

    recipe: dict[str, Any] = {
        "name": f"{name}-{project_name}",
        "task": task,
        "model": {
            "_component_": class_path,
            "pretrained": pretrained,
        },
    }

    # head: head_builtin이 아닌 경우만 추가
    if not head_builtin and preset and preset.default_head:
        head_config: dict[str, Any] = {"_component_": preset.default_head}
        catalog_head = catalog.get("default_head", {})
        if catalog_head.get("hidden_dim"):
            head_config["hidden_dim"] = catalog_head["hidden_dim"]
        if catalog_head.get("dropout") is not None:
            head_config["dropout"] = catalog_head["dropout"]
        head_config["num_classes"] = "???"
        recipe["head"] = head_config

    # data 섹션 — _component_ 패턴
    data: dict[str, Any] = {
        "dataset": {
            "_component_": "mdp.data.datasets.HuggingFaceDataset",
            "source": "???",
            "split": "train",
        },
    }

    # collator 결정: task에 따라 적절한 collator 선택
    label_strategy = _TASK_DEFAULT_LABEL_STRATEGY.get(task, "unlabeled")
    if label_strategy == "causal":
        collator_component = "mdp.data.collators.CausalLMCollator"
    elif label_strategy == "seq2seq":
        collator_component = "mdp.data.collators.Seq2SeqCollator"
    else:
        collator_component = "mdp.data.collators.ClassificationCollator"

    collator: dict[str, Any] = {"_component_": collator_component}
    if tokenizer_pretrained:
        collator["tokenizer"] = tokenizer_pretrained
    else:
        collator["tokenizer"] = "gpt2"
    data["collator"] = collator

    data["dataloader"] = {"batch_size": 32, "num_workers": 4}
    recipe["data"] = data

    recipe["training"] = {"epochs": 3, "precision": "bf16"}
    recipe["optimizer"] = {"_component_": "AdamW", "lr": 2e-5, "weight_decay": 0.01}
    recipe["callbacks"] = [
        {"_component_": "ModelCheckpoint", "save_top_k": 3, "monitor": "val_loss"},
    ]
    recipe["metadata"] = {"author": "???", "description": "???"}

    return _yaml.dump(recipe, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _interactive_select(title: str, choices: list[str]) -> str:
    """Rich 테이블로 선택지를 표시하고 번호 입력을 받는다."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("#", style="bold", justify="right")
    table.add_column("Name")

    for i, choice in enumerate(choices, 1):
        table.add_row(str(i), choice)

    console.print(table)

    while True:
        try:
            raw = input("번호를 입력하세요: ").strip()
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
            console.print(f"[red]1~{len(choices)} 범위의 번호를 입력하세요.[/red]")
        except (ValueError, EOFError):
            console.print("[red]유효한 번호를 입력하세요.[/red]")


def _default_config_yaml() -> str:
    """로컬 기본 Config YAML."""
    return dedent("""\
        # MDP Config -- 인프라 설정
        environment:
          name: local

        compute:
          target: local
          gpus: auto

        mlflow:
          tracking_uri: ./mlruns
          experiment_name: default

        storage:
          checkpoint_dir: ./checkpoints
          output_dir: ./outputs

        job:
          resume: auto
          max_retries: 0
    """)


def _default_recipe_yaml() -> str:
    """ResNet 이미지 분류 예제 Recipe YAML."""
    return dedent("""\
        # MDP Recipe -- 실험 정의서
        name: resnet50-cifar10
        task: image_classification

        model:
          _component_: torchvision.models.resnet50
          pretrained: "hf://microsoft/resnet-50"
          num_classes: 10

        data:
          dataset:
            _component_: mdp.data.datasets.HuggingFaceDataset
            source: ./data/cifar10
            split: train
          collator:
            _component_: mdp.data.collators.ClassificationCollator
            tokenizer: gpt2
          dataloader:
            batch_size: 32
            num_workers: 4

        training:
          epochs: 10
          precision: fp16
          gradient_accumulation_steps: 1

        optimizer:
          _component_: AdamW
          lr: 3.0e-4
          weight_decay: 0.01

        evaluation:
          metrics:
            - accuracy
            - f1

        callbacks:
          - _component_: ModelCheckpoint
            save_top_k: 3
            monitor: val_loss

        metadata:
          author: your-name
          description: "ResNet50 CIFAR-10 분류 예제"
    """)


def _gitignore_content() -> str:
    """프로젝트 .gitignore."""
    return dedent("""\
        # Data & artifacts
        data/
        checkpoints/
        mlruns/

        # Python
        __pycache__/
        *.pyc
        *.pyo
        *.egg-info/
        dist/
        build/

        # IDE
        .vscode/
        .idea/

        # OS
        .DS_Store
        Thumbs.db
    """)


def init_project(
    project_name: str,
    task: str | None = None,
    model: str | None = None,
) -> None:
    """새 MDP 프로젝트 스캐폴딩을 생성한다.

    TTY 환경에서 task/model이 지정되지 않으면 대화형 선택을 수행한다.

    생성 구조::

        {project_name}/
        ├── configs/local.yaml
        ├── recipes/example.yaml
        ├── data/
        ├── checkpoints/
        └── .gitignore
    """
    root = Path(project_name)

    if root.exists():
        msg = f"'{project_name}' 디렉토리가 이미 존재합니다."
        if is_json_mode():
            emit_result(build_error(command="init", error_type="ValidationError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    # TTY 대화형 선택 (JSON 모드에서는 비활성)
    is_tty = sys.stdin.isatty() and not is_json_mode()

    if is_tty and task is None:
        from mdp.task_taxonomy import TASK_PRESETS

        task_choices = list(TASK_PRESETS.keys())
        task = _interactive_select("Task 선택", task_choices)

    if task and is_tty and model is None:
        models = _list_models_for_task(task)
        if models:
            model_names = [m.get("name", "unknown") for m in models]
            model = _interactive_select("Model 선택", model_names)

    try:
        # 디렉토리 생성
        dirs = [
            root / "configs",
            root / "recipes",
            root / "data",
            root / "checkpoints",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        # Config 생성
        (root / "configs" / "local.yaml").write_text(_default_config_yaml())

        # Recipe 생성
        if task and model:
            catalog = _find_catalog_entry(model)
            if catalog:
                recipe_content = _build_recipe_from_catalog(task, catalog, project_name)
            else:
                recipe_content = _default_recipe_yaml()
        else:
            recipe_content = _default_recipe_yaml()

        (root / "recipes" / "example.yaml").write_text(recipe_content)
        (root / ".gitignore").write_text(_gitignore_content())

        if not is_json_mode():
            typer.echo(f"MDP 프로젝트 '{project_name}' 생성 완료:")
            for d in dirs:
                typer.echo(f"  {d}/")
            typer.echo(f"  {root / '.gitignore'}")

        if is_json_mode():
            emit_result(build_result(
                command="init",
                project_name=project_name,
                project_dir=str(root.resolve()),
                task=task,
                model=model,
            ))

    except typer.Exit:
        raise
    except Exception as e:
        if is_json_mode():
            emit_result(build_error(
                command="init", error_type="RuntimeError", message=str(e),
            ))
            raise typer.Exit(code=1)
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(code=1)
