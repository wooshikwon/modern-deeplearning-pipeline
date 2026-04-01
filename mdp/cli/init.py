"""mdp init -- 새 MDP 프로젝트 디렉토리를 생성한다."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

import typer
import yaml


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


def _build_recipe_from_catalog(
    task: str, catalog: dict[str, Any], project_name: str
) -> str:
    """TASK_PRESETS + catalog 데이터를 조합하여 Recipe YAML을 생성한다."""
    from mdp.task_taxonomy import TASK_PRESETS

    preset = TASK_PRESETS.get(task)

    # model 섹션
    lines = [
        f"# MDP Recipe -- {project_name}",
        f"name: {project_name}",
        f"task: {task}",
        "",
        "model:",
        f"  class_path: {catalog.get('class_path', '???')}",
        f"  pretrained: \"{catalog.get('pretrained', '???')}\"",
    ]

    # head 섹션 (head_builtin=false인 경우 TASK_PRESETS의 default_head 사용)
    head_builtin = catalog.get("head_builtin", True)
    if not head_builtin and preset and preset.default_head:
        lines.extend([
            "",
            "head:",
            f"  _component_: {preset.default_head}",
        ])

    # data 섹션
    lines.extend([
        "",
        "data:",
        "  dataset:",
        "    _component_: ???",
        "    root: ./data",
        "    split: train",
    ])

    # fields
    if preset and preset.required_fields:
        lines.append("  fields:")
        for field_name, field_type in preset.required_fields.items():
            lines.append(f"    {field_name}: {field_type}")

    # augmentation (이미지 관련 task)
    if preset and "image" in preset.required_fields:
        lines.extend([
            "  augmentation:",
            "    _component_: torchvision.transforms.Compose",
            "    transforms:",
            "      - _component_: torchvision.transforms.Resize",
            "        size: [224, 224]",
            "      - _component_: torchvision.transforms.ToTensor",
        ])

    # tokenizer (텍스트 관련 task)
    if preset and "text" in preset.required_fields:
        pretrained = catalog.get("pretrained", "???")
        lines.extend([
            "  tokenizer:",
            f"    pretrained: \"{pretrained}\"",
            "    max_length: 512",
        ])

    lines.extend([
        "  dataloader:",
        "    batch_size: 32",
        "    num_workers: 4",
    ])

    # training defaults
    lines.extend([
        "",
        "training:",
        "  epochs: 10",
        "  precision: fp16",
        "  gradient_accumulation_steps: 1",
    ])

    # optimizer
    lines.extend([
        "",
        "optimizer:",
        "  _component_: torch.optim.AdamW",
        "  lr: 3.0e-4",
        "  weight_decay: 0.01",
    ])

    # callbacks
    lines.extend([
        "",
        "callbacks:",
        "  - _component_: ModelCheckpoint",
        "    save_top_k: 3",
        "    monitor: val_loss",
    ])

    # metadata
    lines.extend([
        "",
        "metadata:",
        "  author: ???",
        "  description: ???",
        "",
    ])

    return "\n".join(lines)


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
          class_path: torchvision.models.resnet50
          pretrained: "hf://microsoft/resnet-50"
          init_args:
            num_classes: 10

        data:
          dataset:
            _component_: ImageFolder
            root: ./data/cifar10
            split: train
          fields:
            image: image
            label: label
          dataloader:
            batch_size: 32
            num_workers: 4

        training:
          epochs: 10
          precision: fp16
          gradient_accumulation_steps: 1

        optimizer:
          _component_: torch.optim.AdamW
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
        typer.echo(f"[error] '{project_name}' 디렉토리가 이미 존재합니다.", err=True)
        raise typer.Exit(code=1)

    # TTY 대화형 선택
    is_tty = sys.stdin.isatty()

    if is_tty and task is None:
        from mdp.task_taxonomy import TASK_PRESETS

        task_choices = list(TASK_PRESETS.keys())
        task = _interactive_select("Task 선택", task_choices)

    if task and is_tty and model is None:
        models = _list_models_for_task(task)
        if models:
            model_names = [m.get("name", "unknown") for m in models]
            model = _interactive_select("Model 선택", model_names)

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

    typer.echo(f"MDP 프로젝트 '{project_name}' 생성 완료:")
    for d in dirs:
        typer.echo(f"  {d}/")
    typer.echo(f"  {root / '.gitignore'}")
