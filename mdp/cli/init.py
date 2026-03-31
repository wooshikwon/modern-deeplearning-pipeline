"""mdp init -- 새 MDP 프로젝트 디렉토리를 생성한다."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import typer


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


def _example_recipe_yaml() -> str:
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


def init_project(project_name: str) -> None:
    """새 MDP 프로젝트 스캐폴딩을 생성한다.

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

    dirs = [
        root / "configs",
        root / "recipes",
        root / "data",
        root / "checkpoints",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    (root / "configs" / "local.yaml").write_text(_default_config_yaml())
    (root / "recipes" / "example.yaml").write_text(_example_recipe_yaml())
    (root / ".gitignore").write_text(_gitignore_content())

    typer.echo(f"MDP 프로젝트 '{project_name}' 생성 완료:")
    for d in dirs:
        typer.echo(f"  {d}/")
    typer.echo(f"  {root / '.gitignore'}")
