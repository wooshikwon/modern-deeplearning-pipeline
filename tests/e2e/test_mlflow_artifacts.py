"""MLflow artifact 등록 + run_id 기반 흐름 테스트."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import torch

from mdp.training.trainer import Trainer
from tests.e2e.conftest import make_test_settings
from tests.e2e.datasets import ListDataLoader, make_vision_batches
from tests.e2e.models import TinyVisionModel


def _train_one_run(tmp_path: Path) -> str:
    """1 epoch 학습을 수행하고 MLflow run_id를 반환한다."""
    ckpt_dir = tmp_path / "checkpoints"
    mlruns_dir = tmp_path / "mlruns"

    settings = make_test_settings(
        epochs=1,
        checkpoint_dir=str(ckpt_dir),
        name="mlflow-artifact-test",
    )
    # MLflow 설정을 Settings를 통해 주입 (Trainer가 이 값을 읽음)
    settings.config.mlflow.tracking_uri = str(mlruns_dir)
    settings.config.mlflow.experiment_name = "test-artifacts"

    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    batches = make_vision_batches(3, 4, 2, 8)
    val_batches = make_vision_batches(2, 4, 2, 8, seed=99)

    trainer = Trainer(
        settings=settings,
        model=model,
        train_loader=ListDataLoader(batches),
        val_loader=ListDataLoader(val_batches),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False

    from mdp.training.callbacks.checkpoint import ModelCheckpoint
    trainer.callbacks.append(
        ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", mode="min")
    )

    trainer.train()

    # 가장 최근 run 탐색
    mlflow.set_tracking_uri(str(mlruns_dir))
    client = mlflow.MlflowClient()
    for exp in client.search_experiments():
        runs = client.search_runs(exp.experiment_id, max_results=1, order_by=["start_time DESC"])
        if runs:
            return runs[0].info.run_id
    return ""


def test_train_logs_checkpoint_artifact(tmp_path: Path) -> None:
    """학습 후 MLflow에 checkpoint artifact가 등록되는지."""
    run_id = _train_one_run(tmp_path)
    assert run_id, "run_id를 찾을 수 없습니다"

    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run_id)]

    assert "checkpoint" in artifacts, f"checkpoint artifact 없음. artifacts: {artifacts}"


def test_train_logs_model_artifact(tmp_path: Path) -> None:
    """학습 후 MLflow에 model (서빙 가능) artifact가 등록되는지."""
    run_id = _train_one_run(tmp_path)
    assert run_id

    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run_id)]

    assert "model" in artifacts, f"model artifact 없음. artifacts: {artifacts}"


def test_checkpoint_artifact_has_recipe(tmp_path: Path) -> None:
    """checkpoint artifact에 recipe.yaml이 포함되어 있는지."""
    run_id = _train_one_run(tmp_path)
    assert run_id

    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    ckpt_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="checkpoint",
    )

    assert (Path(ckpt_path) / "recipe.yaml").exists(), "recipe.yaml이 없습니다"
