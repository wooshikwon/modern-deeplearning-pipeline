"""SFT Trainer loss ownership contract tests."""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mdp.settings.schema import Settings
from mdp.training.trainer import Trainer
from tests.e2e.conftest import make_test_settings
from tests.e2e.datasets import ListDataLoader


def _make_settings(
    *,
    epochs: int = 1,
    loss: dict | None = None,
) -> Settings:
    settings = make_test_settings(epochs=epochs, name="trainer-loss-contract-test")
    settings.recipe.loss = loss
    return settings


def _make_batches(num_batches: int = 2, batch_size: int = 4) -> list[dict[str, Tensor]]:
    generator = torch.Generator().manual_seed(7)
    return [
        {
            "features": torch.randn(batch_size, 3, generator=generator),
            "labels": torch.randint(0, 2, (batch_size,), generator=generator),
        }
        for _ in range(num_batches)
    ]


def _make_trainer(settings: Settings, model: nn.Module) -> Trainer:
    trainer = Trainer(
        settings=settings,
        model=model,
        train_loader=ListDataLoader(_make_batches()),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False
    return trainer


class _ForwardLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3, 2)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        logits = self.classifier(batch["features"])
        return {"logits": logits, "loss": F.cross_entropy(logits, batch["labels"])}


class _ForwardLossWithWrongNativeLossModel(_ForwardLossModel):
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        logits = self.classifier(batch["features"])
        wrong_loss = logits.sum() * 0.0
        return {"logits": logits, "loss": wrong_loss}


class _NoLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3, 2)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {"logits": self.classifier(batch["features"])}


def test_loss_fn_absent_uses_forward_output_loss() -> None:
    model = _ForwardLossModel()
    trainer = _make_trainer(_make_settings(loss=None), model)

    result = trainer.train()

    assert result["stopped_reason"] == "completed"


def test_loss_fn_present_ignores_forward_output_loss(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _ForwardLossWithWrongNativeLossModel()
    trainer = _make_trainer(
        _make_settings(loss={"_component_": "torch.nn.CrossEntropyLoss"}),
        model,
    )

    with caplog.at_level(logging.WARNING, logger="mdp.training.trainer"):
        result = trainer.train()

    assert result["stopped_reason"] == "completed"
    ignored_warnings = [
        record for record in caplog.records
        if "model forward output loss will be ignored" in record.message
    ]
    assert len(ignored_warnings) == 1


def test_loss_fn_absent_without_forward_loss_errors() -> None:
    trainer = _make_trainer(_make_settings(loss=None), _NoLossModel())

    with pytest.raises(ValueError, match="No train loss found"):
        trainer.train()
