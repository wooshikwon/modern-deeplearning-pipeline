"""E2E tests for training callbacks (EarlyStopping, Checkpoint, EMA)."""

from __future__ import annotations

import torch
from safetensors.torch import load_file, save_file

from mdp.training.callbacks.early_stopping import EarlyStopping
from mdp.training.callbacks.ema import EMACallback
from tests.e2e.models import TinyVisionModel


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------


def test_early_stopping_triggers() -> None:
    """3 consecutive worsening val_loss values trigger should_stop with patience=3."""
    es = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    # First call sets best_value
    es.on_validation_end(epoch=0, metrics={"val_loss": 1.0})
    assert not es.should_stop

    # 3 worsening calls
    es.on_validation_end(epoch=1, metrics={"val_loss": 1.1})
    assert not es.should_stop
    es.on_validation_end(epoch=2, metrics={"val_loss": 1.2})
    assert not es.should_stop
    es.on_validation_end(epoch=3, metrics={"val_loss": 1.3})
    assert es.should_stop


def test_early_stopping_resets() -> None:
    """Counter resets when an improving value is observed."""
    es = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    es.on_validation_end(epoch=0, metrics={"val_loss": 1.0})

    # 2 worsening
    es.on_validation_end(epoch=1, metrics={"val_loss": 1.1})
    es.on_validation_end(epoch=2, metrics={"val_loss": 1.2})
    assert es.counter == 2

    # Improvement resets counter
    es.on_validation_end(epoch=3, metrics={"val_loss": 0.5})
    assert es.counter == 0
    assert not es.should_stop

    # 2 more worsening -- still not triggered
    es.on_validation_end(epoch=4, metrics={"val_loss": 0.6})
    es.on_validation_end(epoch=5, metrics={"val_loss": 0.7})
    assert es.counter == 2
    assert not es.should_stop


# ---------------------------------------------------------------------------
# Checkpoint: safetensors save & reload
# ---------------------------------------------------------------------------


def test_checkpoint_save_and_load(tmp_path) -> None:
    """Save TinyVisionModel state_dict with safetensors, reload, verify params match."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    state_dict = model.state_dict()

    # safetensors requires all tensors to be contiguous
    safe_dict = {k: v.contiguous() for k, v in state_dict.items()}
    save_path = tmp_path / "model.safetensors"
    save_file(safe_dict, str(save_path))

    assert save_path.exists()
    assert save_path.stat().st_size > 0

    loaded = load_file(str(save_path))

    assert set(loaded.keys()) == set(state_dict.keys())
    for key in state_dict:
        torch.testing.assert_close(loaded[key], state_dict[key])


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


def test_ema_weights_differ() -> None:
    """After EMA update with modified model params, shadow differs from current."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    ema = EMACallback(decay=0.99)

    # Initialise shadow params
    ema.on_train_start(model=model)
    assert len(ema._shadow_params) > 0

    # Snapshot originals
    original_shadows = [s.clone() for s in ema._shadow_params]

    # Simulate a training step: perturb model params
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.5)

    # EMA update
    ema.on_batch_end(step=0)

    # Shadow params should have moved towards the new model params
    # but should NOT be identical to either original or current params
    for shadow, orig in zip(ema._shadow_params, original_shadows):
        assert not torch.equal(shadow, orig), "Shadow should have moved from original"

    trainable = [p for p in model.parameters() if p.requires_grad]
    for shadow, param in zip(ema._shadow_params, trainable):
        assert not torch.equal(
            shadow, param.data
        ), "Shadow should differ from current params (decay < 1)"
