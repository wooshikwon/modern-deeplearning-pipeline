"""E2E tests for vision, detection, and segmentation pipelines.

All tests use manual training loops with AdamW (lr=1e-3), 3 epochs,
CPU device, 8x8 images, and batch size 4.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mdp.models.heads.classification import ClassificationHead
from mdp.models.heads.detection import DetectionHead
from mdp.models.heads.segmentation import SegmentationHead

from tests.e2e.datasets import (
    ListDataLoader,
    make_feature_map_batches,
    make_segmentation_batches,
    make_vision_batches,
)
from tests.e2e.models import TinyFeatureMapModel, TinyVisionModel


# ── Helpers ──


def _train_loop(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    loss_fn,
    epochs: int = 3,
    lr: float = 1e-3,
) -> list[float]:
    """Run a manual training loop, returning per-epoch average losses."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    epoch_losses: list[float] = []

    for _epoch in range(epochs):
        total_loss = 0.0
        for batch in batches:
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_losses.append(total_loss / len(batches))

    return epoch_losses


# ── Tests ──


class TestVisionClassification:
    """Vision classification training and head replacement tests."""

    def test_vision_classification_training(self, device: torch.device) -> None:
        """Train TinyVisionModel for 3 epochs; loss must decrease."""
        model = TinyVisionModel(num_classes=2, hidden_dim=16).to(device)
        batches = make_vision_batches(
            num_batches=5, batch_size=4, num_classes=2, image_size=8
        )

        def loss_fn(m, batch):
            return m(batch)["loss"]

        losses = _train_loop(model, batches, loss_fn, epochs=3)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_vision_head_replacement(self, device: torch.device) -> None:
        """Replace TinyVisionModel head with ClassificationHead(16,10)."""
        model = TinyVisionModel(num_classes=2, hidden_dim=16).to(device)

        # Replace head with a ClassificationHead targeting 10 classes
        new_head = ClassificationHead(
            num_classes=10, hidden_dim=16, dropout=0.0
        )
        model.head = new_head.to(device)

        batch = make_vision_batches(
            num_batches=1, batch_size=4, num_classes=10, image_size=8
        )[0]

        model.eval()
        with torch.no_grad():
            outputs = model.forward(batch)

        logits = outputs["logits"]
        assert logits.shape == (4, 10), f"Expected (4, 10), got {logits.shape}"


class TestDetection:
    """Detection head forward pass tests."""

    def test_detection_head_forward(self, device: torch.device) -> None:
        """TinyFeatureMapModel(16) + DetectionHead(16,5,3) -> (B,30,H,W)."""
        backbone = TinyFeatureMapModel(out_channels=16).to(device)
        det_head = DetectionHead(
            in_channels=16, num_classes=5, num_anchors=3
        ).to(device)

        batch = make_feature_map_batches(
            num_batches=1, batch_size=4, image_size=8
        )[0]

        backbone.eval()
        det_head.eval()
        with torch.no_grad():
            features = backbone.forward(batch)["features"]
            det_output = det_head(features)

        # out_channels = num_anchors * (5 + num_classes) = 3 * 10 = 30
        expected_channels = 3 * (5 + 5)
        assert det_output.shape == (4, expected_channels, 8, 8), (
            f"Expected (4, {expected_channels}, 8, 8), got {det_output.shape}"
        )


class TestSegmentation:
    """Segmentation head forward and training tests."""

    def test_segmentation_head_forward(self, device: torch.device) -> None:
        """TinyFeatureMapModel(16) + SegmentationHead(16,3) -> (B,3,H,W)."""
        backbone = TinyFeatureMapModel(out_channels=16).to(device)
        seg_head = SegmentationHead(in_channels=16, num_classes=3).to(device)

        batch = make_feature_map_batches(
            num_batches=1, batch_size=4, image_size=8
        )[0]

        backbone.eval()
        seg_head.eval()
        with torch.no_grad():
            features = backbone.forward(batch)["features"]
            seg_output = seg_head(features)

        assert seg_output.shape == (4, 3, 8, 8), (
            f"Expected (4, 3, 8, 8), got {seg_output.shape}"
        )

    def test_segmentation_training(self, device: torch.device) -> None:
        """Train backbone + SegmentationHead with pixel-wise CE for 3 epochs."""
        backbone = TinyFeatureMapModel(out_channels=16).to(device)
        seg_head = SegmentationHead(in_channels=16, num_classes=3).to(device)

        batches = make_segmentation_batches(
            num_batches=5, batch_size=4, num_classes=3, image_size=8
        )

        # Combine parameters from both modules
        params = list(backbone.parameters()) + list(seg_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-3)

        epoch_losses: list[float] = []
        for _epoch in range(3):
            backbone.train()
            seg_head.train()
            total_loss = 0.0
            for batch in batches:
                optimizer.zero_grad()
                features = backbone.forward(batch)["features"]
                logits = seg_head(features)  # (B, C, H, W)
                labels = batch["labels"]  # (B, H, W)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_losses.append(total_loss / len(batches))

        assert epoch_losses[-1] < epoch_losses[0], (
            f"Segmentation loss did not decrease: "
            f"first={epoch_losses[0]:.4f}, last={epoch_losses[-1]:.4f}"
        )


class TestLoRA:
    """LoRA adapter integration tests."""

    def test_lora_adapter(self, device: torch.device) -> None:
        """Apply LoRA to TinyVisionModel; verify trainable < total params."""
        import pytest

        pytest.importorskip("peft")
        from mdp.models.adapters.lora import apply_lora

        model = TinyVisionModel(num_classes=2, hidden_dim=16).to(device)

        total_before = sum(p.numel() for p in model.parameters())

        model = apply_lora(
            model,
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["classifier", "head"],
        )
        model = model.to(device)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        assert trainable < total, (
            f"Trainable params ({trainable}) should be less than total ({total})"
        )
        assert trainable > 0, "No trainable parameters after LoRA"

        # Run a few training steps to verify gradients flow
        batches = make_vision_batches(
            num_batches=2, batch_size=4, num_classes=2, image_size=8
        )
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
        )

        model.train()
        for batch in batches:
            optimizer.zero_grad()
            loss = model(batch)["loss"]
            loss.backward()
            optimizer.step()

        # If we got here without error, gradients flowed successfully
