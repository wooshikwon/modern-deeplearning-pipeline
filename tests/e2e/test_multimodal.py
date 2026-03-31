"""E2E tests for multimodal dual-encoder models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mdp.models.heads.dual_encoder import DualEncoderHead
from tests.e2e.datasets import ListDataLoader, make_multimodal_batches
from tests.e2e.models import TinyDualEncoderModel


def _contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """CLIP-style symmetric contrastive loss."""
    image_embeds = F.normalize(image_features, dim=-1)
    text_embeds = F.normalize(text_features, dim=-1)
    logits = image_embeds @ text_embeds.T  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


# ---------------------------------------------------------------------------
# Dual encoder tests
# ---------------------------------------------------------------------------


def test_dual_encoder_training() -> None:
    """Train TinyDualEncoderModel with contrastive loss for 3 epochs; loss decreases."""
    model = TinyDualEncoderModel(hidden_dim=16)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = make_multimodal_batches(5, 4, 8, 8, 128)
    loader = ListDataLoader(batches)

    epoch_losses: list[float] = []
    for _epoch in range(3):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = _contrastive_loss(
                out["image_features"],
                out["text_features"],
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_losses.append(total_loss / len(batches))

    assert epoch_losses[-1] < epoch_losses[0], (
        f"Loss did not decrease: {epoch_losses}"
    )


def test_dual_encoder_head() -> None:
    """Attach DualEncoderHead(16, 8), forward_pair, verify L2 normalized output."""
    head = DualEncoderHead(embed_dim=8, projection_dim=8)
    head.eval()

    model = TinyDualEncoderModel(hidden_dim=16)
    model.eval()

    batches = make_multimodal_batches(1, 4, 8, 8, 128)
    batch = batches[0]
    with torch.no_grad():
        out = model(batch)
        result = head.forward_pair(out["image_features"], out["text_features"])

    image_embeds = result["image_embeds"]
    text_embeds = result["text_embeds"]

    assert image_embeds.shape == (4, 8)
    assert text_embeds.shape == (4, 8)

    # Verify L2 normalization: norms should be ~1.0
    image_norms = image_embeds.norm(dim=-1)
    text_norms = text_embeds.norm(dim=-1)
    assert torch.allclose(image_norms, torch.ones_like(image_norms), atol=1e-5)
    assert torch.allclose(text_norms, torch.ones_like(text_norms), atol=1e-5)


def test_dual_encoder_similarity_matrix() -> None:
    """Compute similarity matrix; diagonal should exceed off-diagonal on average."""
    model = TinyDualEncoderModel(hidden_dim=16)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = make_multimodal_batches(5, 4, 8, 8, 128)
    loader = ListDataLoader(batches)

    # Train for 10 epochs so the model learns some alignment
    for _epoch in range(10):
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = _contrastive_loss(
                out["image_features"],
                out["text_features"],
            )
            loss.backward()
            optimizer.step()

    # Evaluate on the last batch
    model.eval()
    with torch.no_grad():
        out = model(batches[-1])
        image_embeds = F.normalize(out["image_features"], dim=-1)
        text_embeds = F.normalize(out["text_features"], dim=-1)
        sim = image_embeds @ text_embeds.T  # (B, B)

    B = sim.size(0)
    diag_mean = sim.diag().mean().item()
    # Off-diagonal mean
    off_diag_mask = ~torch.eye(B, dtype=torch.bool)
    off_diag_mean = sim[off_diag_mask].mean().item()

    assert diag_mean > off_diag_mean, (
        f"Diagonal mean ({diag_mean:.4f}) should exceed "
        f"off-diagonal mean ({off_diag_mean:.4f})"
    )
