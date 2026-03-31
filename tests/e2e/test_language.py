"""E2E tests for language and token classification models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mdp.models.heads.causal_lm import CausalLMHead
from mdp.models.heads.seq2seq_lm import Seq2SeqLMHead
from mdp.models.heads.token_classification import TokenClassificationHead
from tests.e2e.datasets import (
    ListDataLoader,
    make_language_batches,
    make_token_class_batches,
)
from tests.e2e.models import TinyLanguageModel, TinyTokenClassModel


# ---------------------------------------------------------------------------
# Causal LM tests
# ---------------------------------------------------------------------------


def test_causal_lm_training() -> None:
    """Train TinyLanguageModel for 3 epochs; loss must decrease."""
    model = TinyLanguageModel(vocab_size=128, hidden_dim=32)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = make_language_batches(5, 4, 16, 128)
    loader = ListDataLoader(batches)

    epoch_losses: list[float] = []
    for _epoch in range(3):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            logits = out["logits"]  # (B, L, vocab_size)
            # Causal LM: use input_ids as labels (shifted inside loss)
            labels = batch["input_ids"]  # (B, L)
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_losses.append(total_loss / len(batches))

    assert epoch_losses[-1] < epoch_losses[0], (
        f"Loss did not decrease: {epoch_losses}"
    )


def test_causal_lm_head_replacement() -> None:
    """Replace .lm_head with CausalLMHead(32, 64), verify output shape (B,L,64)."""
    model = TinyLanguageModel(vocab_size=128, hidden_dim=32)
    model.lm_head = CausalLMHead(hidden_dim=32, vocab_size=64)
    model.eval()

    batches = make_language_batches(1, 4, 16, 128)
    batch = batches[0]
    with torch.no_grad():
        out = model(batch)

    assert out["logits"].shape == (4, 16, 64)


def test_generate_not_implemented() -> None:
    """model.generate({}) must raise NotImplementedError."""
    import pytest

    model = TinyLanguageModel(vocab_size=128, hidden_dim=32)
    with pytest.raises(NotImplementedError):
        model.generate({})


# ---------------------------------------------------------------------------
# Token classification tests
# ---------------------------------------------------------------------------


def test_token_classification_training() -> None:
    """Train TinyTokenClassModel for 3 epochs; loss must decrease."""
    model = TinyTokenClassModel(vocab_size=128, hidden_dim=32, num_classes=5)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batches = make_token_class_batches(5, 4, 16, 128, 5)
    loader = ListDataLoader(batches)

    epoch_losses: list[float] = []
    for _epoch in range(3):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            logits = out["logits"]  # (B, L, num_classes)
            labels = batch["labels"]  # (B, L)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_losses.append(total_loss / len(batches))

    assert epoch_losses[-1] < epoch_losses[0], (
        f"Loss did not decrease: {epoch_losses}"
    )


def test_token_classification_head_swap() -> None:
    """Replace .head with TokenClassificationHead(32, 10), verify (B,L,10)."""
    model = TinyTokenClassModel(vocab_size=128, hidden_dim=32, num_classes=5)
    model.head = TokenClassificationHead(hidden_dim=32, num_classes=10)
    model.eval()

    batches = make_token_class_batches(1, 4, 16, 128, 5)
    batch = batches[0]
    with torch.no_grad():
        out = model(batch)

    assert out["logits"].shape == (4, 16, 10)


# ---------------------------------------------------------------------------
# Seq2Seq head forward test
# ---------------------------------------------------------------------------


def test_seq2seq_head_forward() -> None:
    """Attach Seq2SeqLMHead(32, 128) to TinyLanguageModel, verify same shape."""
    model = TinyLanguageModel(vocab_size=128, hidden_dim=32)
    model.lm_head = Seq2SeqLMHead(hidden_dim=32, vocab_size=128)
    model.eval()

    batches = make_language_batches(1, 4, 16, 128)
    batch = batches[0]
    with torch.no_grad():
        out = model(batch)

    assert out["logits"].shape == (4, 16, 128)
