"""Unit tests for mdp.training.losses._ce_helpers.

CPU 테스트 4개는 항상 실행.
CUDA 테스트 3개는 CUDA 장치가 없으면 skip.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mdp.training.losses._ce_helpers import compute_per_token_ce_chunked_from_hidden

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CUDA_AVAILABLE = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


def _make_inputs(
    B: int,
    S: int,
    V: int,
    H: int,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    hidden_requires_grad: bool = True,
    head_requires_grad: bool = True,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (hidden_states, head_weight, labels) on the given device."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    hidden = torch.randn(B, S, H, dtype=dtype, generator=gen).to(device)
    head = torch.randn(V, H, dtype=dtype, generator=gen).to(device)
    labels = torch.randint(0, V, (B, S), generator=gen).to(device)

    hidden.requires_grad_(hidden_requires_grad)
    head.requires_grad_(head_requires_grad)
    return hidden, head, labels


def _naive_ce(
    hidden: torch.Tensor,
    head: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Reference: single-shot logit projection then cross_entropy."""
    shift_hidden = hidden[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    logits = shift_hidden @ head.T                           # (B, S-1, V)
    return F.cross_entropy(
        logits.transpose(1, 2),                              # (B, V, S-1)
        shift_labels,
        ignore_index=ignore_index,
        reduction="none",
    )


# ---------------------------------------------------------------------------
# CPU tests (always run)
# ---------------------------------------------------------------------------


class TestNumericalEquivalenceVsNaive:
    """test_numerical_equivalence_vs_naive — CPU fp32."""

    def test_numerical_equivalence_vs_naive(self) -> None:
        B, S, V, H, chunk_size = 2, 32, 64, 48, 7
        hidden, head, labels = _make_inputs(B, S, V, H)

        result = compute_per_token_ce_chunked_from_hidden(
            hidden, head, labels, chunk_size=chunk_size
        )
        reference = _naive_ce(hidden.detach(), head.detach(), labels)

        assert result.shape == (B, S - 1)
        assert torch.allclose(result, reference, atol=1e-5, rtol=1e-4), (
            f"max abs diff: {(result - reference).abs().max().item():.2e}"
        )


class TestChunkSizeInvariance:
    """test_chunk_size_invariance — all chunk sizes produce the same result."""

    def test_chunk_size_invariance(self) -> None:
        B, S, V, H = 2, 33, 32, 16
        hidden, head, labels = _make_inputs(B, S, V, H)

        reference = compute_per_token_ce_chunked_from_hidden(
            hidden, head, labels, chunk_size=1
        )
        for cs in (5, 17, S - 1, S):
            result = compute_per_token_ce_chunked_from_hidden(
                hidden, head, labels, chunk_size=cs
            )
            assert torch.allclose(result, reference, atol=1e-6), (
                f"chunk_size={cs}: max abs diff {(result - reference).abs().max().item():.2e}"
            )


class TestIgnoreIndexZerosLoss:
    """test_ignore_index_zeros_loss — masked positions contribute 0."""

    def test_ignore_index_zeros_loss(self) -> None:
        B, S, V, H = 2, 16, 32, 16
        hidden, head, labels = _make_inputs(B, S, V, H)

        # Mask a subset of shifted labels (labels[:, 1:]) to -100.
        # We mark positions [0, 2, 4] of the shifted sequence.
        mask_positions = [0, 2, 4]
        masked_labels = labels.clone()
        for pos in mask_positions:
            masked_labels[:, pos + 1] = -100   # +1 because shift applies labels[:,1:]

        result = compute_per_token_ce_chunked_from_hidden(
            hidden, head, masked_labels, chunk_size=8
        )

        assert result.shape == (B, S - 1)
        for pos in mask_positions:
            assert (result[:, pos] == 0.0).all(), (
                f"Expected 0 at masked position {pos}, got {result[:, pos]}"
            )


class TestBackwardReachesHiddenAndHead:
    """test_backward_reaches_hidden_and_head — gradients flow correctly."""

    def test_both_require_grad(self) -> None:
        """hidden.requires_grad=True, head.requires_grad=True."""
        B, S, V, H = 2, 16, 32, 16
        hidden, head, labels = _make_inputs(
            B, S, V, H, hidden_requires_grad=True, head_requires_grad=True
        )

        result = compute_per_token_ce_chunked_from_hidden(
            hidden, head, labels, chunk_size=8
        )
        result.sum().backward()

        assert hidden.grad is not None, "hidden.grad is None"
        assert hidden.grad.abs().sum() > 0, "hidden.grad is all-zero"
        assert head.grad is not None, "head.grad is None"
        assert head.grad.abs().sum() > 0, "head.grad is all-zero"

    def test_frozen_head(self) -> None:
        """hidden.requires_grad=True, head.requires_grad=False (frozen head)."""
        B, S, V, H = 2, 16, 32, 16
        hidden, head, labels = _make_inputs(
            B, S, V, H, hidden_requires_grad=True, head_requires_grad=False
        )

        result = compute_per_token_ce_chunked_from_hidden(
            hidden, head, labels, chunk_size=8
        )
        result.sum().backward()

        assert hidden.grad is not None, "hidden.grad is None (frozen head scenario)"
        assert hidden.grad.abs().sum() > 0, "hidden.grad is all-zero (frozen head scenario)"
        assert head.grad is None, "head.grad should be None when requires_grad=False"


# ---------------------------------------------------------------------------
# CUDA tests (skipped when CUDA unavailable)
# ---------------------------------------------------------------------------


@skip_no_cuda
class TestAmpAutocastEquivalence:
    """test_amp_autocast_on_off_equivalence — PyTorch #141896 regression guard."""

    def test_amp_autocast_on_off_equivalence(self) -> None:
        B, S, V, H = 2, 32, 128, 64
        device = "cuda"
        hidden, head, labels = _make_inputs(B, S, V, H, device=device)

        # Without autocast (fp32)
        result_fp32 = compute_per_token_ce_chunked_from_hidden(
            hidden, head, labels, chunk_size=16
        )

        # With autocast (bf16)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            result_bf16 = compute_per_token_ce_chunked_from_hidden(
                hidden, head, labels, chunk_size=16
            )

        # bf16 has low precision — use loose tolerance
        assert torch.allclose(result_fp32.float(), result_bf16.float(), atol=1e-2), (
            f"AMP autocast mismatch: max abs diff "
            f"{(result_fp32.float() - result_bf16.float()).abs().max().item():.2e}"
        )


@skip_no_cuda
class TestPeakMemoryScalesWithChunkSize:
    """test_peak_memory_scales_with_chunk_size — smaller chunk → lower peak."""

    def test_peak_memory_scales_with_chunk_size(self) -> None:
        B, S, V, H = 4, 1024, 32000, 1024
        device = "cuda"
        dtype = torch.float32

        def _measure_peak(chunk_size: int) -> int:
            hidden, head, labels = _make_inputs(
                B, S, V, H, device=device, dtype=dtype,
                hidden_requires_grad=True, head_requires_grad=False,
                seed=0,
            )
            torch.cuda.reset_peak_memory_stats(device)
            result = compute_per_token_ce_chunked_from_hidden(
                hidden, head, labels, chunk_size=chunk_size
            )
            result.sum().backward()
            return torch.cuda.max_memory_allocated(device)

        peak_small = _measure_peak(chunk_size=64)
        peak_large = _measure_peak(chunk_size=256)

        assert peak_small < peak_large, (
            f"Expected chunk_size=64 to use less memory than chunk_size=256. "
            f"Got peak_small={peak_small / 2**20:.1f} MiB, "
            f"peak_large={peak_large / 2**20:.1f} MiB"
        )
        assert peak_small < 1 * 2**30, (
            f"chunk_size=64 peak exceeds 1 GiB: {peak_small / 2**30:.2f} GiB"
        )


@skip_no_cuda
class TestPeakMemoryVsNoCheckpointVariant:
    """test_peak_memory_vs_no_checkpoint_variant — checkpoint saves ≥30% memory."""

    def test_peak_memory_vs_no_checkpoint_variant(self) -> None:
        B, S, V, H = 4, 1024, 32000, 1024
        device = "cuda"
        dtype = torch.float32
        chunk_size = 64

        # --- helper baseline: chunked loop WITHOUT checkpoint ---
        def _naive_chunked_loop(
            hidden: torch.Tensor,
            head: torch.Tensor,
            labels: torch.Tensor,
        ) -> torch.Tensor:
            shift_hidden = hidden[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            S_minus_1 = shift_hidden.shape[1]
            parts: list[torch.Tensor] = []
            for start in range(0, S_minus_1, chunk_size):
                end = min(start + chunk_size, S_minus_1)
                h_chunk = shift_hidden[:, start:end, :]
                l_chunk = shift_labels[:, start:end]
                logits_chunk = h_chunk @ head.T
                ce_chunk = F.cross_entropy(
                    logits_chunk.transpose(1, 2),
                    l_chunk,
                    ignore_index=-100,
                    reduction="none",
                )
                parts.append(ce_chunk)
            return torch.cat(parts, dim=1)

        def _measure(use_checkpoint: bool) -> int:
            hidden, head, labels = _make_inputs(
                B, S, V, H, device=device, dtype=dtype,
                hidden_requires_grad=True, head_requires_grad=False,
                seed=0,
            )
            torch.cuda.reset_peak_memory_stats(device)
            if use_checkpoint:
                result = compute_per_token_ce_chunked_from_hidden(
                    hidden, head, labels, chunk_size=chunk_size
                )
            else:
                result = _naive_chunked_loop(hidden, head, labels)
            result.sum().backward()
            return torch.cuda.max_memory_allocated(device)

        peak_ckpt = _measure(use_checkpoint=True)
        peak_naive = _measure(use_checkpoint=False)

        reduction = 1.0 - peak_ckpt / peak_naive
        assert reduction >= 0.30, (
            f"Expected ≥30% memory reduction with checkpoint. "
            f"Got {reduction * 100:.1f}% "
            f"(checkpoint={peak_ckpt / 2**20:.1f} MiB, "
            f"naive={peak_naive / 2**20:.1f} MiB)"
        )
