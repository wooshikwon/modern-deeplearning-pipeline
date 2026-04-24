"""Cross-entropy helpers for memory-efficient training paths.

Currently exposes one function: ``compute_per_token_ce_chunked_from_hidden``,
which reconstructs logits chunk-by-chunk from hidden states + head_weight
and wraps each chunk in ``torch.utils.checkpoint.checkpoint`` so that the
(B, chunk, V) intermediate tensor is not retained in the autograd graph.

Design rationale and alternatives:
    - dev-cycle/sincere-gnat-917-diagnosis-2026-04-23.md
    - dev-cycle/spec/spec-chunked-ce-from-hidden.md
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def compute_per_token_ce_chunked_from_hidden(
    hidden_states: torch.Tensor,   # (B, S, H), requires_grad typically True
    head_weight: torch.Tensor,     # (V, H), requires_grad True or False
    labels: torch.Tensor,          # (B, S), int
    chunk_size: int,
    ignore_index: int = -100,
) -> torch.Tensor:                 # (B, S-1), dtype == hidden_states.dtype
    """Per-token cross-entropy via sequence-chunked hidden→logits projection
    with gradient checkpointing.

    Splits the sequence dimension into chunks of ``chunk_size`` tokens, computes
    the (B, chunk, V) logits only for that chunk, and immediately reduces to
    per-token CE losses before discarding the logits.  Each chunk is wrapped in
    ``torch.utils.checkpoint.checkpoint`` so the intermediate logit tensor is
    recomputed during the backward pass instead of being kept in GPU memory.

    Args:
        hidden_states: Last-layer hidden states, shape (B, S, H).
        head_weight: LM head weight matrix, shape (V, H).  Bias is not
            supported — the common case for modern LMs.
        labels: Integer token labels, shape (B, S).  Positions with value
            ``ignore_index`` contribute 0 to the output.
        chunk_size: Number of sequence positions processed per chunk.  Smaller
            values reduce peak memory at the cost of more kernel launches.
        ignore_index: Label value to mask out (default -100, matching PyTorch
            ``F.cross_entropy`` convention).

    Returns:
        Per-token CE losses, shape (B, S-1), same dtype as ``hidden_states``.
        The shift (predicting token t+1 from hidden t) is applied internally.
    """
    shift_hidden = hidden_states[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    _, S_minus_1, _ = shift_hidden.shape

    def _ce_chunk(h_chunk, l_chunk, w):
        logits_chunk = h_chunk @ w.T                       # (B, c, V)
        return F.cross_entropy(
            logits_chunk.transpose(1, 2),                   # (B, V, c)
            l_chunk,
            ignore_index=ignore_index,
            reduction="none",
        )                                                   # (B, c)

    chunks: list[torch.Tensor] = []
    for start in range(0, S_minus_1, chunk_size):
        end = min(start + chunk_size, S_minus_1)
        chunk_ce = checkpoint(
            _ce_chunk,
            shift_hidden[:, start:end, :],
            shift_labels[:, start:end],
            head_weight,
            use_reentrant=False,
        )
        chunks.append(chunk_ce)
    return torch.cat(chunks, dim=1)                          # (B, S-1)
