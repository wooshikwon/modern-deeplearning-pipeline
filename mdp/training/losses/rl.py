"""RL alignment loss 함수 — DPO, weighted-NTP, GRPO, PPO."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


# ── 공유 유틸 ──


def compute_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits에서 labels 위치의 per-token log probability를 추출한다.

    Args:
        logits: (batch, seq, vocab) — 모델 출력.
        labels: (batch, seq) — 정답 토큰 ID. -100은 무시.

    Returns:
        (batch, seq-1) per-token log_prob. shifted (logits[:, :-1] vs labels[:, 1:]).
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_softmax = F.log_softmax(shift_logits, dim=-1)
    log_probs = log_softmax.gather(dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)

    # -100 위치는 0으로 마스킹
    mask = shift_labels != -100
    return log_probs * mask


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask된 위치만 평균."""
    if mask.sum() == 0:
        return tensor.new_tensor(0.0)
    return (tensor * mask).sum() / mask.sum()


# ── DPO ──


class DPOLoss:
    """Direct Preference Optimization loss.

    _component_ 패턴으로 인스턴스화된다:
        algorithm:
          _component_: DPO
          beta: 0.1
    """

    def __init__(self, beta: float = 0.1) -> None:
        self.beta = beta

    def __call__(
        self,
        trainable_out: dict[str, Any],
        frozen_out: dict[str, Any],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        policy = trainable_out["policy"]
        ref = frozen_out["reference"]

        policy_chosen_lp = compute_log_probs(policy["chosen_logits"], batch["chosen_labels"])
        policy_rejected_lp = compute_log_probs(policy["rejected_logits"], batch["rejected_labels"])
        ref_chosen_lp = compute_log_probs(ref["chosen_logits"], batch["chosen_labels"])
        ref_rejected_lp = compute_log_probs(ref["rejected_logits"], batch["rejected_labels"])

        chosen_mask = batch["chosen_labels"][:, 1:] != -100
        rejected_mask = batch["rejected_labels"][:, 1:] != -100

        policy_chosen_sum = (policy_chosen_lp * chosen_mask).sum(dim=-1)
        policy_rejected_sum = (policy_rejected_lp * rejected_mask).sum(dim=-1)
        ref_chosen_sum = (ref_chosen_lp * chosen_mask).sum(dim=-1)
        ref_rejected_sum = (ref_rejected_lp * rejected_mask).sum(dim=-1)

        logits = self.beta * (
            (policy_chosen_sum - ref_chosen_sum) - (policy_rejected_sum - ref_rejected_sum)
        )

        return {"policy": -F.logsigmoid(logits).mean()}
