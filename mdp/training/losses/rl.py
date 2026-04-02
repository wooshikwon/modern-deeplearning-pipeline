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


def dpo_loss(
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    ref_chosen_logits: torch.Tensor,
    ref_rejected_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Direct Preference Optimization loss.

    Returns:
        {"policy": loss_tensor}
    """
    policy_chosen_lp = compute_log_probs(policy_chosen_logits, chosen_labels)
    policy_rejected_lp = compute_log_probs(policy_rejected_logits, rejected_labels)
    ref_chosen_lp = compute_log_probs(ref_chosen_logits, chosen_labels)
    ref_rejected_lp = compute_log_probs(ref_rejected_logits, rejected_labels)

    chosen_mask = chosen_labels[:, 1:] != -100
    rejected_mask = rejected_labels[:, 1:] != -100

    # per-sequence log_prob sum
    policy_chosen_sum = (policy_chosen_lp * chosen_mask).sum(dim=-1)
    policy_rejected_sum = (policy_rejected_lp * rejected_mask).sum(dim=-1)
    ref_chosen_sum = (ref_chosen_lp * chosen_mask).sum(dim=-1)
    ref_rejected_sum = (ref_rejected_lp * rejected_mask).sum(dim=-1)

    # DPO logit
    logits = beta * (
        (policy_chosen_sum - ref_chosen_sum) - (policy_rejected_sum - ref_rejected_sum)
    )

    loss = -F.logsigmoid(logits).mean()
    return {"policy": loss}


# ── 라우터 ──


def compute_rl_loss(
    algorithm: str,
    trainable_out: dict[str, Any],
    frozen_out: dict[str, Any],
    batch: dict[str, torch.Tensor],
    algo_config: Any,
) -> dict[str, torch.Tensor]:
    """알고리즘에 따라 적절한 loss를 계산한다.

    Returns:
        모델별 loss dict. 예: {"policy": tensor} 또는 {"policy": tensor, "value": tensor}
    """
    if algorithm == "dpo":
        beta = algo_config.beta if hasattr(algo_config, "beta") else 0.1
        return dpo_loss(
            policy_chosen_logits=trainable_out["policy"]["chosen_logits"],
            policy_rejected_logits=trainable_out["policy"]["rejected_logits"],
            ref_chosen_logits=frozen_out["reference"]["chosen_logits"],
            ref_rejected_logits=frozen_out["reference"]["rejected_logits"],
            chosen_labels=batch["chosen_labels"],
            rejected_labels=batch["rejected_labels"],
            beta=beta,
        )
    else:
        raise NotImplementedError(f"알고리즘 '{algorithm}'은 아직 구현되지 않았습니다")
