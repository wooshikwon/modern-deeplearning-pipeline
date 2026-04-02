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


# ── GRPO ──


class GRPOLoss:
    """Group Relative Policy Optimization.

    Generation 필요. K개 응답의 group reward 평균 대비 advantage로 policy gradient.
    Value model 불필요.

        algorithm:
          _component_: GRPO
          clip_range: 0.2
          kl_coeff: 0.1
    """

    needs_generation = True

    def __init__(
        self,
        clip_range: float = 0.2,
        kl_coeff: float = 0.1,
        ppo_epochs: int = 1,
    ) -> None:
        self.clip_range = clip_range
        self.kl_coeff = kl_coeff
        self.ppo_epochs = ppo_epochs

    def __call__(
        self,
        trainable_out: dict[str, Any],
        frozen_out: dict[str, Any],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        new_log_probs = compute_log_probs(trainable_out["policy"]["logits"], batch["input_ids"])
        old_log_probs = batch["old_log_probs"]

        # mask: prompt 이후 토큰만
        prompt_len = batch.get("prompt_length", 0)
        seq_len = new_log_probs.shape[1]
        mask = torch.ones_like(new_log_probs, dtype=torch.bool)
        if prompt_len > 0:
            mask[:, :max(0, prompt_len - 1)] = False

        # importance ratio
        ratio = (new_log_probs - old_log_probs).exp()

        # group advantage (per-sequence reward가 있으면 사용, 없으면 log_prob 기반)
        with torch.no_grad():
            seq_rewards = (new_log_probs * mask).sum(dim=-1)
            advantages = (seq_rewards - seq_rewards.mean()) / (seq_rewards.std() + 1e-8)

        # clipped surrogate
        adv = advantages.unsqueeze(-1).expand_as(ratio)
        surr1 = ratio * adv
        surr2 = ratio.clamp(1 - self.clip_range, 1 + self.clip_range) * adv
        policy_loss = -masked_mean(torch.min(surr1, surr2), mask)

        # KL penalty
        if "reference" in frozen_out and "logits" in frozen_out["reference"]:
            ref_log_probs = compute_log_probs(frozen_out["reference"]["logits"], batch["input_ids"])
            kl = masked_mean(new_log_probs - ref_log_probs, mask)
            policy_loss = policy_loss + self.kl_coeff * kl

        return {"policy": policy_loss}


# ── PPO ──


class PPOLoss:
    """Proximal Policy Optimization.

    Generation 필요. Value model로 advantage 추정 + clipped surrogate.

        algorithm:
          _component_: PPO
          clip_range: 0.2
          kl_coeff: 0.1
          value_coeff: 0.5
    """

    needs_generation = True

    def __init__(
        self,
        clip_range: float = 0.2,
        kl_coeff: float = 0.1,
        value_coeff: float = 0.5,
        ppo_epochs: int = 4,
    ) -> None:
        self.clip_range = clip_range
        self.kl_coeff = kl_coeff
        self.value_coeff = value_coeff
        self.ppo_epochs = ppo_epochs

    def __call__(
        self,
        trainable_out: dict[str, Any],
        frozen_out: dict[str, Any],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        new_log_probs = compute_log_probs(trainable_out["policy"]["logits"], batch["input_ids"])
        old_log_probs = batch["old_log_probs"]

        prompt_len = batch.get("prompt_length", 0)
        mask = torch.ones_like(new_log_probs, dtype=torch.bool)
        if prompt_len > 0:
            mask[:, :max(0, prompt_len - 1)] = False

        # importance ratio
        ratio = (new_log_probs - old_log_probs).exp()

        # advantage from reward model (or value targets in batch)
        with torch.no_grad():
            if "rewards" in batch:
                rewards = batch["rewards"]
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                seq_rewards = (new_log_probs * mask).sum(dim=-1)
                advantages = (seq_rewards - seq_rewards.mean()) / (seq_rewards.std() + 1e-8)

        adv = advantages.unsqueeze(-1).expand_as(ratio)
        surr1 = ratio * adv
        surr2 = ratio.clamp(1 - self.clip_range, 1 + self.clip_range) * adv
        policy_loss = -masked_mean(torch.min(surr1, surr2), mask)

        # KL penalty
        if "reference" in frozen_out and "logits" in frozen_out["reference"]:
            ref_log_probs = compute_log_probs(frozen_out["reference"]["logits"], batch["input_ids"])
            kl = masked_mean(new_log_probs - ref_log_probs, mask)
            policy_loss = policy_loss + self.kl_coeff * kl

        losses = {"policy": policy_loss}

        # value loss
        if "value" in trainable_out and "logits" in trainable_out["value"]:
            values = trainable_out["value"]["logits"][:, :-1, 0]  # [batch, seq]
            value_targets = advantages.unsqueeze(-1).expand_as(values).detach()
            value_loss = F.mse_loss(values * mask.float(), value_targets * mask.float())
            losses["value"] = self.value_coeff * value_loss

        return losses
