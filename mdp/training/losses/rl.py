"""RL alignment loss 함수 — DPO, GRPO, PPO."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from mdp.training.losses.base import BaseAlgorithm


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


def normalize_advantages(
    advantages: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Advantage를 정규화한다. 분산 학습 시 전체 GPU에서 통계를 동기화한다."""
    import torch.distributed as dist

    masked_adv = advantages * mask.float()
    count = mask.sum()
    adv_sum = masked_adv.sum()
    adv_sq_sum = (masked_adv ** 2).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count)
        dist.all_reduce(adv_sum)
        dist.all_reduce(adv_sq_sum)

    if count == 0:
        return advantages

    mean = adv_sum / count
    var = adv_sq_sum / count - mean ** 2
    std = var.clamp(min=0).sqrt()

    return (advantages - mean) / (std + 1e-8)


# ── DPO ──


class DPOLoss(BaseAlgorithm):
    """Direct Preference Optimization loss.

    _component_ 패턴으로 인스턴스화된다:
        algorithm:
          _component_: DPO
          beta: 0.1
    """

    def __init__(self, beta: float = 0.1) -> None:
        self.beta = beta

    def compute_loss(
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


# ── 공유: Advantage 계산 ──


def _make_response_mask(
    shape: tuple, prompt_length: int, device: torch.device | str | None = None,
) -> torch.Tensor:
    """prompt 이후 토큰만 True인 mask를 생성한다."""
    mask = torch.ones(shape, dtype=torch.bool, device=device)
    if prompt_length > 0:
        mask[:, :max(0, prompt_length - 1)] = False
    return mask


def compute_gae(
    values: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
    last_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generalized Advantage Estimation.

    Args:
        values: (batch, seq) — value model의 per-token value 예측.
        rewards: (batch, seq) — per-token reward. 보통 마지막 토큰에만 scalar reward, 나머지 0.
        mask: (batch, seq) — response 토큰만 True.
        gamma: 할인율. 텍스트 생성에서는 보통 1.0.
        lam: GAE lambda. 편향-분산 트레이드오프.
        last_values: (batch,) — truncation bootstrap. None이면 V(T+1)=0.

    Returns:
        (batch, seq) per-token advantage.
    """
    seq_len = values.shape[1]
    advantages = torch.zeros_like(values)
    last_gae = torch.zeros(values.shape[0], device=values.device)

    for t in reversed(range(seq_len)):
        if t + 1 < seq_len:
            next_value = values[:, t + 1]
        elif last_values is not None:
            next_value = last_values
        else:
            next_value = torch.zeros_like(last_gae)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae

    return advantages * mask.float()


def _clipped_surrogate(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    """Clipped surrogate objective."""
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - clip_range, 1 + clip_range) * advantages
    return -masked_mean(torch.min(surr1, surr2), mask)


# ── GRPO ──


class GRPOLoss(BaseAlgorithm):
    """Group Relative Policy Optimization.

    Generation 필요. group 내 reward 평균 대비 advantage.
    Value model 불필요 — reward의 mean-normalization이 baseline 역할.

    batch에 "rewards" (batch,) scalar per sequence가 필수.
    RLTrainer가 frozen reward model forward 결과를 batch["rewards"]에 넣어준다.

        algorithm:
          _component_: GRPO
          clip_range: 0.2
          kl_coeff: 0.1
    """

    needs_generation: ClassVar[bool] = True

    def __init__(
        self,
        clip_range: float = 0.2,
        kl_coeff: float = 0.1,
        mini_epochs: int = 1,
    ) -> None:
        self.clip_range = clip_range
        self.kl_coeff = kl_coeff
        self.mini_epochs = mini_epochs

    def compute_loss(
        self,
        trainable_out: dict[str, Any],
        frozen_out: dict[str, Any],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        new_log_probs = compute_log_probs(trainable_out["policy"]["logits"], batch["input_ids"])
        old_log_probs = batch["old_log_probs"]
        mask = _make_response_mask(new_log_probs.shape, batch.get("prompt_length", 0), device=new_log_probs.device)

        log_ratio = (new_log_probs - old_log_probs).clamp(min=-20.0, max=20.0)
        ratio = log_ratio.exp()

        # GRPO advantage: per-sequence reward → group 또는 batch normalization
        rewards = batch["rewards"]  # (batch*K,) or (batch,) scalar per sequence
        K = batch.get("group_size", 1)
        with torch.no_grad():
            if K > 1:
                # group 내 정규화: (batch*K,) → (batch, K) → normalize → (batch*K,)
                grouped = rewards.view(-1, K)
                g_mean = grouped.mean(dim=1, keepdim=True)
                g_std = grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
                advantages = ((grouped - g_mean) / g_std).view(-1)
            else:
                reward_mask = torch.ones_like(rewards, dtype=torch.bool)
                advantages = normalize_advantages(rewards, reward_mask)
        # broadcast to per-token
        adv = advantages.unsqueeze(-1).expand_as(ratio)

        policy_loss = _clipped_surrogate(ratio, adv, mask, self.clip_range)

        # KL penalty
        if "reference" in frozen_out and "logits" in frozen_out["reference"]:
            ref_log_probs = compute_log_probs(frozen_out["reference"]["logits"], batch["input_ids"])
            kl = masked_mean(new_log_probs - ref_log_probs, mask)
            policy_loss = policy_loss + self.kl_coeff * kl

        return {"policy": policy_loss}


# ── PPO ──


class PPOLoss(BaseAlgorithm):
    """Proximal Policy Optimization.

    Generation 필요. Value model로 per-token GAE advantage + clipped surrogate.

    batch에 "rewards" (batch, seq) per-token reward가 필수.
    보통 마지막 토큰에만 scalar reward, 나머지 0.

        algorithm:
          _component_: PPO
          clip_range: 0.2
          kl_coeff: 0.1
          value_coeff: 0.5
          gae_lambda: 0.95
    """

    needs_generation: ClassVar[bool] = True

    def __init__(
        self,
        clip_range: float = 0.2,
        kl_coeff: float = 0.1,
        value_coeff: float = 0.5,
        gae_lambda: float = 0.95,
        mini_epochs: int = 4,
    ) -> None:
        self.clip_range = clip_range
        self.kl_coeff = kl_coeff
        self.value_coeff = value_coeff
        self.gae_lambda = gae_lambda
        self.mini_epochs = mini_epochs

    def compute_loss(
        self,
        trainable_out: dict[str, Any],
        frozen_out: dict[str, Any],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        new_log_probs = compute_log_probs(trainable_out["policy"]["logits"], batch["input_ids"])
        old_log_probs = batch["old_log_probs"]
        mask = _make_response_mask(new_log_probs.shape, batch.get("prompt_length", 0), device=new_log_probs.device)

        log_ratio = (new_log_probs - old_log_probs).clamp(min=-20.0, max=20.0)
        ratio = log_ratio.exp()

        # rewards: (batch,) scalar → per-token (마지막 response 토큰에 배치)
        raw_rewards = batch["rewards"]  # (batch,) scalar from reward model
        per_token_rewards = torch.zeros_like(new_log_probs)
        # 마지막 유효 토큰에 scalar reward 배치
        response_lengths = mask.sum(dim=-1).long()
        for i in range(per_token_rewards.shape[0]):
            idx = response_lengths[i].item() - 1
            if 0 <= idx < per_token_rewards.shape[1]:
                per_token_rewards[i, idx] = raw_rewards[i]

        # Value model → per-token value → GAE
        # _forward_model(role="value")가 {"values": (batch, seq)} 를 반환한다.
        if "value" in trainable_out and "values" in trainable_out["value"]:
            values_full = trainable_out["value"]["values"]
            values_raw = values_full[:, :-1]  # causal shift: (batch, seq-1)
            if values_raw.shape[1] < new_log_probs.shape[1]:
                values_padded = F.pad(values_raw, (0, new_log_probs.shape[1] - values_raw.shape[1]))
            else:
                values_padded = values_raw[:, :new_log_probs.shape[1]]
            # truncation bootstrap: 마지막 토큰의 value 예측.
            # EOS 종료 시퀀스에서는 value model이 ~0을 예측하므로 안전.
            last_values = values_full[:, -1].detach()
        else:
            values_padded = torch.zeros_like(new_log_probs)
            last_values = None

        with torch.no_grad():
            advantages = compute_gae(values_padded.detach(), per_token_rewards, mask, lam=self.gae_lambda, last_values=last_values)
            advantages = normalize_advantages(advantages, mask)

        # clipped surrogate
        policy_loss = _clipped_surrogate(ratio, advantages, mask, self.clip_range)

        # KL penalty
        if "reference" in frozen_out and "logits" in frozen_out["reference"]:
            ref_log_probs = compute_log_probs(frozen_out["reference"]["logits"], batch["input_ids"])
            kl = masked_mean(new_log_probs - ref_log_probs, mask)
            policy_loss = policy_loss + self.kl_coeff * kl

        losses = {"policy": policy_loss}

        # value loss: MSE(values, returns)
        if "value" in trainable_out and "values" in trainable_out["value"]:
            returns = (advantages + values_padded).detach()
            value_loss = F.mse_loss(values_padded * mask.float(), returns * mask.float())
            losses["value"] = self.value_coeff * value_loss

        return losses
