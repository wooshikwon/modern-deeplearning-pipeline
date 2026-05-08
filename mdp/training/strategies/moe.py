"""Expert Parallelism for MoE models.

MoE 레이어의 expert를 GPU에 분배하고, forward에 all-to-all token dispatch를
삽입한다. 기존 분산 전략(FSDP, DeepSpeed)과 **조합**하여 사용한다.

EP는 분산 전략이 아니다. DDP/FSDP/DeepSpeed가 "파라미터를 어떻게 분배할
것인가"를 결정하는 반면, EP는 "MoE expert를 GPU에 어떻게 배치하고 토큰을
어떻게 라우팅할 것인가"를 결정한다. 둘은 직교하는 관심사이므로, EP는
전략 setup 전에 모델에 적용되고, 이후 전략이 wrapping한다.

사용 흐름:
  1. ExpertParallel.setup(model) → EP hooks + expert 분배
  2. strategy.setup(model) → FSDP/DeepSpeed wrapping (공유 레이어)

Process Group 구성 (world_size=8, ep_size=4):
  EP groups: [[0,1,2,3], [4,5,6,7]]  — expert 간 token 교환
  DP groups: [[0,4], [1,5], [2,6], [3,7]]  — gradient 동기화
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

logger = logging.getLogger(__name__)


class _AllToAllDispatch(torch.autograd.Function):
    """all-to-all token dispatch의 커스텀 forward/backward."""

    @staticmethod
    def forward(
        ctx: Any,
        input_tensor: torch.Tensor,
        send_counts: torch.Tensor,
        recv_counts: torch.Tensor,
        ep_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.save_for_backward(send_counts, recv_counts)
        ctx.ep_group = ep_group
        return _all_to_all_tokens(input_tensor, send_counts, recv_counts, ep_group)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        send_counts, recv_counts = ctx.saved_tensors
        grad_input = _all_to_all_tokens(grad_output, recv_counts, send_counts, ctx.ep_group)
        return grad_input, None, None, None


def _all_to_all_tokens(
    tensor: torch.Tensor,
    send_counts: torch.Tensor,
    recv_counts: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """send_counts/recv_counts에 따라 텐서를 all-to-all 교환한다.

    nccl에서는 dist.all_to_all을 사용하고,
    gloo에서는 all_gather 기반 fallback으로 동일한 의미론을 구현한다.
    """
    send_sizes = send_counts.tolist()
    recv_sizes = recv_counts.tolist()
    total_recv = sum(recv_sizes)

    try:
        send_list = list(tensor.split(send_sizes))
        recv_tensor = torch.empty(
            total_recv, tensor.shape[-1],
            dtype=tensor.dtype, device=tensor.device,
        )
        recv_list = list(recv_tensor.split(recv_sizes))
        dist.all_to_all(recv_list, send_list, group=group)
        return torch.cat(recv_list, dim=0)
    except RuntimeError as e:
        if "alltoall" not in str(e).lower():
            raise
        return _all_to_all_via_gather(tensor, send_counts, recv_counts, group)


def _all_to_all_via_gather(
    tensor: torch.Tensor,
    send_counts: torch.Tensor,
    recv_counts: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """all_gather 기반 all-to-all 에뮬레이션 (gloo CPU용)."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    dim = tensor.shape[-1]

    local_total = send_counts.sum()
    max_tensor = local_total.clone()
    dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX, group=group)
    max_tokens = int(max_tensor.item())

    padded = torch.zeros(max_tokens, dim, dtype=tensor.dtype, device=tensor.device)
    actual = int(local_total.item())
    if actual > 0:
        padded[:actual] = tensor[:actual]

    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)

    all_send_counts = [torch.zeros_like(send_counts) for _ in range(world_size)]
    dist.all_gather(all_send_counts, send_counts, group=group)

    result_parts = []
    for src_rank in range(world_size):
        src_sends = all_send_counts[src_rank].tolist()
        offset = sum(src_sends[:rank])
        count = src_sends[rank]
        if count > 0:
            result_parts.append(gathered[src_rank][offset:offset + count])

    if result_parts:
        return torch.cat(result_parts, dim=0)
    return torch.empty(0, dim, dtype=tensor.dtype, device=tensor.device)


class ExpertParallel:
    """MoE Expert Parallelism — 기존 분산 전략과 조합하여 사용한다.

    Parameters
    ----------
    ep_size:
        Expert Parallel degree.  ``world_size``가 ``ep_size``로 나누어져야 한다.
    expert_module_pattern:
        MoE expert container를 식별하는 모듈 이름 패턴 (e.g. ``"experts"``).
    """

    def __init__(
        self,
        ep_size: int,
        expert_module_pattern: str = "experts",
    ) -> None:
        self.ep_size = ep_size
        self.expert_module_pattern = expert_module_pattern

        self._ep_group: dist.ProcessGroup | None = None
        self._dp_group: dist.ProcessGroup | None = None
        self._ep_rank: int = 0
        self._expert_range: tuple[int, int] = (0, 0)

    @property
    def dp_group(self) -> dist.ProcessGroup | None:
        """DP process group — 전략이 공유 레이어 sharding에 사용할 수 있다."""
        return self._dp_group

    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        """모델에 EP를 적용한다: hook 설치 + expert 분배.

        분산 전략(FSDP 등)의 setup **전에** 호출해야 한다.
        """
        if not dist.is_initialized():
            raise RuntimeError(
                "ExpertParallel.setup()은 dist.init_process_group() 이후에 호출해야 합니다. "
                "분산 전략의 setup()이 process group을 초기화하므로, "
                "Trainer에서 올바른 순서로 호출되어야 합니다."
            )

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        self._create_process_groups(rank, world_size)

        num_experts = self._detect_num_experts(model)
        experts_per_gpu = num_experts // self.ep_size
        self._expert_range = (
            self._ep_rank * experts_per_gpu,
            (self._ep_rank + 1) * experts_per_gpu,
        )

        dp_size = world_size // self.ep_size
        logger.info(
            "EP setup: rank=%d, ep_rank=%d, experts=%d-%d (of %d), "
            "ep_size=%d, dp_size=%d",
            rank, self._ep_rank,
            self._expert_range[0], self._expert_range[1] - 1,
            num_experts, self.ep_size, dp_size,
        )

        top_k = self._detect_top_k(model)

        if self.ep_size > 1:
            self._install_ep_hooks(model, num_experts, top_k)

        self._distribute_experts(model, device)

        return model

    # ------------------------------------------------------------------
    # Process group
    # ------------------------------------------------------------------

    def _create_process_groups(self, rank: int, world_size: int) -> None:
        for ep_start in range(0, world_size, self.ep_size):
            ep_ranks = list(range(ep_start, ep_start + self.ep_size))
            group = dist.new_group(ep_ranks)
            if rank in ep_ranks:
                self._ep_group = group
                self._ep_rank = ep_ranks.index(rank)

        for dp_offset in range(self.ep_size):
            dp_ranks = list(range(dp_offset, world_size, self.ep_size))
            group = dist.new_group(dp_ranks)
            if rank in dp_ranks:
                self._dp_group = group

    # ------------------------------------------------------------------
    # MoE 감지
    # ------------------------------------------------------------------

    def _detect_num_experts(self, model: nn.Module) -> int:
        config = getattr(model, "config", None)
        if config is not None:
            for attr in ("num_local_experts", "num_experts", "n_routed_experts"):
                val = getattr(config, attr, None)
                if val is not None:
                    return val

        for name, module in model.named_modules():
            if self.expert_module_pattern in name and isinstance(module, nn.ModuleList):
                return len(module)

        raise ValueError(
            f"MoE expert 수를 감지할 수 없습니다. "
            f"모델에 config.num_local_experts 또는 '{self.expert_module_pattern}' "
            f"ModuleList가 필요합니다."
        )

    def _detect_top_k(self, model: nn.Module) -> int:
        """모델 config에서 top-k (활성 expert 수)를 감지한다."""
        config = getattr(model, "config", None)
        if config is not None:
            val = getattr(config, "num_experts_per_tok", None)
            if val is not None:
                return val
        return 2  # MoE 모델의 일반적 기본값

    def _is_moe_layer(self, name: str, module: nn.Module) -> bool:
        for child_name, child in module.named_children():
            if self.expert_module_pattern in child_name and isinstance(child, nn.ModuleList):
                return True
        return False

    def _is_expert_param(self, param_name: str) -> bool:
        return self.expert_module_pattern in param_name

    # ------------------------------------------------------------------
    # EP hooks
    # ------------------------------------------------------------------

    def _install_ep_hooks(self, model: nn.Module, num_experts: int, top_k: int) -> None:
        hooked = 0
        for name, module in model.named_modules():
            if self._is_moe_layer(name, module):
                original_forward = module.forward
                module.forward = self._make_ep_forward(
                    module, original_forward, num_experts, top_k,
                )
                hooked += 1
        logger.info("EP hooks installed on %d MoE layers", hooked)

    def _make_ep_forward(
        self,
        moe_module: nn.Module,
        original_forward: Any,
        num_experts: int,
        top_k: int,
    ) -> Any:
        ep_group = self._ep_group
        ep_size = self.ep_size
        ep_rank = self._ep_rank
        expert_module_pattern = self.expert_module_pattern

        def ep_forward(hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            batch_seq_shape = hidden_states.shape[:-1]
            hidden_dim = hidden_states.shape[-1]
            flat_hidden = hidden_states.view(-1, hidden_dim)
            num_tokens = flat_hidden.shape[0]

            gate = None
            for child_name, child in moe_module.named_children():
                if "gate" in child_name or "router" in child_name:
                    gate = child
                    break
            if gate is None:
                return original_forward(hidden_states, **kwargs)

            router_logits = gate(flat_hidden)
            k = min(top_k, router_logits.shape[-1])
            routing_weights, selected_experts = torch.topk(
                router_logits.softmax(dim=-1), k=k, dim=-1,
            )
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

            experts_per_gpu = num_experts // ep_size
            target_ep_ranks = selected_experts // experts_per_gpu
            local_expert_indices = selected_experts % experts_per_gpu

            flat_targets = target_ep_ranks.view(-1)
            flat_local_experts = local_expert_indices.view(-1)

            send_counts = torch.zeros(ep_size, dtype=torch.long, device=flat_hidden.device)
            for r in range(ep_size):
                send_counts[r] = (flat_targets == r).sum()

            recv_counts = torch.empty_like(send_counts)
            dist.all_to_all_single(recv_counts, send_counts, group=ep_group)

            sort_indices = flat_targets.argsort(stable=True)
            sorted_token_indices = sort_indices // k
            total_send = int(send_counts.sum().item())
            dispatched = flat_hidden[sorted_token_indices[:total_send]]
            sorted_local_experts = flat_local_experts[sort_indices[:total_send]]

            received = _AllToAllDispatch.apply(dispatched, send_counts, recv_counts, ep_group)
            expert_idx_float = sorted_local_experts.float().unsqueeze(-1)
            received_idx = _AllToAllDispatch.apply(
                expert_idx_float, send_counts, recv_counts, ep_group,
            )
            received_local_experts = received_idx.squeeze(-1).long()

            experts = None
            for child_name, child in moe_module.named_children():
                if expert_module_pattern in child_name and isinstance(child, nn.ModuleList):
                    experts = child
                    break
            if experts is None:
                return original_forward(hidden_states, **kwargs)

            local_start = ep_rank * experts_per_gpu
            local_output = torch.zeros_like(received)
            for local_idx in range(experts_per_gpu):
                mask = received_local_experts == local_idx
                if mask.any():
                    expert_global_idx = local_start + local_idx
                    local_output[mask] = experts[expert_global_idx](received[mask])

            result = _AllToAllDispatch.apply(local_output, recv_counts, send_counts, ep_group)

            unsort_indices = sort_indices[:total_send].argsort()
            unsorted_result = result[unsort_indices]

            output = torch.zeros(num_tokens, hidden_dim, dtype=flat_hidden.dtype, device=flat_hidden.device)
            for ki in range(k):
                k_indices = torch.arange(ki, total_send, k, device=flat_hidden.device)
                if k_indices.numel() == 0:
                    continue
                token_ids = torch.arange(num_tokens, device=flat_hidden.device)
                valid = k_indices < unsorted_result.shape[0]
                k_indices = k_indices[valid]
                token_ids = token_ids[valid[:token_ids.shape[0]]]
                if token_ids.numel() > 0:
                    weight = routing_weights[token_ids, ki].unsqueeze(-1)
                    output[token_ids] += unsorted_result[k_indices] * weight

            return output.view(*batch_seq_shape, hidden_dim)

        return ep_forward

    # ------------------------------------------------------------------
    # Expert 분배
    # ------------------------------------------------------------------

    def _distribute_experts(self, model: nn.Module, device: torch.device) -> None:
        start, end = self._expert_range
        for name, module in model.named_modules():
            if self.expert_module_pattern in name and isinstance(module, nn.ModuleList):
                for idx, expert in enumerate(module):
                    if start <= idx < end:
                        expert.to(device)
                    else:
                        for p in expert.parameters():
                            p.requires_grad = False
                        expert.to("cpu")

    # ------------------------------------------------------------------
    # Checkpoint 지원: gather/scatter
    # ------------------------------------------------------------------

    def gather_experts(self, model: nn.Module) -> None:
        """모든 rank의 expert 가중치를 각 rank의 모델에 채운다.

        checkpoint 저장 전에 호출하면, 이후 strategy.save_checkpoint()이
        완전한 state_dict를 저장할 수 있다. 기존 checkpoint 흐름을 변경하지
        않고 EP를 투명하게 통합하기 위한 메서드.

        동작: 각 rank가 자기 expert 파라미터를 broadcast → 다른 rank의
        비담당 expert 슬롯에 학습된 가중치가 채워진다.
        """
        if self._ep_group is None:
            return

        for name, module in model.named_modules():
            if self._is_moe_layer(name, module):
                for child_name, child in module.named_children():
                    if self.expert_module_pattern in child_name and isinstance(child, nn.ModuleList):
                        self._broadcast_experts(child)

    def scatter_experts(self, model: nn.Module, device: torch.device) -> None:
        """gather 후 원래 상태로 복원: 비담당 expert를 다시 CPU + frozen."""
        self._distribute_experts(model, device)

    def _broadcast_experts(self, experts: nn.ModuleList) -> None:
        """각 expert의 파라미터를 담당 rank에서 전체 EP group으로 broadcast."""
        num_experts = len(experts)
        experts_per_gpu = num_experts // self.ep_size

        for expert_idx in range(num_experts):
            # 이 expert를 담당하는 EP rank
            owner_ep_rank = expert_idx // experts_per_gpu

            for param in experts[expert_idx].parameters():
                # broadcast 전에 모든 rank에서 같은 device에 있어야 함
                orig_device = param.device
                param.data = param.data.to("cpu")
                dist.broadcast(param.data, src=self._ep_rank_to_global(owner_ep_rank), group=self._ep_group)
                param.data = param.data.to(orig_device)

    def _ep_rank_to_global(self, ep_rank: int) -> int:
        """EP group 내 rank를 global rank로 변환한다."""
        # EP group의 첫 번째 rank = global rank에서 ep_rank 0에 해당
        global_rank = dist.get_rank()
        ep_group_start = (global_rank // self.ep_size) * self.ep_size
        return ep_group_start + ep_rank
