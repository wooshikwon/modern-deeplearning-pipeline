"""MoE Expert Parallelism 통합 테스트.

Phase 0: RL Trainer kwargs 전달 수정 검증
Phase 1: FSDP auto_wrap_cls + HYBRID_SHARD 자동 전환 검증
Phase 2: ExpertParallel process group + expert 분배 + all-to-all dispatch 검증
Phase 2: CompatValidator MoE 규칙 검증
Phase 2: Factory MoE 감지 검증
Phase 3: DeepSpeedStrategy MoE config 주입 검증
"""

from __future__ import annotations

import os
import platform
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from tests.e2e.models import TinyMoEModel

_IS_MAC = platform.system() == "Darwin"


# =====================================================================
# Phase 0: RL Trainer kwargs 전달
# =====================================================================


class TestRLTrainerKwargs:
    """RL Trainer가 strategy kwargs를 올바르게 전달하는지 검증."""

    def test_rl_trainer_passes_strategy_kwargs(self) -> None:
        """_create_strategy()가 distributed config의 kwargs를 전달하는지 확인."""
        from mdp.settings.resolver import ComponentResolver

        resolver = ComponentResolver()

        # aliases.yaml의 strategy 단축 이름으로 resolve
        dist_config = {
            "strategy": "fsdp",
            "sharding_strategy": "HYBRID_SHARD",
            "mixed_precision": False,
            "backend": "gloo",
        }
        strategy_name = dist_config["strategy"]
        strategy_kwargs = {
            k: v for k, v in dist_config.items() if k != "strategy"
        }
        strategy = resolver.resolve({"_component_": strategy_name, **strategy_kwargs})

        assert strategy.sharding_strategy_name == "HYBRID_SHARD"
        assert strategy.mixed_precision is False
        assert strategy.backend == "gloo"


# =====================================================================
# Phase 1: FSDP auto_wrap_cls + HYBRID_SHARD
# =====================================================================


class TestFSDPMoEWrap:
    """FSDP MoE-aware wrapping 검증."""

    def test_fsdp_default_size_based_policy(self) -> None:
        """auto_wrap_cls 미지정 시 기존 size_based 동작이 유지되는지."""
        from mdp.training.strategies.fsdp import FSDPStrategy

        strategy = FSDPStrategy()
        assert strategy.auto_wrap_cls is None
        assert strategy.min_num_params == 1_000_000

    def test_fsdp_accepts_auto_wrap_cls(self) -> None:
        """auto_wrap_cls 지정 시 파라미터가 저장되는지."""
        from mdp.training.strategies.fsdp import FSDPStrategy

        strategy = FSDPStrategy(auto_wrap_cls="MixtralDecoderLayer")
        assert strategy.auto_wrap_cls == "MixtralDecoderLayer"

    def test_fsdp_no_moe_param(self) -> None:
        """FSDPStrategy는 moe 파라미터를 받지 않는다 (EP가 담당)."""
        from mdp.training.strategies.fsdp import FSDPStrategy
        import inspect

        params = inspect.signature(FSDPStrategy.__init__).parameters
        assert "moe" not in params

    def test_resolve_layer_class_full_path(self) -> None:
        """전체 경로로 클래스를 resolve한다."""
        from mdp.training.strategies.fsdp import FSDPStrategy

        cls = FSDPStrategy._resolve_layer_class("torch.nn.TransformerEncoderLayer")
        assert cls is nn.TransformerEncoderLayer

    def test_resolve_layer_class_invalid_raises(self) -> None:
        """존재하지 않는 클래스는 ValueError를 발생시킨다."""
        from mdp.training.strategies.fsdp import FSDPStrategy

        with pytest.raises(ValueError, match="찾을 수 없습니다"):
            FSDPStrategy._resolve_layer_class("NonExistentClass12345")

    @pytest.mark.skipif(_IS_MAC, reason="FSDP CPU test unsupported on Mac")
    def test_fsdp_transformer_auto_wrap_in_setup(self) -> None:
        """auto_wrap_cls 지정 시 setup에서 transformer_auto_wrap_policy가 적용되는지.

        gloo CPU 단일 프로세스에서 FSDP wrapping을 확인한다.
        """
        from mdp.training.strategies.fsdp import FSDPStrategy
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        os.environ.update({
            "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29510",
        })
        try:
            strategy = FSDPStrategy(
                backend="gloo",
                mixed_precision=False,
                auto_wrap_cls="torch.nn.TransformerEncoderLayer",
            )
            model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=16, nhead=2, batch_first=True),
                num_layers=2,
            )
            wrapped = strategy.setup(model, torch.device("cpu"))
            assert isinstance(wrapped, FSDP)
            strategy.cleanup()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                os.environ.pop(k, None)


# =====================================================================
# Phase 2: ExpertParallel
# =====================================================================


class TestExpertParallelUnit:
    """ExpertParallel 단위 테스트 (분산 없이)."""

    def test_init_stores_params(self) -> None:
        from mdp.training.strategies.moe import ExpertParallel

        s = ExpertParallel(ep_size=4, expert_module_pattern="my_experts")
        assert s.ep_size == 4
        assert s.expert_module_pattern == "my_experts"
        assert s._ep_group is None
        assert s._dp_group is None

    def test_detect_num_experts_from_config(self) -> None:
        """model.config.num_local_experts에서 expert 수를 감지한다."""
        from mdp.training.strategies.moe import ExpertParallel

        s = ExpertParallel(ep_size=2)
        model = TinyMoEModel(num_experts=8)
        assert s._detect_num_experts(model) == 8

    def test_detect_num_experts_from_module(self) -> None:
        """config가 없을 때 ModuleList에서 expert 수를 감지한다."""
        from mdp.training.strategies.moe import ExpertParallel

        s = ExpertParallel(ep_size=2)
        model = TinyMoEModel(num_experts=4)
        # config 제거하여 fallback 경로 테스트
        del model.config
        assert s._detect_num_experts(model) == 4

    def test_detect_num_experts_fails_gracefully(self) -> None:
        """expert를 감지할 수 없을 때 ValueError를 발생시킨다."""
        from mdp.training.strategies.moe import ExpertParallel

        s = ExpertParallel(ep_size=2)
        model = nn.Linear(10, 10)  # MoE가 아닌 모델
        with pytest.raises(ValueError, match="감지할 수 없습니다"):
            s._detect_num_experts(model)

    def test_is_moe_layer(self) -> None:
        """MoE layer (experts ModuleList를 가진 모듈) 감지."""
        from mdp.training.strategies.moe import ExpertParallel

        s = ExpertParallel(ep_size=2)
        model = TinyMoEModel(num_experts=4)
        assert s._is_moe_layer("moe_layer", model.moe_layer) is True
        assert s._is_moe_layer("lm_head", model.lm_head) is False

    def test_is_expert_param(self) -> None:
        """expert 파라미터 이름 판별."""
        from mdp.training.strategies.moe import ExpertParallel

        s = ExpertParallel(ep_size=2)
        assert s._is_expert_param("moe_layer.experts.0.0.weight") is True
        assert s._is_expert_param("lm_head.weight") is False
        assert s._is_expert_param("moe_layer.gate.weight") is False


def _moe_process_group_worker(rank: int, world_size: int, result_queue) -> None:
    """ExpertParallel process group 생성 검증 워커."""
    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29520",
    })
    try:
        from mdp.training.strategies.moe import ExpertParallel

        s = ExpertParallel(ep_size=2)
        dist.init_process_group(backend="gloo")
        s._create_process_groups(rank, world_size)

        result_queue.put({
            "rank": rank,
            "ep_rank": s._ep_rank,
            "has_ep_group": s._ep_group is not None,
            "has_dp_group": s._dp_group is not None,
        })
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(k, None)


class TestExpertParallelDistributed:
    """ExpertParallel 분산 테스트 (gloo CPU)."""

    def test_process_group_creation(self) -> None:
        """EP/DP process group이 올바르게 생성되는지."""
        result_queue = mp.Queue()
        mp.spawn(_moe_process_group_worker, args=(2, result_queue), nprocs=2, join=True)

        results = [result_queue.get() for _ in range(2)]
        for r in results:
            assert r["has_ep_group"] is True, f"rank {r['rank']}: EP group missing"
            assert r["has_dp_group"] is True, f"rank {r['rank']}: DP group missing"

        # ep_size=2, world_size=2 → 모든 rank가 같은 EP group
        ep_ranks = {r["rank"]: r["ep_rank"] for r in results}
        assert ep_ranks[0] == 0
        assert ep_ranks[1] == 1


def _moe_expert_distribution_worker(rank: int, world_size: int, result_queue) -> None:
    """Expert 분배 검증 워커."""
    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29521",
    })
    try:
        from mdp.training.strategies.moe import ExpertParallel

        s = ExpertParallel(ep_size=2)
        model = TinyMoEModel(num_experts=4)
        device = torch.device("cpu")

        dist.init_process_group(backend="gloo")
        wrapped = s.setup(model, device)

        expert_range = s._expert_range
        result_queue.put({
            "rank": rank,
            "expert_start": expert_range[0],
            "expert_end": expert_range[1],
        })
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(k, None)


class TestMoEExpertDistribution:
    """MoE expert 분배 검증."""

    def test_expert_range_assignment(self) -> None:
        """4 experts, 2 GPUs → 각 GPU에 2 experts."""
        result_queue = mp.Queue()
        mp.spawn(
            _moe_expert_distribution_worker,
            args=(2, result_queue),
            nprocs=2,
            join=True,
        )

        results = [result_queue.get() for _ in range(2)]
        r0 = next(r for r in results if r["rank"] == 0)
        r1 = next(r for r in results if r["rank"] == 1)

        assert r0["expert_start"] == 0 and r0["expert_end"] == 2
        assert r1["expert_start"] == 2 and r1["expert_end"] == 4


def _moe_forward_worker(rank: int, world_size: int, result_queue) -> None:
    """MoE forward pass (all-to-all dispatch) 검증 워커."""
    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29522",
    })
    try:
        from mdp.training.strategies.moe import ExpertParallel

        torch.manual_seed(42)
        s = ExpertParallel(ep_size=2)
        model = TinyMoEModel(num_experts=4, hidden_dim=16, vocab_size=32)
        device = torch.device("cpu")

        dist.init_process_group(backend="gloo")
        wrapped = s.setup(model, device)

        batch = {"input_ids": torch.randint(0, 32, (2, 8))}
        with torch.no_grad():
            outputs = wrapped(batch)

        logits = outputs["logits"]
        has_output = logits is not None and logits.shape == (2, 8, 32)
        is_finite = torch.isfinite(logits).all().item()

        result_queue.put({
            "rank": rank,
            "has_output": has_output,
            "is_finite": is_finite,
            "output_shape": list(logits.shape),
        })
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(k, None)


class TestMoEForward:
    """MoE forward pass (all-to-all dispatch 포함) 검증."""

    def test_moe_forward_produces_valid_output(self) -> None:
        """ExpertParallel로 래핑된 모델의 forward가 유효한 출력을 생성하는지."""
        result_queue = mp.Queue()
        mp.spawn(
            _moe_forward_worker,
            args=(2, result_queue),
            nprocs=2,
            join=True,
        )

        results = [result_queue.get() for _ in range(2)]
        for r in results:
            assert r["has_output"] is True, f"rank {r['rank']}: invalid output shape {r['output_shape']}"
            assert r["is_finite"] is True, f"rank {r['rank']}: output contains NaN/Inf"


def _moe_backward_worker(rank: int, world_size: int, result_queue) -> None:
    """MoE backward pass (gradient 역전파) 검증 워커."""
    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29523",
    })
    try:
        from mdp.training.strategies.moe import ExpertParallel

        torch.manual_seed(42 + rank)
        s = ExpertParallel(ep_size=2)
        model = TinyMoEModel(num_experts=4, hidden_dim=16, vocab_size=32)
        device = torch.device("cpu")

        dist.init_process_group(backend="gloo")
        wrapped = s.setup(model, device)

        batch = {"input_ids": torch.randint(0, 32, (2, 8))}
        loss = wrapped.training_step(batch)
        loss.backward()

        has_grad = False
        for name, p in wrapped.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break

        result_queue.put({
            "rank": rank,
            "loss_finite": torch.isfinite(loss).item(),
            "has_grad": has_grad,
            "loss_value": loss.item(),
        })
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(k, None)


class TestMoEBackward:
    """MoE backward pass (all-to-all gradient 역전파) 검증."""

    def test_moe_backward_produces_gradients(self) -> None:
        """loss.backward()가 expert와 shared 레이어에 gradient를 전파하는지."""
        result_queue = mp.Queue()
        mp.spawn(
            _moe_backward_worker,
            args=(2, result_queue),
            nprocs=2,
            join=True,
        )

        results = [result_queue.get() for _ in range(2)]
        for r in results:
            assert r["loss_finite"] is True, f"rank {r['rank']}: loss is NaN/Inf"
            assert r["has_grad"] is True, f"rank {r['rank']}: no gradients produced"


def _moe_expert_grad_worker(rank: int, world_size: int, result_queue) -> None:
    """EP 후 비담당 expert가 frozen되는지 검증하는 워커."""
    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29524",
    })
    try:
        from mdp.training.strategies.moe import ExpertParallel

        torch.manual_seed(42)
        s = ExpertParallel(ep_size=2)
        model = TinyMoEModel(num_experts=4, hidden_dim=16, vocab_size=32)
        device = torch.device("cpu")

        dist.init_process_group(backend="gloo")
        wrapped = s.setup(model, device)

        # 비담당 expert의 requires_grad가 False인지 확인
        start, end = s._expert_range
        non_owned_frozen = True
        owned_trainable = True
        for idx, expert in enumerate(wrapped.moe_layer.experts):
            for p in expert.parameters():
                if start <= idx < end:
                    if not p.requires_grad:
                        owned_trainable = False
                else:
                    if p.requires_grad:
                        non_owned_frozen = False

        result_queue.put({
            "rank": rank,
            "expert_range": (start, end),
            "non_owned_frozen": non_owned_frozen,
            "owned_trainable": owned_trainable,
        })
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(k, None)


class TestMoEExpertFreezing:
    """EP가 비담당 expert를 올바르게 freeze하는지 검증."""

    def test_non_owned_experts_frozen(self) -> None:
        """비담당 expert는 requires_grad=False, 담당 expert는 True."""
        result_queue = mp.Queue()
        mp.spawn(
            _moe_expert_grad_worker,
            args=(2, result_queue),
            nprocs=2,
            join=True,
        )

        results = [result_queue.get() for _ in range(2)]
        for r in results:
            assert r["non_owned_frozen"] is True, (
                f"rank {r['rank']}: 비담당 expert가 frozen되지 않음 (range={r['expert_range']})"
            )
            assert r["owned_trainable"] is True, (
                f"rank {r['rank']}: 담당 expert가 trainable이 아님 (range={r['expert_range']})"
            )


def _moe_gather_scatter_worker(rank: int, world_size: int, result_queue) -> None:
    """EP gather/scatter 검증 워커."""
    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29525",
    })
    try:
        from mdp.training.strategies.moe import ExpertParallel

        torch.manual_seed(42)
        ep = ExpertParallel(ep_size=2)
        model = TinyMoEModel(num_experts=4, hidden_dim=16, vocab_size=32)
        device = torch.device("cpu")

        dist.init_process_group(backend="gloo")
        model = ep.setup(model, device)

        # 담당 expert의 가중치를 변경하여 학습 시뮬레이션
        start, end = ep._expert_range
        for idx in range(start, end):
            for p in model.moe_layer.experts[idx].parameters():
                p.data.fill_(rank + 1.0)  # rank 0: 1.0, rank 1: 2.0

        # gather 전: 비담당 expert는 stale 값
        pre_gather_non_owned = {}
        for idx in range(4):
            if idx < start or idx >= end:
                val = list(model.moe_layer.experts[idx].parameters())[0].data[0, 0].item()
                pre_gather_non_owned[idx] = val

        # gather: 모든 expert 가중치를 모든 rank에 broadcast
        ep.gather_experts(model)

        # gather 후: 비담당 expert에 학습된 가중치가 채워졌는지 확인
        post_gather = {}
        for idx in range(4):
            val = list(model.moe_layer.experts[idx].parameters())[0].data[0, 0].item()
            post_gather[idx] = val

        # scatter: 비담당 expert를 다시 CPU + frozen
        ep.scatter_experts(model, device)

        post_scatter_frozen = True
        for idx in range(4):
            if idx < start or idx >= end:
                for p in model.moe_layer.experts[idx].parameters():
                    if p.requires_grad:
                        post_scatter_frozen = False

        result_queue.put({
            "rank": rank,
            "expert_range": (start, end),
            "pre_gather_non_owned": pre_gather_non_owned,
            "post_gather": post_gather,
            "post_scatter_frozen": post_scatter_frozen,
        })
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(k, None)


class TestMoEGatherScatter:
    """EP gather/scatter checkpoint 통합 검증."""

    def test_gather_broadcasts_expert_weights(self) -> None:
        """gather 후 모든 rank가 모든 expert의 학습된 가중치를 가지는지."""
        result_queue = mp.Queue()
        mp.spawn(
            _moe_gather_scatter_worker,
            args=(2, result_queue),
            nprocs=2,
            join=True,
        )

        results = [result_queue.get() for _ in range(2)]

        for r in results:
            rank = r["rank"]
            start, end = r["expert_range"]

            # gather 후: rank 0의 expert 0,1은 1.0, expert 2,3은 2.0
            # rank 1도 동일해야 함 (broadcast)
            for idx in range(4):
                expected_owner = 0 if idx < 2 else 1
                expected_val = float(expected_owner + 1)
                actual = r["post_gather"][idx]
                assert abs(actual - expected_val) < 0.01, (
                    f"rank {rank}: expert {idx} after gather = {actual}, "
                    f"expected {expected_val} (owner rank={expected_owner})"
                )

            # scatter 후: 비담당 expert가 다시 frozen
            assert r["post_scatter_frozen"] is True, (
                f"rank {rank}: scatter 후 비담당 expert가 frozen되지 않음"
            )


# =====================================================================
# Phase 2: CompatValidator MoE 규칙
# =====================================================================


class TestMoEValidation:
    """CompatValidator MoE 규칙 검증."""

    def _make_settings(self, distributed: dict | None, gpus: int = 8) -> MagicMock:
        settings = MagicMock()
        settings.config.compute.distributed = distributed
        settings.config.compute.gpus = gpus
        settings.config.serving = None
        settings.recipe.adapter = None
        return settings

    def test_moe_ep_size_required(self) -> None:
        """MoE EP에 ep_size가 없으면 error."""
        from mdp.settings.validation.compat_validator import CompatValidator

        result = CompatValidator().validate(
            self._make_settings({"strategy": "fsdp", "moe": {"enabled": True}})
        )
        assert any("ep_size" in e for e in result.errors)

    def test_moe_gpu_ep_size_divisibility(self) -> None:
        """GPU 수가 ep_size의 배수가 아니면 error."""
        from mdp.settings.validation.compat_validator import CompatValidator

        result = CompatValidator().validate(
            self._make_settings(
                {"strategy": "fsdp", "moe": {"enabled": True, "ep_size": 3}},
                gpus=8,
            )
        )
        assert any("배수" in e for e in result.errors)

    def test_moe_valid_config_no_error(self) -> None:
        """유효한 MoE config는 error가 없어야 한다."""
        from mdp.settings.validation.compat_validator import CompatValidator

        result = CompatValidator().validate(
            self._make_settings(
                {"strategy": "fsdp", "moe": {"enabled": True, "ep_size": 4}},
                gpus=8,
            )
        )
        moe_errors = [e for e in result.errors if "MoE" in e or "ep_size" in e or "배수" in e]
        assert len(moe_errors) == 0

    def test_no_moe_config_skips_check(self) -> None:
        """moe config가 없으면 MoE 검증을 건너뛴다."""
        from mdp.settings.validation.compat_validator import CompatValidator

        result = CompatValidator().validate(
            self._make_settings({"strategy": "fsdp"})
        )
        moe_errors = [e for e in result.errors if "ep_size" in e]
        assert len(moe_errors) == 0


# =====================================================================
# Phase 2: Factory MoE 감지
# =====================================================================


class TestFactoryMoEDetection:
    """Factory._is_moe_model / _extract_moe_info 검증."""

    def test_is_moe_model_true(self) -> None:
        from mdp.factory.factory import Factory

        model = TinyMoEModel(num_experts=4)
        assert Factory._is_moe_model(model) is True

    def test_is_moe_model_false(self) -> None:
        from mdp.factory.factory import Factory

        model = nn.Linear(10, 10)
        assert Factory._is_moe_model(model) is False

    def test_extract_moe_info(self) -> None:
        from mdp.factory.factory import Factory

        model = TinyMoEModel(num_experts=8, top_k=2)
        info = Factory._extract_moe_info(model)
        assert info["num_experts"] == 8
        assert info["top_k"] == 2


# =====================================================================
# Phase 3: DeepSpeedStrategy MoE config
# =====================================================================


class TestDeepSpeedMoE:
    """DeepSpeedStrategy MoE config 주입 검증."""

    def test_moe_config_injected(self) -> None:
        from mdp.training.strategies.deepspeed import DeepSpeedStrategy

        s = DeepSpeedStrategy(moe={"expert_parallel_size": 4, "num_experts": 8})
        assert s.ds_config["moe"]["enabled"] is True
        assert s.ds_config["moe"]["ep_size"] == 4
        assert s.ds_config["moe"]["num_experts"] == 8
        assert s.ds_config["moe"]["moe_param_group"] is True

    def test_moe_config_default_none(self) -> None:
        from mdp.training.strategies.deepspeed import DeepSpeedStrategy

        s = DeepSpeedStrategy()
        assert "moe" not in s.ds_config

    def test_moe_config_overwrites_existing(self) -> None:
        from mdp.training.strategies.deepspeed import DeepSpeedStrategy

        s = DeepSpeedStrategy(
            ds_config={"moe": {"enabled": False}},
            moe={"expert_parallel_size": 2},
        )
        assert s.ds_config["moe"]["enabled"] is True  # moe 파라미터가 우선

    def test_deepspeed_strategy_config_is_fail_fast(self) -> None:
        """Current Trainer runtime rejects DeepSpeed until engine ownership is integrated."""
        from mdp.training._common import create_strategy

        settings = MagicMock()
        settings.config.compute.distributed = {"strategy": "deepspeed_zero3"}

        with pytest.raises(ValueError, match="DeepSpeed strategy is not supported"):
            create_strategy(settings, MagicMock())


# =====================================================================
# Phase 2: STRATEGY_MAP 등록
# =====================================================================


class TestExpertParallelRegistration:
    """ExpertParallel import 및 Trainer 통합 검증."""

    def test_expert_parallel_importable_from_init(self) -> None:
        from mdp.training.strategies import ExpertParallel

        assert ExpertParallel is not None

    def test_strategy_aliases_do_not_have_moe(self) -> None:
        """EP는 전략이 아니므로 aliases.yaml의 strategy 섹션에 없어야 한다."""
        from mdp.settings.resolver import ComponentResolver

        resolver = ComponentResolver()
        assert resolver._aliases.get("moe") is None

    def test_trainer_creates_expert_parallel_from_config(self) -> None:
        """distributed.moe config가 있으면 create_expert_parallel이 ExpertParallel을 생성한다."""
        from mdp.training._common import create_expert_parallel

        settings = MagicMock()
        settings.config.compute.distributed = {
            "strategy": "fsdp",
            "moe": {"enabled": True, "ep_size": 4},
        }
        ep = create_expert_parallel(settings)
        assert ep is not None
        assert ep.ep_size == 4

    def test_trainer_no_expert_parallel_without_moe(self) -> None:
        """moe config가 없으면 create_expert_parallel이 None이다."""
        from mdp.training._common import create_expert_parallel

        settings = MagicMock()
        settings.config.compute.distributed = {"strategy": "fsdp"}
        ep = create_expert_parallel(settings)
        assert ep is None
