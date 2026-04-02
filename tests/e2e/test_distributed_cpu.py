"""분산 학습 통합 테스트 — gloo CPU backend로 DDPStrategy/FSDPStrategy 검증.

실제 DDPStrategy/FSDPStrategy 클래스를 gloo + CPU에서 실행한다.
검증: strategy.setup → 학습 → strategy.save_checkpoint → 로딩 → strategy.cleanup.

FSDP 테스트는 Mac(MPS)에서 PyTorch FSDP 호환성 문제로 skip된다.
Linux CI에서는 정상 동작.
"""

from __future__ import annotations

import os
import platform
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.e2e.models import TinyVisionModel

_IS_MAC = platform.system() == "Darwin"


def _ddp_strategy_worker(rank: int, world_size: int, ckpt_dir: str, result_queue) -> None:
    """DDPStrategy를 gloo CPU에서 실행하는 워커."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    try:
        from mdp.training.strategies.ddp import DDPStrategy

        strategy = DDPStrategy(backend="gloo")
        model = TinyVisionModel(num_classes=2, hidden_dim=16)
        device = torch.device("cpu")

        # setup: DDP 래핑
        ddp_model = strategy.setup(model, device)
        assert hasattr(ddp_model, "module"), "DDP wrapping failed"

        # 학습 2 step
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
        for _ in range(2):
            batch = {
                "pixel_values": torch.randn(4, 3, 8, 8),
                "labels": torch.randint(0, 2, (4,)),
            }
            loss = ddp_model.module.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # rank-0 체크포인트 저장 (DDPStrategy.save_checkpoint)
        ckpt_path = str(Path(ckpt_dir) / "model.safetensors")
        strategy.save_checkpoint(ddp_model, ckpt_path)

        saved = rank == 0 and Path(ckpt_path).exists()
        result_queue.put({"rank": rank, "loss": loss.item(), "saved": saved})

        # cleanup
        strategy.cleanup()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(key, None)


def _fsdp_strategy_worker(rank: int, world_size: int, ckpt_dir: str, result_queue) -> None:
    """FSDPStrategy를 gloo CPU에서 실행하는 워커."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # MPS/CUDA 간섭 방지

    try:
        from mdp.training.strategies.fsdp import FSDPStrategy

        strategy = FSDPStrategy(
            backend="gloo",
            mixed_precision=False,
            min_num_params=1,  # TinyModel은 작으므로 모든 레이어를 FSDP wrapping
        )
        model = TinyVisionModel(num_classes=2, hidden_dim=16)
        device = torch.device("cpu")

        # setup: FSDP 래핑
        fsdp_model = strategy.setup(model, device)

        # 학습 2 step
        optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-3)
        for _ in range(2):
            batch = {
                "pixel_values": torch.randn(4, 3, 8, 8),
                "labels": torch.randint(0, 2, (4,)),
            }
            # FSDP에서는 module.training_step 대신 직접 forward
            outputs = fsdp_model(batch)
            logits = outputs["logits"]
            labels = batch["labels"]
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # rank-0 체크포인트 저장 (FSDPStrategy.save_checkpoint)
        ckpt_path = str(Path(ckpt_dir) / "model.safetensors")
        strategy.save_checkpoint(fsdp_model, ckpt_path)

        saved = rank == 0 and Path(ckpt_path).exists()
        result_queue.put({"rank": rank, "loss": loss.item(), "saved": saved})

        # cleanup
        strategy.cleanup()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(key, None)


def test_ddp_strategy_train_and_checkpoint() -> None:
    """DDPStrategy: gloo CPU에서 setup → train → save_checkpoint → load 성공."""
    with tempfile.TemporaryDirectory() as ckpt_dir:
        result_queue = mp.Queue()
        mp.spawn(_ddp_strategy_worker, args=(2, ckpt_dir, result_queue), nprocs=2, join=True)

        results = [result_queue.get() for _ in range(2)]
        rank0 = next(r for r in results if r["rank"] == 0)
        assert rank0["saved"] is True

        # safetensors 로딩 검증
        ckpt_path = Path(ckpt_dir) / "model.safetensors"
        from safetensors.torch import load_file

        state_dict = load_file(str(ckpt_path))
        model_new = TinyVisionModel(num_classes=2, hidden_dim=16)
        model_new.load_state_dict(state_dict)

        batch = {"pixel_values": torch.randn(2, 3, 8, 8), "labels": torch.tensor([0, 1])}
        assert torch.isfinite(model_new.training_step(batch))


@pytest.mark.skipif(_IS_MAC, reason="FSDP CPU test unsupported on Mac (MPS backend interference)")
def test_fsdp_strategy_train_and_checkpoint() -> None:
    """FSDPStrategy: gloo CPU에서 setup → train → save_checkpoint → load 성공."""
    with tempfile.TemporaryDirectory() as ckpt_dir:
        result_queue = mp.Queue()
        mp.spawn(_fsdp_strategy_worker, args=(2, ckpt_dir, result_queue), nprocs=2, join=True)

        results = [result_queue.get() for _ in range(2)]
        rank0 = next(r for r in results if r["rank"] == 0)
        assert rank0["saved"] is True

        # safetensors 로딩 검증
        ckpt_path = Path(ckpt_dir) / "model.safetensors"
        from safetensors.torch import load_file

        state_dict = load_file(str(ckpt_path))
        model_new = TinyVisionModel(num_classes=2, hidden_dim=16)
        model_new.load_state_dict(state_dict)

        batch = {"pixel_values": torch.randn(2, 3, 8, 8), "labels": torch.tensor([0, 1])}
        assert torch.isfinite(model_new.training_step(batch))
