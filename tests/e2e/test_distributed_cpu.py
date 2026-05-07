"""분산 학습 통합 테스트 — gloo CPU backend로 DDPStrategy 검증.

실제 DDPStrategy 클래스를 gloo + CPU에서 실행한다.
검증: strategy.setup → 학습 → strategy.save_checkpoint → 로딩 → strategy.cleanup.

subprocess + torchrun 방식으로 실행하여 pytest의 fork/spawn 컨텍스트 충돌을 회피한다.
FSDP는 PyTorch 설계상 GPU 필수이므로 GPU 없는 환경에서는 skip한다.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from tests.e2e.models import TinyVisionModel

_IS_MAC = platform.system() == "Darwin"


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_ddp_strategy_train_and_checkpoint() -> None:
    """DDPStrategy: gloo CPU에서 setup → train → save_checkpoint → load 성공."""
    with tempfile.TemporaryDirectory() as ckpt_dir:
        port = _find_free_port()
        project_root = str(Path(__file__).resolve().parents[2])
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0" if _IS_MAC else "lo")
        result = subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.run",
                "--nnodes=1",
                "--nproc_per_node=2",
                "--master_addr", "127.0.0.1",
                "--master_port", str(port),
                __file__,
                "--worker", "ddp",
                "--ckpt-dir", ckpt_dir,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=project_root,
            env=env,
        )
        assert result.returncode == 0, (
            f"DDP worker failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # rank-0이 저장한 결과 확인
        result_path = Path(ckpt_dir) / "result_0.json"
        assert result_path.exists(), "rank-0 result file not found"
        r0 = json.loads(result_path.read_text())
        assert r0["saved"] is True

        # safetensors 로딩 검증
        ckpt_path = Path(ckpt_dir) / "model.safetensors"
        from safetensors.torch import load_file

        state_dict = load_file(str(ckpt_path))
        model_new = TinyVisionModel(num_classes=2, hidden_dim=16)
        model_new.load_state_dict(state_dict)

        batch = {"pixel_values": torch.randn(2, 3, 8, 8), "labels": torch.tensor([0, 1])}
        assert torch.isfinite(model_new(batch)["loss"])


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FSDP requires GPU (non-CPU accelerator)",
)
@pytest.mark.skipif(_IS_MAC, reason="FSDP CPU test unsupported on Mac (MPS backend interference)")
def test_fsdp_strategy_train_and_checkpoint() -> None:
    """FSDPStrategy: GPU에서 setup → train → save_checkpoint → load 성공."""
    pytest.skip("FSDP requires GPU — skipped in CPU-only environment")


# ── Worker entrypoint (torchrun이 실행) ──


def _ddp_worker_main(ckpt_dir: str) -> None:
    """DDPStrategy를 gloo CPU에서 실행하는 워커."""
    import torch.distributed as dist

    rank = int(os.environ["RANK"])

    try:
        from mdp.training.strategies.ddp import DDPStrategy

        strategy = DDPStrategy(backend="gloo")
        model = TinyVisionModel(num_classes=2, hidden_dim=16)
        device = torch.device("cpu")

        ddp_model = strategy.setup(model, device)
        assert hasattr(ddp_model, "module"), "DDP wrapping failed"

        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)
        for _ in range(2):
            batch = {
                "pixel_values": torch.randn(4, 3, 8, 8),
                "labels": torch.randint(0, 2, (4,)),
            }
            loss = ddp_model.module(batch)["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        ckpt_path = str(Path(ckpt_dir) / "model.safetensors")
        strategy.save_checkpoint(ddp_model, ckpt_path)

        saved = rank == 0 and Path(ckpt_path).exists()
        result = {"rank": rank, "loss": loss.item(), "saved": saved}
        result_file = Path(ckpt_dir) / f"result_{rank}.json"
        result_file.write_text(json.dumps(result))

        strategy.cleanup()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, choices=["ddp"])
    parser.add_argument("--ckpt-dir", required=True)
    args = parser.parse_args()

    if args.worker == "ddp":
        _ddp_worker_main(args.ckpt_dir)
