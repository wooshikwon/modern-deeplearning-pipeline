"""Global test fixtures for MDP test suite.

Adds:
  - Auto-skip logic for @pytest.mark.{gpu,distributed,fixtures,memory,distributed_cpu}
    on environments where the requirement is not met (no failure, just skip).
  - Session-scoped paths for the model/dataset fixtures produced by
    `scripts/prepare_test_fixtures.py` (consumed via MDP_TEST_FIXTURES env).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


# ────────────────────────────────────────────────────────────
# Marker auto-skip (gpu / distributed / fixtures / memory / distributed_cpu)
# ────────────────────────────────────────────────────────────

def pytest_collection_modifyitems(config, items):
    """Add skip markers based on the host environment.

    Tests carry @pytest.mark.gpu / @pytest.mark.distributed / @pytest.mark.fixtures
    / @pytest.mark.memory / @pytest.mark.distributed_cpu declaratively. We translate
    those to pytest.mark.skip(reason=...) when the required hardware/env is not present,
    so the same suite runs cleanly on:
      - CPU laptop: gpu/distributed/fixtures tests skip
      - Single-GPU instance: gpu tests run, distributed skip
      - Multi-GPU instance + MDP_TEST_FIXTURES set: everything runs
    """
    has_cuda = torch.cuda.is_available()
    n_cuda = torch.cuda.device_count() if has_cuda else 0
    fixtures_set = bool(os.environ.get("MDP_TEST_FIXTURES"))
    memory_profile = os.environ.get("MDP_MEMORY_BUDGET_PROFILE")
    run_distributed_cpu = os.environ.get("MDP_RUN_DISTRIBUTED_CPU") == "1"

    skip_no_gpu = pytest.mark.skip(reason="requires CUDA GPU (none available)")
    skip_no_multigpu = pytest.mark.skip(reason=f"requires >=2 CUDA GPUs (have {n_cuda})")
    skip_no_fixtures = pytest.mark.skip(reason="MDP_TEST_FIXTURES env var not set")
    skip_no_memory_profile = pytest.mark.skip(reason="MDP_MEMORY_BUDGET_PROFILE env var not set")
    skip_no_distributed_cpu = pytest.mark.skip(reason="MDP_RUN_DISTRIBUTED_CPU=1 not set")

    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_no_gpu)
        if "distributed" in item.keywords and n_cuda < 2:
            item.add_marker(skip_no_multigpu)
        if "fixtures" in item.keywords and not fixtures_set:
            item.add_marker(skip_no_fixtures)
        if "memory" in item.keywords and not memory_profile:
            item.add_marker(skip_no_memory_profile)
        if "distributed_cpu" in item.keywords and not run_distributed_cpu:
            item.add_marker(skip_no_distributed_cpu)


# ────────────────────────────────────────────────────────────
# Existing CPU fixtures (preserved)
# ────────────────────────────────────────────────────────────

@pytest.fixture
def device() -> torch.device:
    """Always use CPU for test reliability."""
    return torch.device("cpu")


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Temporary directory for checkpoint tests."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    return ckpt_dir


# ────────────────────────────────────────────────────────────
# Session fixtures for cloud test fixtures (MDP_TEST_FIXTURES)
# ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Root of the prepared test fixtures (MDP_TEST_FIXTURES env)."""
    p = os.environ.get("MDP_TEST_FIXTURES")
    if not p:
        pytest.skip("MDP_TEST_FIXTURES not set (run via cloud_test.sh)")
    path = Path(p)
    if not path.exists():
        pytest.skip(f"MDP_TEST_FIXTURES dir not found: {path}")
    return path


def _model_dir(fixtures_dir: Path, name: str) -> Path:
    p = fixtures_dir / "models" / name
    has_weights = (p / "model.safetensors").exists() or any(p.glob("model-*.safetensors"))
    if not has_weights:
        pytest.skip(f"fixture model {name} not present at {p}")
    return p


# Names map to the local directories produced by scripts/prepare_test_fixtures.py.
# Each fixture is the matching small + safetensors HF model + tokenizer pair.

@pytest.fixture(scope="session")
def smollm2(fixtures_dir):
    """HuggingFaceTB/SmolLM2-135M — small trained causal LM (LLaMA-arch family)."""
    return _model_dir(fixtures_dir, "smollm2")


@pytest.fixture(scope="session")
def gpt2(fixtures_dir):
    """gpt2 — classic 124M causal LM, second causal-LM family for factory routing."""
    return _model_dir(fixtures_dir, "gpt2")


@pytest.fixture(scope="session")
def bert_tiny(fixtures_dir):
    """google/bert_uncased_L-2_H-128_A-2 — safetensors mini BERT fixture."""
    return _model_dir(fixtures_dir, "bert-tiny")


@pytest.fixture(scope="session")
def vit_tiny(fixtures_dir):
    """WinKawaks/vit-tiny-patch16-224 — 5.7M trained ViT for image classification."""
    return _model_dir(fixtures_dir, "vit-tiny")


@pytest.fixture(scope="session")
def wikitext_tiny(fixtures_dir) -> Path:
    p = fixtures_dir / "data" / "wikitext-2-tiny" / "train.jsonl"
    if not p.exists():
        pytest.skip(f"wikitext fixture not found: {p}")
    return p


@pytest.fixture(scope="session")
def cifar10_tiny(fixtures_dir) -> Path:
    p = fixtures_dir / "data" / "cifar10-tiny" / "samples.pt"
    if not p.exists():
        pytest.skip(f"cifar fixture not found: {p}")
    return p


def _data_file(fixtures_dir: Path, dataset_name: str, filename: str) -> Path:
    p = fixtures_dir / "data" / dataset_name / filename
    if not p.exists():
        pytest.skip(f"{dataset_name} fixture not found: {p}")
    return p


@pytest.fixture(scope="session")
def preference_tiny(fixtures_dir) -> Path:
    return _data_file(fixtures_dir, "preference-tiny", "train.jsonl")


@pytest.fixture(scope="session")
def prompt_tiny(fixtures_dir) -> Path:
    return _data_file(fixtures_dir, "prompt-tiny", "train.jsonl")


@pytest.fixture(scope="session")
def classification_text_tiny(fixtures_dir) -> Path:
    return _data_file(fixtures_dir, "classification-text-tiny", "train.jsonl")
