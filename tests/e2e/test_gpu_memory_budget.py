"""GPU memory budget smoke tests.

These tests require CUDA, prepared fixtures, and an explicit
MDP_MEMORY_BUDGET_PROFILE. They are regression ceilings, not exact theoretical
memory assertions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from tests.e2e.conftest import make_checkpoint_callbacks_yaml, e2e_artifact_dir
from tests.e2e.test_gpu_rl import _make_dpo_yaml
from tests.e2e.test_gpu_sft import _make_yaml as _make_sft_yaml


BUDGETS = Path(__file__).resolve().parents[1] / "fixtures" / "memory_budgets.yaml"


def _budget(key: str) -> float:
    profile = os.environ.get("MDP_MEMORY_BUDGET_PROFILE")
    if not profile:
        pytest.skip("MDP_MEMORY_BUDGET_PROFILE not set")
    profiles = yaml.safe_load(BUDGETS.read_text())["profiles"]
    if profile not in profiles:
        raise AssertionError(f"unknown memory budget profile: {profile}")
    if key not in profiles[profile]:
        raise AssertionError(f"profile {profile} missing memory budget key: {key}")
    return float(profiles[profile][key])


def _reset_cuda_peak() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _peak_allocated_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3


def _measure_peak_allocated_gb(fn) -> float:
    _reset_cuda_peak()
    fn()
    torch.cuda.synchronize()
    return _peak_allocated_gb()


def _write_memory_summary(case: str, *, key: str, peak_gb: float, limit_gb: float) -> None:
    root = os.environ.get("MDP_TEST_ARTIFACT_DIR")
    if not root:
        return
    memory_dir = Path(root) / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": case,
        "budget_key": key,
        "peak_allocated_gb": round(peak_gb, 4),
        "limit_allocated_gb": limit_gb,
    }
    (memory_dir / f"{case}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    summary_path = memory_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {"cases": []}
    summary["cases"] = [item for item in summary["cases"] if item.get("case") != case]
    summary["cases"].append(payload)
    summary["cases"].sort(key=lambda item: item["case"])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))


def _assert_peak_under(case: str, key: str, peak: float) -> None:
    limit = _budget(key)
    _write_memory_summary(case, key=key, peak_gb=peak, limit_gb=limit)
    assert peak <= limit, f"{key}: peak allocated {peak:.2f} GiB > budget {limit:.2f} GiB"


@pytest.mark.gpu
@pytest.mark.fixtures
@pytest.mark.memory
def test_sft_smollm2_peak_memory(tmp_path, request, smollm2, wikitext_tiny):
    """SmolLM2 SFT CLI/trainer path stays under the selected ceiling."""
    from mdp.cli.train import run_train

    run_dir = e2e_artifact_dir(tmp_path, request.node.name, clean=True)

    def _run_sft() -> None:
        recipe_path, config_path = _make_sft_yaml(run_dir, smollm2, wikitext_tiny, gpus=1)
        callbacks_path = make_checkpoint_callbacks_yaml(run_dir)
        run_train(recipe_path, config_path, callbacks_file=callbacks_path)

    peak = _measure_peak_allocated_gb(_run_sft)
    _assert_peak_under("sft-smollm2", "sft_smollm2_allocated_gb", peak)


def _read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if len(rows) >= limit:
            break
    return rows


@pytest.mark.gpu
@pytest.mark.fixtures
@pytest.mark.memory
def test_dpo_gpt2_peak_memory(tmp_path, request, gpt2, preference_tiny):
    """DPO preference CLI/RLTrainer path stays under the selected ceiling."""
    from mdp.cli.rl_train import run_rl_train

    run_dir = e2e_artifact_dir(tmp_path, request.node.name, clean=True)

    def _run_dpo() -> None:
        recipe_path, config_path = _make_dpo_yaml(run_dir, gpt2, preference_tiny)
        callbacks_path = make_checkpoint_callbacks_yaml(run_dir, save_top_k=1)
        run_rl_train(recipe_path, config_path, callbacks_file=callbacks_path)

    peak = _measure_peak_allocated_gb(_run_dpo)
    _assert_peak_under("dpo-gpt2", "dpo_gpt2_allocated_gb", peak)


@pytest.mark.gpu
@pytest.mark.fixtures
@pytest.mark.memory
def test_inference_callback_only_gpt2_peak_memory(tmp_path, request, gpt2, prompt_tiny):
    """Callback-only inference avoids output postprocessing memory growth."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mdp.callbacks.base import BaseInferenceCallback
    from mdp.serving.inference import run_batch_inference

    class _Counter(BaseInferenceCallback):
        def __init__(self) -> None:
            self.n_batches = 0

        def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:
            self.n_batches += 1

    tok = AutoTokenizer.from_pretrained(str(gpt2))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    counter = _Counter()

    def _run_inference() -> None:
        model = AutoModelForCausalLM.from_pretrained(str(gpt2)).cuda().eval()
        rows = _read_jsonl(prompt_tiny, limit=2)
        encoded = tok(
            [row["text"] for row in rows],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        loader = DataLoader([dict(encoded)], batch_size=None)
        run_dir = e2e_artifact_dir(tmp_path, request.node.name, clean=True)
        result_path, eval_results = run_batch_inference(
            model=model,
            dataloader=loader,
            output_path=run_dir / "inference",
            output_format="jsonl",
            task="text_generation",
            device="cuda",
            callbacks=[counter],
            tokenizer=tok,
        )
        assert result_path is None
        assert eval_results == {}

    peak = _measure_peak_allocated_gb(_run_inference)

    assert counter.n_batches == 1
    _assert_peak_under("inference-gpt2-callback", "inference_gpt2_callback_allocated_gb", peak)
