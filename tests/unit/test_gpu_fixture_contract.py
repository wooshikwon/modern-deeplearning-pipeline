"""GPU fixture producer/consumer contract tests.

These tests do not require CUDA or downloaded fixtures. They validate the local
Python contracts that decide which cloud fixtures are produced and consumed.
"""

from __future__ import annotations

import inspect
import importlib
import json
import pkgutil
import subprocess
from pathlib import Path

import pytest
import yaml

from mdp.settings.schema import Config, Recipe
from mdp.settings.loader import SettingsLoader
from scripts.prepare_test_fixtures import DATASETS, TINY_MODELS, cache_dataset_slice


DATASET_FIXTURE_NAMES = {
    "wikitext-2-tiny": "wikitext_tiny",
    "cifar10-tiny": "cifar10_tiny",
    "preference-tiny": "preference_tiny",
    "prompt-tiny": "prompt_tiny",
    "classification-text-tiny": "classification_text_tiny",
}

CLOUD_TEST_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cloud_test.sh"


def _fixture_name(local_model_name: str) -> str:
    return local_model_name.replace("-", "_")


def _gpu_test_modules():
    import tests.e2e as e2e

    for module_info in pkgutil.iter_modules(e2e.__path__):
        if module_info.name.startswith("test_gpu_"):
            yield importlib.import_module(f"tests.e2e.{module_info.name}")


def test_tiny_model_names_have_matching_session_fixtures() -> None:
    """Every model produced by prepare_test_fixtures has a pytest fixture."""
    import tests.conftest as conftest

    produced = {local_name for local_name, _repo, _family in TINY_MODELS}
    missing = [
        _fixture_name(local_name)
        for local_name in produced
        if not hasattr(conftest, _fixture_name(local_name))
    ]

    assert not missing, f"missing pytest fixtures for TINY_MODELS: {missing}"


def test_no_stale_clip_fixture_after_clip_model_removed() -> None:
    """The CLIP fixture was removed from producer and must not remain as consumer."""
    import tests.conftest as conftest

    produced = {local_name for local_name, _repo, _family in TINY_MODELS}

    assert "clip-base" not in produced
    assert not hasattr(conftest, "clip_base")


def test_gpu_tests_only_request_produced_fixtures() -> None:
    """GPU test function parameters must refer to produced model/data fixtures."""
    produced_fixtures = {_fixture_name(local_name) for local_name, _repo, _family in TINY_MODELS}
    allowed_non_model = {"tmp_path", "request", *DATASET_FIXTURE_NAMES.values()}
    requested: set[str] = set()

    for module in _gpu_test_modules():
        for name, obj in vars(module).items():
            if name.startswith("test_") and callable(obj):
                requested.update(inspect.signature(obj).parameters)

    requested_model_fixtures = requested - allowed_non_model
    assert requested_model_fixtures <= produced_fixtures, (
        f"GPU tests request fixtures not produced by prepare_test_fixtures: "
        f"{sorted(requested_model_fixtures - produced_fixtures)}"
    )


def test_marker_skip_reasons_preserve_hardware_boundaries() -> None:
    """GPU/cloud markers skip only when their declared boundary is unavailable."""
    import tests.conftest as conftest

    laptop = conftest._marker_skip_reasons(
        has_cuda=False,
        n_cuda=0,
        fixtures_set=False,
        memory_profile=None,
        run_distributed_cpu=False,
    )
    cloud_single_gpu = conftest._marker_skip_reasons(
        has_cuda=True,
        n_cuda=1,
        fixtures_set=True,
        memory_profile="cuda_24gb",
        run_distributed_cpu=True,
    )
    cloud_multi_gpu = conftest._marker_skip_reasons(
        has_cuda=True,
        n_cuda=2,
        fixtures_set=True,
        memory_profile="cuda_24gb",
        run_distributed_cpu=True,
    )

    assert laptop == {
        "gpu": "requires CUDA GPU (none available)",
        "distributed": "requires >=2 CUDA GPUs (have 0)",
        "fixtures": "MDP_TEST_FIXTURES env var not set",
        "memory": "MDP_MEMORY_BUDGET_PROFILE env var not set",
        "distributed_cpu": "MDP_RUN_DISTRIBUTED_CPU=1 not set",
    }
    assert cloud_single_gpu == {"distributed": "requires >=2 CUDA GPUs (have 1)"}
    assert cloud_multi_gpu == {}


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["--dry-run"],
            "tests/unit/test_gpu_fixture_contract.py tests/e2e/test_gpu_sft.py "
            "tests/e2e/test_gpu_inference.py -m not distributed and not memory",
        ),
        (["--suite", "gpu", "--dry-run"], "-m gpu and fixtures and not distributed and not memory"),
        (["--suite", "distributed", "--dry-run"], "-m distributed and fixtures"),
        (["--suite", "memory", "--memory-profile", "cuda_24gb", "--dry-run"], "-m memory"),
    ],
)
def test_cloud_test_suite_dry_run_preserves_pytest_selection(args: list[str], expected: str) -> None:
    """cloud_test.sh suite names keep selecting the documented hardware boundary."""
    result = subprocess.run(
        ["bash", str(CLOUD_TEST_SCRIPT), *args],
        cwd=CLOUD_TEST_SCRIPT.parents[1],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert result.returncode == 0, result.stdout
    assert expected in result.stdout


def test_tiny_model_families_are_supported_by_verify_models() -> None:
    """Fixture script should only declare families with local verify paths."""
    families = {family for _local_name, _repo, family in TINY_MODELS}

    assert families <= {"causal-lm", "encoder", "vision"}


def test_memory_budget_profiles_have_required_keys() -> None:
    """Memory suite profiles expose all budget ceilings used by tests."""
    budget_path = Path(__file__).resolve().parent.parent / "fixtures" / "memory_budgets.yaml"
    raw = yaml.safe_load(budget_path.read_text())
    required = {
        "sft_smollm2_allocated_gb",
        "dpo_gpt2_allocated_gb",
        "inference_gpt2_callback_allocated_gb",
    }

    profiles = raw.get("profiles")
    assert isinstance(profiles, dict) and profiles
    for name, profile in profiles.items():
        assert required <= set(profile), f"{name}: missing {sorted(required - set(profile))}"
        assert all(float(profile[key]) > 0 for key in required)


def test_e2e_artifact_dir_uses_stable_root_when_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cloud e2e artifacts are grouped under MDP_TEST_ARTIFACT_DIR when set."""
    from tests.e2e.conftest import e2e_artifact_dir

    root = tmp_path / "artifacts"
    monkeypatch.setenv("MDP_TEST_ARTIFACT_DIR", str(root))

    path = e2e_artifact_dir(tmp_path / "scratch", "test case/name", "ckpt")

    assert path == root / "tests" / "test_case_name" / "ckpt"
    assert path.exists()


def test_dataset_names_have_matching_session_fixtures() -> None:
    """Every dataset produced by prepare_test_fixtures has a pytest fixture."""
    import tests.conftest as conftest

    expected = set(DATASET_FIXTURE_NAMES.values())
    produced = {DATASET_FIXTURE_NAMES[name] for name in DATASETS}

    assert produced == expected
    assert all(hasattr(conftest, name) for name in expected)


@pytest.mark.parametrize(
    ("dataset_name", "required_keys"),
    [
        ("preference-tiny", {"chosen", "rejected"}),
        ("prompt-tiny", {"text"}),
        ("classification-text-tiny", {"text", "label"}),
    ],
)
def test_synthetic_dataset_fixture_schema(
    tmp_path: Path,
    dataset_name: str,
    required_keys: set[str],
) -> None:
    """Synthetic fixture slices have the schema expected by later e2e units."""
    info = cache_dataset_slice(tmp_path, dataset_name, DATASETS[dataset_name])
    rows = [json.loads(line) for line in (Path(info["path"]) / "train.jsonl").read_text().splitlines()]

    assert len(rows) == DATASETS[dataset_name]["n_rows"]
    assert required_keys <= rows[0].keys()


def _assert_gpu_sft_yaml_contract(recipe_path: str, config_path: str) -> None:
    raw_recipe = yaml.safe_load(Path(recipe_path).read_text())
    raw_config = yaml.safe_load(Path(config_path).read_text())

    recipe = Recipe(**raw_recipe)
    config = Config(**raw_config)
    settings = SettingsLoader().load_training_settings(recipe_path, config_path)

    assert recipe.loss is None
    assert settings.recipe.loss is None
    assert recipe.model.component == "AutoModelForCausalLM"
    assert recipe.data.dataset.kwargs["padding"] == "max_length"
    assert recipe.data.dataset.kwargs["truncation"] is True
    assert recipe.data.collator.component == "CausalLMCollator"
    assert config.storage.checkpoint_dir


def test_single_gpu_sft_yaml_dry_run_contract(tmp_path: Path) -> None:
    """Single-GPU SFT helper emits a forward-native HF loss recipe shape."""
    from tests.e2e.test_gpu_sft import _make_yaml

    recipe_path, config_path = _make_yaml(
        tmp_path,
        tmp_path / "models" / "smollm2",
        tmp_path / "data" / "wikitext-2-tiny" / "train.jsonl",
        gpus=1,
    )

    _assert_gpu_sft_yaml_contract(recipe_path, config_path)


def test_distributed_gpu_sft_yaml_dry_run_contract(tmp_path: Path) -> None:
    """DDP SFT helper emits the same forward-native loss recipe shape."""
    from tests.e2e.test_gpu_distributed_sft import _make_yaml

    recipe_path, config_path = _make_yaml(
        tmp_path,
        tmp_path / "models" / "smollm2",
        tmp_path / "data" / "wikitext-2-tiny" / "train.jsonl",
        gpus=2,
    )

    _assert_gpu_sft_yaml_contract(recipe_path, config_path)
