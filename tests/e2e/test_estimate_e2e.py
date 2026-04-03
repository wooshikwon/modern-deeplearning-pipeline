"""메모리 추정기 테스트 — MemoryEstimator의 추정 로직 검증.

알려진 모델 크기로 추정이 합리적인 범위인지 확인.
"""

from __future__ import annotations

from tests.e2e.conftest import make_test_settings
from mdp.utils.estimator import MemoryEstimator


def test_estimate_returns_all_fields() -> None:
    """MemoryEstimator.estimate()가 필수 필드를 모두 반환하는지."""
    settings = make_test_settings(epochs=1)
    estimator = MemoryEstimator()
    result = estimator.estimate(settings)

    required_keys = {
        "model_mem_gb", "gradient_mem_gb", "optimizer_mem_gb",
        "activation_mem_gb", "total_mem_gb",
        "suggested_gpus", "suggested_strategy",
    }
    assert required_keys.issubset(result.keys())
    assert all(isinstance(result[k], (int, float)) for k in required_keys if k != "suggested_strategy")
    assert isinstance(result["suggested_strategy"], str)


def test_estimate_small_model_suggests_single_gpu() -> None:
    """작은 모델(TinyVisionModel)에 대해 단일 GPU를 추천하는지."""
    settings = make_test_settings(epochs=1)
    estimator = MemoryEstimator()
    result = estimator.estimate(settings)

    # class_path에서 모델 크기를 추정할 수 없으면 기본값 100M 사용
    # 100M 모델의 총 메모리는 ~2.5GB (fp32) → 24GB GPU 1대에 충분
    assert result["suggested_gpus"] == 1
    assert result["suggested_strategy"] == "none"
    assert result["total_mem_gb"] < 24.0


def test_estimate_precision_affects_memory() -> None:
    """precision이 메모리 추정에 영향을 미치는지."""
    estimator = MemoryEstimator()

    settings_fp32 = make_test_settings(precision="fp32")
    settings_bf16 = make_test_settings(precision="bf16")

    result_fp32 = estimator.estimate(settings_fp32)
    result_bf16 = estimator.estimate(settings_bf16)

    # bf16은 모델 메모리가 fp32의 절반이어야 함
    assert result_bf16["model_mem_gb"] < result_fp32["model_mem_gb"]


def test_estimate_gradient_checkpointing_reduces_activation() -> None:
    """gradient_checkpointing이 활성화 메모리를 줄이는지."""
    from mdp.settings.schema import (
        Config, DataSpec, MetadataSpec, ModelSpec, Recipe, Settings, TrainingSpec,
    )

    def _settings(grad_ckpt: bool) -> Settings:
        recipe = Recipe(
            name="est-test",
            task="image_classification",
            model=ModelSpec(class_path="tests.e2e.models.TinyVisionModel"),
            data=DataSpec(source="/tmp/fake", label_strategy="causal"),
            training=TrainingSpec(epochs=1, gradient_checkpointing=grad_ckpt),
            optimizer={"_component_": "AdamW", "lr": 1e-3},
            metadata=MetadataSpec(author="test", description="test"),
        )
        return Settings(recipe=recipe, config=Config())

    estimator = MemoryEstimator()
    result_no_ckpt = estimator.estimate(_settings(False))
    result_ckpt = estimator.estimate(_settings(True))

    assert result_ckpt["activation_mem_gb"] < result_no_ckpt["activation_mem_gb"]
