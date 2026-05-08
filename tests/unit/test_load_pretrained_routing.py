"""_load_pretrained 분기 로직 유닛 테스트.

AssemblyMaterializer contract로 이동되지 않은 component+pretrained routing을 검증한다:
① _component_(HF) + pretrained → klass.from_pretrained 호출
② _component_(커스텀) + pretrained → klass(pretrained=..., **kwargs) 생성자 호출

+ BaseModel 검증: 커스텀 클래스는 pretrained 유무와 무관하게 BaseModel 필수.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from torch import Tensor, nn

from mdp.assembly.materializer import AssemblyMaterializer
from mdp.assembly.planner import AssemblyPlanner
from mdp.models.base import BaseModel
from mdp.settings.run_plan import RunPlan, RunSources
from mdp.settings.schema import (
    Config,
    DataSpec,
    MetadataSpec,
    Recipe,
    Settings,
    TrainingSpec,
)


# ── 테스트용 커스텀 모델 ──


class TinyCustomWithPretrained(BaseModel):
    """pretrained를 생성자 인자로 받는 커스텀 BaseModel.

    CriticValueModel과 동일한 패턴: pretrained URI를 받아 내부에서 처리.
    테스트에서는 실제 모델을 로드하지 않고 dummy backbone을 사용.
    """

    _block_classes = None

    def __init__(self, pretrained: str, hidden_dim: int = 8, **kwargs: Any) -> None:
        super().__init__()
        self.pretrained_source = pretrained
        self.hidden_dim = hidden_dim
        self.extra_kwargs = kwargs
        # 실제 로딩 대신 dummy backbone
        self.backbone = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x = batch["input_ids"].float()
        x = self.backbone(x)
        return {"logits": self.head(x)}

    def training_step(self, batch: dict[str, Tensor]) -> Tensor:
        return self.forward(batch)["logits"].mean()

    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        return {"val_loss": self.forward(batch)["logits"].mean().item()}


class NonBaseModelWithPretrained(nn.Module):
    """BaseModel을 상속하지 않는 커스텀 클래스. 검증 테스트용."""

    def __init__(self, pretrained: str, **kwargs: Any) -> None:
        super().__init__()
        self.pretrained_source = pretrained
        self.linear = nn.Linear(8, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


# ── 헬퍼 ──


def _make_settings(model_config: dict[str, Any]) -> Settings:
    recipe = Recipe(
        name="routing-test",
        task="text_generation",
        model=model_config,
        data=DataSpec(
            dataset={"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "/tmp/fake", "split": "train"},
            collator={"_component_": "mdp.data.collators.CausalLMCollator", "tokenizer": "gpt2"},
        ),
        training=TrainingSpec(epochs=1),
        metadata=MetadataSpec(author="test", description="routing test"),
    )
    return Settings(recipe=recipe, config=Config())


def _materializer(settings: Settings) -> AssemblyMaterializer:
    run_plan = RunPlan(
        command="train",
        mode="sft",
        settings=settings,
        sources=RunSources(),
        overrides=(),
        callback_configs=(),
        validation_scope="training",
        distributed_intent=False,
    )
    return AssemblyMaterializer(AssemblyPlanner.from_run_plan(run_plan))


# ── 경우 ②: 커스텀 BaseModel + pretrained → 생성자 호출 ──


class TestCustomBaseModelWithPretrained:
    """경우 ②: _component_가 커스텀 BaseModel이고 pretrained가 있을 때,
    from_pretrained가 아닌 생성자로 인스턴스화되는지 검증."""

    def test_pretrained_passed_as_constructor_arg(self) -> None:
        """pretrained가 생성자 인자로 전달되어야 한다."""
        settings = _make_settings({
            "_component_": "tests.unit.test_load_pretrained_routing.TinyCustomWithPretrained",
            "pretrained": "hf://fake-org/fake-model",
            "hidden_dim": 16,
        })
        materializer = _materializer(settings)
        model = materializer.materialize_policy_model()

        assert type(model).__name__ == "TinyCustomWithPretrained"
        assert model.pretrained_source == "hf://fake-org/fake-model"
        assert model.hidden_dim == 16

    def test_pretrained_uri_not_stripped(self) -> None:
        """커스텀 클래스에서는 URI를 그대로 전달한다 (파싱은 클래스 내부 책임)."""
        settings = _make_settings({
            "_component_": "tests.unit.test_load_pretrained_routing.TinyCustomWithPretrained",
            "pretrained": "local:///path/to/checkpoint",
            "hidden_dim": 8,
        })
        materializer = _materializer(settings)
        model = materializer.materialize_policy_model()

        assert model.pretrained_source == "local:///path/to/checkpoint"

    def test_extra_kwargs_forwarded(self) -> None:
        """_component_, pretrained, torch_dtype, attn_implementation 외의 키는 kwargs로 전달."""
        settings = _make_settings({
            "_component_": "tests.unit.test_load_pretrained_routing.TinyCustomWithPretrained",
            "pretrained": "hf://fake",
            "hidden_dim": 32,
        })
        materializer = _materializer(settings)
        model = materializer.materialize_policy_model()

        assert model.hidden_dim == 32


# ── 경우 ①: HF 클래스 + pretrained → from_pretrained 호출 ──


class TestHFClassWithPretrained:
    """경우 ①: _component_가 from_pretrained를 가진 HF 클래스일 때,
    from_pretrained가 호출되는지 검증."""

    def test_from_pretrained_called(self) -> None:
        """hasattr(klass, 'from_pretrained')이 True이면 from_pretrained 경로를 탄다."""
        mock_model = MagicMock(spec=nn.Module)
        mock_model.from_pretrained = MagicMock()  # spec=nn.Module이 제거하므로 재설정

        mock_cls = MagicMock()
        mock_cls.from_pretrained = MagicMock(return_value=mock_model)

        settings = _make_settings({
            "_component_": "some.hf.ModelClass",
            "pretrained": "hf://org/model-name",
        })
        materializer = _materializer(settings)

        with patch.object(
            materializer.resolver, "import_class", return_value=mock_cls,
        ):
            materializer.materialize_policy_model(skip_base_check=True)

        # from_pretrained가 URI에서 추출된 identifier로 호출되어야 한다
        mock_cls.from_pretrained.assert_called_once_with("org/model-name")

    def test_uri_protocol_stripped_for_from_pretrained(self) -> None:
        """hf:// 접두사가 제거된 identifier가 from_pretrained에 전달된다."""
        mock_cls = MagicMock()
        mock_model = MagicMock(spec=nn.Module)
        mock_model.from_pretrained = MagicMock()
        mock_cls.from_pretrained = MagicMock(return_value=mock_model)

        settings = _make_settings({
            "_component_": "some.hf.ModelClass",
            "pretrained": "hf://meta-llama/Meta-Llama-3-8B",
        })
        materializer = _materializer(settings)

        with patch.object(
            materializer.resolver, "import_class", return_value=mock_cls,
        ):
            materializer.materialize_policy_model(skip_base_check=True)

        mock_cls.from_pretrained.assert_called_once_with("meta-llama/Meta-Llama-3-8B")

    def test_no_protocol_treated_as_hf(self) -> None:
        """프로토콜 없는 pretrained는 hf://로 취급, identifier가 그대로 전달."""
        mock_cls = MagicMock()
        mock_model = MagicMock(spec=nn.Module)
        mock_model.from_pretrained = MagicMock()
        mock_cls.from_pretrained = MagicMock(return_value=mock_model)

        settings = _make_settings({
            "_component_": "some.hf.ModelClass",
            "pretrained": "bert-base-uncased",
        })
        materializer = _materializer(settings)

        with patch.object(
            materializer.resolver, "import_class", return_value=mock_cls,
        ):
            materializer.materialize_policy_model(skip_base_check=True)

        mock_cls.from_pretrained.assert_called_once_with("bert-base-uncased")


# ── BaseModel 검증 강화 ──


class TestBaseModelValidation:
    """커스텀 클래스는 pretrained 유무와 무관하게 BaseModel 상속 필수."""

    def test_custom_with_pretrained_no_basemodel_raises(self) -> None:
        """pretrained가 있어도 BaseModel 미상속 커스텀 클래스는 TypeError."""
        settings = _make_settings({
            "_component_": "tests.unit.test_load_pretrained_routing.NonBaseModelWithPretrained",
            "pretrained": "hf://fake",
        })
        materializer = _materializer(settings)

        with pytest.raises(TypeError, match="BaseModel을 상속하지 않습니다") as exc_info:
            materializer.materialize_policy_model()
        message = str(exc_info.value)
        assert "forward를 구현" in message
        assert "training_step" not in message
        assert "validation_step" not in message

    def test_custom_with_pretrained_and_basemodel_passes(self) -> None:
        """BaseModel 상속 커스텀 클래스 + pretrained는 정상 통과."""
        settings = _make_settings({
            "_component_": "tests.unit.test_load_pretrained_routing.TinyCustomWithPretrained",
            "pretrained": "hf://fake",
        })
        materializer = _materializer(settings)
        model = materializer.materialize_policy_model()

        assert isinstance(model, BaseModel)
        assert type(model).__name__ == "TinyCustomWithPretrained"
        assert model.pretrained_source == "hf://fake"

    def test_hf_class_skips_basemodel_check(self) -> None:
        """from_pretrained가 있는 HF 클래스는 BaseModel 검증을 면제받는다."""
        mock_model = MagicMock(spec=nn.Module)
        # HF 모델은 from_pretrained가 인스턴스에도 존재
        mock_model.from_pretrained = MagicMock()

        mock_cls = MagicMock()
        mock_cls.from_pretrained = MagicMock(return_value=mock_model)

        settings = _make_settings({
            "_component_": "some.hf.Model",
            "pretrained": "hf://fake",
        })
        materializer = _materializer(settings)

        with patch.object(
            materializer.resolver, "import_class", return_value=mock_cls,
        ):
            # BaseModel이 아니어도 에러가 나지 않아야 한다
            model = materializer.materialize_policy_model()
            assert model is mock_model
