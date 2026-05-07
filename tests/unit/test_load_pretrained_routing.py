"""_load_pretrained 분기 로직 유닛 테스트.

4가지 경우를 검증한다:
① _component_ 없음 + pretrained → PretrainedResolver가 클래스 추론
② _component_ 있음 + pretrained 없음 → 생성자 호출 (랜덤 초기화)
③ _component_(HF) + pretrained → klass.from_pretrained 호출
④ _component_(커스텀) + pretrained → klass(pretrained=..., **kwargs) 생성자 호출

+ BaseModel 검증: 커스텀 클래스는 pretrained 유무와 무관하게 BaseModel 필수.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor, nn

from mdp.factory.factory import Factory, _ModelLoadRoute, _decide_model_load_route
from mdp.models.base import BaseModel
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


# ── 경우 ④: 커스텀 BaseModel + pretrained → 생성자 호출 ──


class TestModelLoadRouteDecision:
    """Factory의 model load route pure helper 계약을 고정한다."""

    def test_component_with_pretrained_uses_component_from_pretrained_route(self) -> None:
        route = _decide_model_load_route(
            {"_component_": "some.Model", "pretrained": "hf://org/model"},
            None,
        )

        assert route is _ModelLoadRoute.COMPONENT_FROM_PRETRAINED

    def test_component_only_uses_component_constructor_route(self) -> None:
        route = _decide_model_load_route({"_component_": "some.Model"}, None)

        assert route is _ModelLoadRoute.COMPONENT_CONSTRUCTOR

    def test_pretrained_only_uses_pretrained_resolver_route(self) -> None:
        route = _decide_model_load_route({"pretrained": "hf://org/model"}, None)

        assert route is _ModelLoadRoute.PRETRAINED_RESOLVER

    def test_qlora_adapter_uses_qlora_route(self) -> None:
        route = _decide_model_load_route(
            {"pretrained": "hf://Qwen/Qwen2.5-7B", "torch_dtype": "bfloat16"},
            {"_component_": "QLoRA", "target": ["attn.q", "attn.v"]},
        )

        assert route is _ModelLoadRoute.QLORA

    def test_missing_model_source_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="_component_ 또는 pretrained"):
            _decide_model_load_route({}, None)


class TestCustomBaseModelWithPretrained:
    """경우 ④: _component_가 커스텀 BaseModel이고 pretrained가 있을 때,
    from_pretrained가 아닌 생성자로 인스턴스화되는지 검증."""

    def test_pretrained_passed_as_constructor_arg(self) -> None:
        """pretrained가 생성자 인자로 전달되어야 한다."""
        settings = _make_settings({
            "_component_": "tests.unit.test_load_pretrained_routing.TinyCustomWithPretrained",
            "pretrained": "hf://fake-org/fake-model",
            "hidden_dim": 16,
        })
        factory = Factory(settings)
        model = factory.create_model()

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
        factory = Factory(settings)
        model = factory.create_model()

        assert model.pretrained_source == "local:///path/to/checkpoint"

    def test_extra_kwargs_forwarded(self) -> None:
        """_component_, pretrained, torch_dtype, attn_implementation 외의 키는 kwargs로 전달."""
        settings = _make_settings({
            "_component_": "tests.unit.test_load_pretrained_routing.TinyCustomWithPretrained",
            "pretrained": "hf://fake",
            "hidden_dim": 32,
        })
        factory = Factory(settings)
        model = factory.create_model()

        assert model.hidden_dim == 32


# ── 경우 ③: HF 클래스 + pretrained → from_pretrained 호출 ──


class TestHFClassWithPretrained:
    """경우 ③: _component_가 from_pretrained를 가진 HF 클래스일 때,
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
        factory = Factory(settings)

        with patch.object(
            factory.resolver, "import_class", return_value=mock_cls,
        ):
            model = factory.create_model(skip_base_check=True)

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
        factory = Factory(settings)

        with patch.object(
            factory.resolver, "import_class", return_value=mock_cls,
        ):
            factory.create_model(skip_base_check=True)

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
        factory = Factory(settings)

        with patch.object(
            factory.resolver, "import_class", return_value=mock_cls,
        ):
            factory.create_model(skip_base_check=True)

        mock_cls.from_pretrained.assert_called_once_with("bert-base-uncased")


# ── 경우 ②: _component_만, pretrained 없음 → 생성자 호출 ──


class TestComponentOnlyNoPretrained:
    """경우 ②: pretrained 없이 _component_만 있으면 생성자 호출."""

    def test_constructor_called_without_pretrained(self) -> None:
        settings = _make_settings({
            "_component_": "tests.e2e.models.TinyVisionModel",
            "num_classes": 3,
            "hidden_dim": 8,
        })
        factory = Factory(settings)
        model = factory.create_model()

        from tests.e2e.models import TinyVisionModel
        assert isinstance(model, TinyVisionModel)
        assert model.num_classes == 3


# ── 경우 ①: _component_ 없음 + pretrained → PretrainedResolver ──


class TestPretrainedOnlyNoComponent:
    """경우 ①: _component_가 없고 pretrained만 있으면 PretrainedResolver 위임."""

    def test_resolver_called_when_no_component(self) -> None:
        mock_model = MagicMock(spec=nn.Module)

        settings = _make_settings({
            "pretrained": "hf://some-org/some-model",
        })
        factory = Factory(settings)

        with patch(
            "mdp.models.pretrained.PretrainedResolver.load",
            return_value=mock_model,
        ) as mock_load:
            model = factory.create_model(skip_base_check=True)

        mock_load.assert_called_once_with("hf://some-org/some-model")
        assert model is mock_model


# ── BaseModel 검증 강화 ──


class TestBaseModelValidation:
    """커스텀 클래스는 pretrained 유무와 무관하게 BaseModel 상속 필수."""

    def test_custom_with_pretrained_no_basemodel_raises(self) -> None:
        """pretrained가 있어도 BaseModel 미상속 커스텀 클래스는 TypeError."""
        settings = _make_settings({
            "_component_": "tests.unit.test_load_pretrained_routing.NonBaseModelWithPretrained",
            "pretrained": "hf://fake",
        })
        factory = Factory(settings)

        with pytest.raises(TypeError, match="BaseModel을 상속하지 않습니다") as exc_info:
            factory.create_model()
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
        factory = Factory(settings)
        model = factory.create_model()

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
        factory = Factory(settings)

        with patch.object(
            factory.resolver, "import_class", return_value=mock_cls,
        ):
            # BaseModel이 아니어도 에러가 나지 않아야 한다
            model = factory.create_model()
            assert model is mock_model
