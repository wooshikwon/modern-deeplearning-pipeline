"""BaseModel._block_classes 계약 검증 테스트.

_block_classes 선언을 강제하는 __init_subclass__ 메커니즘과
_inherit_block_classes helper의 동작을 검증한다.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor, nn

from mdp.models.base import BaseModel


# ── _block_classes 미선언 시 TypeError ──


class TestBlockClassesContract:
    """_block_classes 필수 선언 계약을 검증한다."""

    def test_missing_block_classes_raises_type_error(self) -> None:
        """_block_classes를 선언하지 않은 서브클래스는 클래스 정의 시점에 TypeError."""
        with pytest.raises(TypeError, match="_block_classes를 선언해야 합니다"):

            class BadModel(BaseModel):
                # _block_classes 미선언
                def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
                    return {}

                def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
                    return {}

    def test_block_classes_none_is_allowed(self) -> None:
        """_block_classes = None 선언은 TypeError 없이 허용된다."""

        class SimpleModel(BaseModel):
            _block_classes = None

            def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
                return {}

            def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
                return {}

        # 클래스가 정상적으로 정의됨
        assert SimpleModel._block_classes is None

    def test_forward_and_block_classes_only_model_is_allowed(self) -> None:
        """forward와 _block_classes만 구현한 BaseModel 서브클래스가 인스턴스화된다."""

        class ForwardOnlyModel(BaseModel):
            _block_classes = None

            def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
                return {"loss": torch.tensor(0.0)}

        model = ForwardOnlyModel()

        assert model._block_classes is None
        with pytest.raises(NotImplementedError, match="validation_step"):
            model.validation_step({})

    def test_forward_is_still_required(self) -> None:
        """forward 미구현 BaseModel 서브클래스는 여전히 abstract다."""

        class MissingForwardModel(BaseModel):
            _block_classes = None

        with pytest.raises(TypeError):
            MissingForwardModel()

    def test_block_classes_set_is_allowed(self) -> None:
        """_block_classes = {"LlamaDecoderLayer"} 선언은 TypeError 없이 허용된다."""

        class LLMModel(BaseModel):
            _block_classes = {"LlamaDecoderLayer"}

            def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
                return {}

            def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
                return {}

        assert LLMModel._block_classes == {"LlamaDecoderLayer"}

    def test_block_classes_multiple_classes(self) -> None:
        """복수 블록 클래스 (VLM 등) 선언도 허용된다."""

        class VLMModel(BaseModel):
            _block_classes = {"LlamaDecoderLayer", "CLIPEncoderLayer"}

            def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
                return {}

            def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
                return {}

        assert VLMModel._block_classes == {"LlamaDecoderLayer", "CLIPEncoderLayer"}

    def test_error_message_includes_class_name(self) -> None:
        """에러 메시지에 서브클래스 이름이 포함된다."""
        with pytest.raises(TypeError, match="MyBrokenModel"):

            class MyBrokenModel(BaseModel):
                def forward(self, batch):
                    return {}

                def validation_step(self, batch):
                    return {}


# ── _inherit_block_classes ──


class TestInheritBlockClasses:
    """_inherit_block_classes helper 동작을 검증한다."""

    def test_inherit_from_child_with_block_classes(self) -> None:
        """자식 모듈의 _block_classes에서 상속한다 (MDP 계약 우선)."""

        class WrapperModel(BaseModel):
            _block_classes = None  # placeholder

            def __init__(self) -> None:
                super().__init__()
                # _block_classes를 가진 자식 모듈 시뮬레이션
                self.backbone = nn.Linear(8, 8)
                self.backbone._block_classes = {"FakeDecoderLayer"}
                self._inherit_block_classes()

            def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
                return {}

            def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
                return {}

        model = WrapperModel()
        assert model._block_classes == {"FakeDecoderLayer"}

    def test_inherit_from_hf_no_split_modules(self) -> None:
        """자식 모듈의 _no_split_modules(HF)에서 상속한다."""

        class HFWrapperModel(BaseModel):
            _block_classes = None

            def __init__(self) -> None:
                super().__init__()
                self.backbone = nn.Linear(8, 8)
                self.backbone._no_split_modules = ["LlamaDecoderLayer"]
                self._inherit_block_classes()

            def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
                return {}

            def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
                return {}

        model = HFWrapperModel()
        assert model._block_classes == {"LlamaDecoderLayer"}

    def test_mdp_block_classes_takes_priority_over_hf(self) -> None:
        """_block_classes와 _no_split_modules가 모두 있으면 _block_classes 우선."""

        class PriorityModel(BaseModel):
            _block_classes = None

            def __init__(self) -> None:
                super().__init__()
                self.backbone = nn.Linear(8, 8)
                self.backbone._block_classes = {"MdpBlock"}
                self.backbone._no_split_modules = ["HfBlock"]
                self._inherit_block_classes()

            def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
                return {}

            def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
                return {}

        model = PriorityModel()
        assert model._block_classes == {"MdpBlock"}

    def test_no_inherit_source_leaves_none(self) -> None:
        """자식 모듈에 _block_classes도 _no_split_modules도 없으면 변경 없음."""

        class PlainWrapperModel(BaseModel):
            _block_classes = None

            def __init__(self) -> None:
                super().__init__()
                self.backbone = nn.Linear(8, 8)
                self._inherit_block_classes()

            def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
                return {}

            def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
                return {}

        model = PlainWrapperModel()
        assert PlainWrapperModel._block_classes is None

    def test_inherit_sets_instance_attribute_not_class(self) -> None:
        """같은 클래스의 두 인스턴스가 각각 다른 _block_classes를 설정해도 간섭 없음.

        _inherit_block_classes()는 인스턴스 속성(self._block_classes)을 설정하므로
        같은 클래스의 다른 인스턴스에 영향을 주지 않아야 한다.
        """

        class MultiBackboneModel(BaseModel):
            _block_classes = None  # 클래스 수준 placeholder

            def __init__(self, block_name: str) -> None:
                super().__init__()
                self.backbone = nn.Linear(8, 8)
                self.backbone._block_classes = {block_name}
                self._inherit_block_classes()

            def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
                return {}

            def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
                return {}

        m1 = MultiBackboneModel("LlamaDecoderLayer")
        m2 = MultiBackboneModel("ViTLayer")

        # 각 인스턴스는 자신만의 _block_classes를 가진다
        assert m1._block_classes == {"LlamaDecoderLayer"}
        assert m2._block_classes == {"ViTLayer"}
        # 클래스 속성은 원래 placeholder(None)를 유지한다
        assert MultiBackboneModel._block_classes is None
