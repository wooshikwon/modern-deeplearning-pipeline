"""데이터 인터페이스 검증 테스트."""

from __future__ import annotations

import pytest

from mdp.cli.inference import _resolve_fields, _validate_data_interface


class TestResolveFields:
    def test_no_override(self) -> None:
        """CLI fields 없으면 체크포인트 fields 그대로."""
        result = _resolve_fields({"image": "photo", "label": "cat"}, None)
        assert result == {"image": "photo", "label": "cat"}

    def test_override(self) -> None:
        """CLI fields가 체크포인트 fields를 오버라이드."""
        result = _resolve_fields(
            {"image": "photo", "label": "category"},
            ["image=img", "label=class"],
        )
        assert result == {"image": "img", "label": "class"}

    def test_partial_override(self) -> None:
        """일부 role만 오버라이드."""
        result = _resolve_fields(
            {"image": "photo", "label": "category"},
            ["image=img"],
        )
        assert result == {"image": "img", "label": "category"}

    def test_empty_checkpoint_fields(self) -> None:
        result = _resolve_fields(None, ["image=img"])
        assert result == {"image": "img"}

    def test_both_none(self) -> None:
        result = _resolve_fields(None, None)
        assert result is None

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="형식 오류"):
            _resolve_fields(None, ["image_without_equals"])


class TestValidateDataInterface:
    def test_missing_input_field_raises(self) -> None:
        """입력 컬럼이 없으면 에러."""
        with pytest.raises(ValueError, match="image"):
            _validate_data_interface(
                {"image": "image", "label": "label"},
                ["text", "label"],
            )

    def test_missing_label_ok(self) -> None:
        """label 컬럼 없는 것은 허용 (순수 추론)."""
        _validate_data_interface(
            {"image": "image", "label": "label"},
            ["image"],
        )

    def test_missing_target_ok(self) -> None:
        """target 컬럼 없는 것도 허용."""
        _validate_data_interface(
            {"text": "text", "target": "target"},
            ["text"],
        )

    def test_all_present(self) -> None:
        """모든 컬럼이 있으면 에러 없음."""
        _validate_data_interface(
            {"image": "photo", "label": "cat"},
            ["photo", "cat", "id"],
        )

    def test_none_fields(self) -> None:
        """fields=None이면 검증 건너뜀."""
        _validate_data_interface(None, ["anything"])

    def test_error_message_includes_hint(self) -> None:
        """에러 메시지에 --fields 힌트 포함."""
        with pytest.raises(ValueError, match="--fields"):
            _validate_data_interface(
                {"image": "photo"},
                ["img", "id"],
            )
