"""mdp.cli._override 유닛 테스트."""

import pytest

from mdp.cli._override import apply_overrides, parse_value


class TestParseValue:
    def test_null(self):
        assert parse_value("null") is None
        assert parse_value("None") is None

    def test_bool(self):
        assert parse_value("true") is True
        assert parse_value("false") is False
        assert parse_value("True") is True

    def test_int(self):
        assert parse_value("42") == 42
        assert parse_value("-1") == -1

    def test_float(self):
        assert parse_value("0.1") == 0.1
        assert parse_value("3.14") == 3.14

    def test_json_array(self):
        assert parse_value("[1,2,3]") == [1, 2, 3]

    def test_json_object(self):
        assert parse_value('{"a":1}') == {"a": 1}

    def test_string(self):
        assert parse_value("hello") == "hello"
        assert parse_value("gpt2") == "gpt2"

    def test_invalid_json_falls_back_to_str(self):
        assert parse_value("[broken") == "[broken"


class TestApplyOverrides:
    def test_simple(self):
        d = {"training": {"epochs": 10}}
        apply_overrides(d, ["training.epochs=0.1"])
        assert d["training"]["epochs"] == 0.1

    def test_nested_create(self):
        d = {}
        apply_overrides(d, ["a.b.c=hello"])
        assert d["a"]["b"]["c"] == "hello"

    def test_multiple_overrides(self):
        d = {"x": 1, "y": {"z": 2}}
        apply_overrides(d, ["x=10", "y.z=20"])
        assert d["x"] == 10
        assert d["y"]["z"] == 20

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="올바른 형식"):
            apply_overrides({}, ["no_equals"])

    def test_bool_override(self):
        d = {"gradient_checkpointing": False}
        apply_overrides(d, ["gradient_checkpointing=true"])
        assert d["gradient_checkpointing"] is True

    def test_null_override(self):
        d = {"adapter": {"method": "lora"}}
        apply_overrides(d, ["adapter=null"])
        assert d["adapter"] is None
