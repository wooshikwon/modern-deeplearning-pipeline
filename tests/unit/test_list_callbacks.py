"""Unit tests for _list_callbacks() type classification (U8)."""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

import pytest


# ── _classify helper ──────────────────────────────────────────────────────────


def test_classify_intervention_returns_int_tag():
    from mdp.cli.list_cmd import _classify

    result = _classify("mdp.callbacks.interventions.residual_add.ResidualAdd")
    assert result == "[Int]"


def test_classify_intervention_logit_bias():
    from mdp.cli.list_cmd import _classify

    result = _classify("mdp.callbacks.interventions.logit_bias.LogitBias")
    assert result == "[Int]"


def test_classify_observational_returns_obs_tag():
    from mdp.cli.list_cmd import _classify

    result = _classify("mdp.callbacks.inference.DefaultOutputCallback")
    assert result == "[Obs]"


def test_classify_training_callback_returns_train_tag():
    from mdp.cli.list_cmd import _classify

    result = _classify("mdp.training.callbacks.checkpoint.ModelCheckpoint")
    assert result == "[Train]"


def test_classify_bad_path_returns_question_mark():
    from mdp.cli.list_cmd import _classify

    result = _classify("nonexistent.module.SomeClass")
    assert result == "[?]"


# ── _classify_to_type_str ─────────────────────────────────────────────────────


def test_classify_to_type_str_intervention():
    from mdp.cli.list_cmd import _classify_to_type_str

    assert _classify_to_type_str("mdp.callbacks.interventions.residual_add.ResidualAdd") == "intervention"


def test_classify_to_type_str_observational():
    from mdp.cli.list_cmd import _classify_to_type_str

    assert _classify_to_type_str("mdp.callbacks.inference.DefaultOutputCallback") == "observational"


def test_classify_to_type_str_training():
    from mdp.cli.list_cmd import _classify_to_type_str

    assert _classify_to_type_str("mdp.training.callbacks.checkpoint.ModelCheckpoint") == "training"


def test_classify_to_type_str_unknown():
    from mdp.cli.list_cmd import _classify_to_type_str

    assert _classify_to_type_str("nonexistent.module.X") == "unknown"


# ── JSON mode output ──────────────────────────────────────────────────────────


def test_list_callbacks_json_includes_type_field():
    """JSON 모드에서 callback_aliases 각 항목에 type 필드가 있어야 한다."""
    with patch("mdp.cli.list_cmd.is_json_mode", return_value=True):
        with patch("mdp.cli.list_cmd.emit_result") as mock_emit:
            from mdp.cli.list_cmd import _list_callbacks
            _list_callbacks()

    assert mock_emit.called
    payload = mock_emit.call_args[0][0]
    aliases = payload.get("callback_aliases", [])
    assert len(aliases) > 0, "callback_aliases가 비어 있으면 안 됨"
    for entry in aliases:
        assert "type" in entry, f"type 필드 없음: {entry}"
        assert entry["type"] in ("intervention", "observational", "training", "unknown")


def test_list_callbacks_json_residual_add_is_intervention():
    """aliases.yaml에 ResidualAdd가 있으면 type=intervention이어야 한다."""
    with patch("mdp.cli.list_cmd.is_json_mode", return_value=True):
        with patch("mdp.cli.list_cmd.emit_result") as mock_emit:
            from mdp.cli.list_cmd import _list_callbacks
            _list_callbacks()

    payload = mock_emit.call_args[0][0]
    aliases = payload.get("callback_aliases", [])
    residual_entries = [e for e in aliases if e["alias"] == "ResidualAdd"]
    assert residual_entries, "ResidualAdd가 aliases에 없음"
    assert residual_entries[0]["type"] == "intervention"


def test_list_callbacks_json_default_output_is_observational():
    """DefaultOutputCallback은 type=observational이어야 한다."""
    with patch("mdp.cli.list_cmd.is_json_mode", return_value=True):
        with patch("mdp.cli.list_cmd.emit_result") as mock_emit:
            from mdp.cli.list_cmd import _list_callbacks
            _list_callbacks()

    payload = mock_emit.call_args[0][0]
    aliases = payload.get("callback_aliases", [])
    entries = [e for e in aliases if e["alias"] == "DefaultOutputCallback"]
    assert entries, "DefaultOutputCallback이 aliases에 없음"
    assert entries[0]["type"] == "observational"


def test_list_callbacks_json_model_checkpoint_is_training():
    """ModelCheckpoint는 type=training이어야 한다."""
    with patch("mdp.cli.list_cmd.is_json_mode", return_value=True):
        with patch("mdp.cli.list_cmd.emit_result") as mock_emit:
            from mdp.cli.list_cmd import _list_callbacks
            _list_callbacks()

    payload = mock_emit.call_args[0][0]
    aliases = payload.get("callback_aliases", [])
    entries = [e for e in aliases if e["alias"] == "ModelCheckpoint"]
    assert entries, "ModelCheckpoint가 aliases에 없음"
    assert entries[0]["type"] == "training"


# ── Rich table (non-JSON) ─────────────────────────────────────────────────────


def test_list_callbacks_table_has_type_column():
    """Rich 테이블 모드에서 Type 컬럼이 렌더링 결과에 있어야 한다."""
    captured_tables = []

    class _FakeConsole:
        def print(self, table):
            captured_tables.append(table)

    with patch("mdp.cli.output.is_json_mode", return_value=False):
        with patch("mdp.cli.list_cmd.is_json_mode", return_value=False):
            with patch("rich.console.Console", return_value=_FakeConsole()):
                from mdp.cli.list_cmd import _list_callbacks
                _list_callbacks()

    assert captured_tables, "console.print가 호출되지 않음"
    table = captured_tables[0]
    column_headers = [col.header for col in table.columns]
    assert "Type" in column_headers, f"Type 컬럼 없음: {column_headers}"
