"""Unit tests for U6: apply_intervention_tags — MLflow tag 적재 검증."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch


from mdp.callbacks.interventions import apply_intervention_tags
from mdp.callbacks.base import BaseInterventionCallback, BaseInferenceCallback


# ---------------------------------------------------------------------------
# Minimal stub callbacks
# ---------------------------------------------------------------------------


class _FakeIntervention(BaseInterventionCallback):
    """Minimal intervention callback stub with controllable metadata."""

    def __init__(self, meta: dict[str, Any]) -> None:
        self._meta = meta

    @property
    def metadata(self) -> dict[str, Any]:
        return self._meta


class _FakeObserver(BaseInferenceCallback):
    """Non-intervention inference callback (is_intervention=False)."""
    pass


# ---------------------------------------------------------------------------
# No interventions — nothing should happen
# ---------------------------------------------------------------------------


class TestNoInterventions:
    def test_empty_list_does_nothing(self) -> None:
        """apply_intervention_tags([]) must not raise and must not call mlflow."""
        with patch("mlflow.set_tag") as mock_set_tag, \
             patch("mlflow.active_run") as mock_run:
            apply_intervention_tags([])
            mock_set_tag.assert_not_called()
            mock_run.assert_not_called()

    def test_only_observer_callbacks_skipped(self) -> None:
        """Non-intervention callbacks are ignored."""
        observer = _FakeObserver()
        with patch("mlflow.set_tag") as mock_set_tag:
            apply_intervention_tags([observer])
            mock_set_tag.assert_not_called()


# ---------------------------------------------------------------------------
# No active MLflow run — stdout/log only
# ---------------------------------------------------------------------------


class TestNoActiveRun:
    def test_no_active_run_logs_to_stdout_only(self, caplog) -> None:
        """Without an active run, metadata is logged and set_tag is NOT called."""
        cb = _FakeIntervention({"type": "TestCb", "strength": 1.0})

        with patch("mlflow.active_run", return_value=None), \
             patch("mlflow.set_tag") as mock_set_tag:
            import logging
            with caplog.at_level(logging.INFO, logger="mdp.callbacks.interventions"):
                apply_intervention_tags([cb])
            mock_set_tag.assert_not_called()

    def test_no_active_run_message_mentions_no_run(self, caplog) -> None:
        """Log message explicitly states that there is no active MLflow run."""
        cb = _FakeIntervention({"type": "TestCb"})

        with patch("mlflow.active_run", return_value=None), \
             patch("mlflow.set_tag"):
            import logging
            with caplog.at_level(logging.INFO, logger="mdp.callbacks.interventions"):
                apply_intervention_tags([cb])

        assert any("no active MLflow run" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Active MLflow run — tag setting
# ---------------------------------------------------------------------------


class TestWithActiveRun:
    def _mock_active_run(self):
        """Return a MagicMock representing an active MLflow run."""
        return MagicMock()

    def test_set_tag_called_for_each_metadata_key(self) -> None:
        """mlflow.set_tag is called once per metadata key."""
        cb = _FakeIntervention({"type": "ResidualAdd", "strength": 1.5})

        with patch("mlflow.active_run", return_value=self._mock_active_run()), \
             patch("mlflow.set_tag") as mock_set_tag:
            apply_intervention_tags([cb])

        assert mock_set_tag.call_count == 2

    def test_tag_key_format(self) -> None:
        """Tag key format is 'intervention.{i}.{k}'."""
        cb = _FakeIntervention({"type": "LogitBias"})

        with patch("mlflow.active_run", return_value=self._mock_active_run()), \
             patch("mlflow.set_tag") as mock_set_tag:
            apply_intervention_tags([cb])

        mock_set_tag.assert_called_once_with("intervention.0.type", "LogitBias")

    def test_tag_value_is_string_for_str_metadata(self) -> None:
        """str metadata values are passed as-is (str conversion)."""
        cb = _FakeIntervention({"type": "ResidualAdd"})

        with patch("mlflow.active_run", return_value=self._mock_active_run()), \
             patch("mlflow.set_tag") as mock_set_tag:
            apply_intervention_tags([cb])

        _, kwargs = mock_set_tag.call_args
        # positional call
        args = mock_set_tag.call_args[0]
        assert isinstance(args[1], str)
        assert args[1] == "ResidualAdd"

    def test_tag_value_is_string_for_numeric_metadata(self) -> None:
        """Numeric metadata values are converted to str."""
        cb = _FakeIntervention({"strength": 2.5})

        with patch("mlflow.active_run", return_value=self._mock_active_run()), \
             patch("mlflow.set_tag") as mock_set_tag:
            apply_intervention_tags([cb])

        args = mock_set_tag.call_args[0]
        assert args[1] == "2.5"

    def test_tag_value_list_is_json_serialized(self) -> None:
        """list metadata values are JSON-serialized."""
        cb = _FakeIntervention({"target_layers": [10, 20, 30]})

        with patch("mlflow.active_run", return_value=self._mock_active_run()), \
             patch("mlflow.set_tag") as mock_set_tag:
            apply_intervention_tags([cb])

        args = mock_set_tag.call_args[0]
        assert args[1] == json.dumps([10, 20, 30])

    def test_multiple_intervention_callbacks_indexed(self) -> None:
        """Multiple intervention callbacks use incremented index in tag key."""
        cb0 = _FakeIntervention({"type": "ResidualAdd"})
        cb1 = _FakeIntervention({"type": "LogitBias"})

        with patch("mlflow.active_run", return_value=self._mock_active_run()), \
             patch("mlflow.set_tag") as mock_set_tag:
            apply_intervention_tags([cb0, cb1])

        calls = mock_set_tag.call_args_list
        keys = [c[0][0] for c in calls]
        assert "intervention.0.type" in keys
        assert "intervention.1.type" in keys

    def test_observer_between_interventions_not_counted(self) -> None:
        """Non-intervention callbacks between interventions do not shift index.

        Only is_intervention=True callbacks are enumerated.
        """
        cb0 = _FakeIntervention({"type": "A"})
        observer = _FakeObserver()
        cb1 = _FakeIntervention({"type": "B"})

        with patch("mlflow.active_run", return_value=self._mock_active_run()), \
             patch("mlflow.set_tag") as mock_set_tag:
            apply_intervention_tags([cb0, observer, cb1])

        keys = [c[0][0] for c in mock_set_tag.call_args_list]
        # observer is skipped; intervention indices are 0 and 1
        assert "intervention.0.type" in keys
        assert "intervention.1.type" in keys
        assert all("observer" not in k for k in keys)

    def test_tag_value_truncated_at_5000_chars(self) -> None:
        """Tag values exceeding 5000 characters are truncated to <=5000 chars and end with '...'."""
        long_list = list(range(2000))  # JSON > 5000 chars
        cb = _FakeIntervention({"tokens": long_list})

        with patch("mlflow.active_run", return_value=self._mock_active_run()), \
             patch("mlflow.set_tag") as mock_set_tag:
            apply_intervention_tags([cb])

        args = mock_set_tag.call_args[0]
        assert len(args[1]) <= 5000
        assert args[1].endswith("..."), (
            f"Truncated tag value should end with '...' to signal truncation, got: {args[1][-10:]!r}"
        )


# ---------------------------------------------------------------------------
# mlflow ImportError — graceful degradation
# ---------------------------------------------------------------------------


class TestMlflowImportError:
    def test_import_error_does_not_raise(self) -> None:
        """If mlflow is not installed, apply_intervention_tags must not raise."""
        cb = _FakeIntervention({"type": "ResidualAdd", "strength": 1.0})

        import sys
        with patch.dict(sys.modules, {"mlflow": None}):
            # Force re-evaluation of the import inside the function by removing
            # mdp.callbacks.interventions from the cache so we get the real
            # ImportError path; alternatively, patch builtins.__import__.
            import builtins
            real_import = builtins.__import__

            def _mock_import(name, *args, **kwargs):
                if name == "mlflow":
                    raise ImportError("mlflow not installed")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_mock_import):
                apply_intervention_tags([cb])  # must not raise
