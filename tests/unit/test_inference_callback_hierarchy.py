"""Unit tests for U4: Inference callback type hierarchy.

Covers BaseInferenceCallback.is_intervention default and
BaseInterventionCallback contract.
"""

from __future__ import annotations

import pytest

from mdp.callbacks.base import BaseInferenceCallback, BaseInterventionCallback


# ---------------------------------------------------------------------------
# BaseInferenceCallback.is_intervention default
# ---------------------------------------------------------------------------


def test_base_inference_callback_is_intervention_false() -> None:
    """BaseInferenceCallback.is_intervention defaults to False."""
    cb = BaseInferenceCallback()
    assert cb.is_intervention is False


def test_base_inference_callback_is_intervention_class_attr() -> None:
    """is_intervention is a class-level attribute on BaseInferenceCallback."""
    assert BaseInferenceCallback.is_intervention is False


# ---------------------------------------------------------------------------
# BaseInterventionCallback class hierarchy and is_intervention
# ---------------------------------------------------------------------------


def test_base_intervention_callback_is_subclass_of_base_inference_callback() -> None:
    """BaseInterventionCallback is a subclass of BaseInferenceCallback."""
    assert issubclass(BaseInterventionCallback, BaseInferenceCallback)


def test_base_intervention_callback_is_intervention_true() -> None:
    """BaseInterventionCallback.is_intervention is True at class level."""
    assert BaseInterventionCallback.is_intervention is True


def test_base_intervention_callback_is_intervention_instance_true() -> None:
    """BaseInterventionCallback instance has is_intervention == True."""

    class ConcreteIntervention(BaseInterventionCallback):
        @property
        def metadata(self):
            return {"type": "TestIntervention"}

    cb = ConcreteIntervention()
    assert cb.is_intervention is True


def test_base_intervention_callback_is_intervention_propagates_to_subclass() -> None:
    """is_intervention=True propagates automatically to subclasses."""

    class MyIntervention(BaseInterventionCallback):
        @property
        def metadata(self):
            return {"type": "MyIntervention"}

    assert MyIntervention.is_intervention is True
    assert MyIntervention().is_intervention is True


# ---------------------------------------------------------------------------
# BaseInterventionCallback.metadata contract
# ---------------------------------------------------------------------------


def test_base_intervention_callback_metadata_raises_not_implemented() -> None:
    """Calling metadata on a subclass that does not implement it raises NotImplementedError."""

    class IncompleteIntervention(BaseInterventionCallback):
        pass

    cb = IncompleteIntervention()
    with pytest.raises(NotImplementedError, match="metadata must be implemented"):
        _ = cb.metadata


def test_base_intervention_callback_metadata_type_key_required() -> None:
    """Concrete implementation returning metadata with 'type' key works correctly."""

    class TypedIntervention(BaseInterventionCallback):
        @property
        def metadata(self):
            return {"type": "ResidualAdd", "strength": 1.0}

    cb = TypedIntervention()
    md = cb.metadata
    assert "type" in md
    assert isinstance(md["type"], str)


def test_base_intervention_callback_metadata_returns_dict() -> None:
    """metadata property returns a dict."""

    class SimpleIntervention(BaseInterventionCallback):
        @property
        def metadata(self):
            return {"type": "Simple", "target_layers": [20, 21], "strength": 0.5}

    cb = SimpleIntervention()
    md = cb.metadata
    assert isinstance(md, dict)


# ---------------------------------------------------------------------------
# Hook signature preservation
# ---------------------------------------------------------------------------


def test_base_intervention_callback_inherits_setup_hook() -> None:
    """BaseInterventionCallback inherits setup hook from BaseInferenceCallback."""
    assert hasattr(BaseInterventionCallback, "setup")
    assert callable(BaseInterventionCallback.setup)


def test_base_intervention_callback_inherits_on_batch_hook() -> None:
    """BaseInterventionCallback inherits on_batch hook from BaseInferenceCallback."""
    assert hasattr(BaseInterventionCallback, "on_batch")
    assert callable(BaseInterventionCallback.on_batch)


def test_base_intervention_callback_inherits_teardown_hook() -> None:
    """BaseInterventionCallback inherits teardown hook from BaseInferenceCallback."""
    assert hasattr(BaseInterventionCallback, "teardown")
    assert callable(BaseInterventionCallback.teardown)


def test_base_intervention_callback_setup_signature_accepts_model_tokenizer() -> None:
    """setup hook signature accepts model and tokenizer (no TypeError)."""
    import inspect

    sig = inspect.signature(BaseInterventionCallback.setup)
    params = list(sig.parameters.keys())
    assert "model" in params
    assert "tokenizer" in params


def test_base_intervention_callback_on_batch_signature() -> None:
    """on_batch hook signature accepts batch_idx, batch, outputs."""
    import inspect

    sig = inspect.signature(BaseInterventionCallback.on_batch)
    params = list(sig.parameters.keys())
    assert "batch_idx" in params
    assert "batch" in params
    assert "outputs" in params


# ---------------------------------------------------------------------------
# Observation callback (direct BaseInferenceCallback subclass) keeps is_intervention=False
# ---------------------------------------------------------------------------


def test_observation_callback_is_intervention_false() -> None:
    """A direct subclass of BaseInferenceCallback keeps is_intervention=False."""

    class ObservationCallback(BaseInferenceCallback):
        pass

    assert ObservationCallback.is_intervention is False
    assert ObservationCallback().is_intervention is False
