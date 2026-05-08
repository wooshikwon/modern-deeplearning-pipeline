"""Unit tests for U5: ResidualAdd and LogitBias intervention callbacks.

Covers:
  - Hook registration and teardown (hook behaviour)
  - metadata dict structure and required keys
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mdp.callbacks.base import BaseInterventionCallback
from mdp.callbacks.interventions import LogitBias, ResidualAdd


# ---------------------------------------------------------------------------
# Minimal model fixtures
# ---------------------------------------------------------------------------


class _FakeLayer(nn.Module):
    """Identity layer used to test hook registration."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeModel(nn.Module):
    """Minimal transformer-like model with model.layers and lm_head."""

    def __init__(self, hidden: int = 8, vocab: int = 32) -> None:
        super().__init__()
        self.model = nn.ModuleList([_FakeLayer() for _ in range(4)])
        self.model.layers = self.model  # expose as .layers attribute
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.model.layers:
            h = layer(h)
        return self.lm_head(h)


def _make_model(hidden: int = 8, vocab: int = 32) -> _FakeModel:
    return _FakeModel(hidden=hidden, vocab=vocab)


def _save_vector(path: Path, shape: tuple) -> torch.Tensor:
    vec = torch.randn(*shape)
    torch.save(vec, path)
    return vec


# ---------------------------------------------------------------------------
# ResidualAdd — hook behaviour
# ---------------------------------------------------------------------------


class TestResidualAddHook:
    def test_setup_registers_hook_on_target_layers(self) -> None:
        """setup() registers exactly one pre-hook per target layer."""
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "vec.pt"
            _save_vector(vpath, (8,))
            cb = ResidualAdd(target_layers=[0, 2], vector_path=vpath, strength=1.0)
            cb.setup(model)

            # Each targeted layer should have 1 forward pre-hook
            assert len(model.model[0]._forward_pre_hooks) == 1
            assert len(model.model[2]._forward_pre_hooks) == 1
            # Non-targeted layer has no hook
            assert len(model.model[1]._forward_pre_hooks) == 0

            cb.teardown()

    def test_teardown_removes_all_hooks(self) -> None:
        """teardown() removes all registered hook handles."""
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "vec.pt"
            _save_vector(vpath, (8,))
            cb = ResidualAdd(target_layers=[0, 1, 2], vector_path=vpath, strength=1.0)
            cb.setup(model)
            cb.teardown()

            for i in range(3):
                assert len(model.model[i]._forward_pre_hooks) == 0

    def test_teardown_twice_is_safe(self) -> None:
        """Calling teardown() twice should not raise."""
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "vec.pt"
            _save_vector(vpath, (8,))
            cb = ResidualAdd(target_layers=[0], vector_path=vpath)
            cb.setup(model)
            cb.teardown()
            cb.teardown()  # must not raise

    def test_hook_modifies_hidden_state(self) -> None:
        """The registered hook actually adds the vector to the hidden state."""
        model = _make_model(hidden=4)
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "vec.pt"
            vec = torch.ones(4)
            torch.save(vec, vpath)

            cb = ResidualAdd(target_layers=[0], vector_path=vpath, strength=2.0)
            cb.setup(model)

            captured_before = []
            captured_after = []

            def _capture_pre(m, args):
                captured_before.append(args[0].clone())

            def _capture_post(m, inp, out):
                captured_after.append(out.clone())

            model.model[0].register_forward_pre_hook(_capture_pre)
            model.model[0].register_forward_hook(_capture_post)

            x = torch.zeros(1, 3, 4)
            model.model[0](x)

            cb.teardown()

            # After hook: output should be x + 2.0 * ones = 2.0
            assert captured_after[0].allclose(torch.full((1, 3, 4), 2.0))

    def test_per_layer_vector_shape(self) -> None:
        """When vector shape is (num_layers, hidden_dim), each layer gets its own slice."""
        model = _make_model(hidden=4)
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "vec.pt"
            vecs = torch.stack([torch.full((4,), float(i)) for i in range(2)])
            torch.save(vecs, vpath)

            cb = ResidualAdd(target_layers=[0, 1], vector_path=vpath, strength=1.0)
            cb.setup(model)

            outputs = []

            def _cap0(m, inp, out):
                outputs.append(("layer0", out.clone()))

            def _cap1(m, inp, out):
                outputs.append(("layer1", out.clone()))

            model.model[0].register_forward_hook(_cap0)
            model.model[1].register_forward_hook(_cap1)

            x = torch.zeros(1, 1, 4)
            model.model[0](x)
            model.model[1](x)

            cb.teardown()

            l0_out = next(v for k, v in outputs if k == "layer0")
            l1_out = next(v for k, v in outputs if k == "layer1")
            # layer 0 slice = 0.0, layer 1 slice = 1.0
            assert l0_out.allclose(torch.zeros(1, 1, 4))
            assert l1_out.allclose(torch.ones(1, 1, 4))

    def test_per_layer_vector_uses_actual_layer_index(self) -> None:
        """2D vector rows are indexed by layer id, not target_layers rank."""
        model = _make_model(hidden=4)
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "vec.pt"
            vecs = torch.stack([torch.full((4,), float(i)) for i in range(3)])
            torch.save(vecs, vpath)

            cb = ResidualAdd(target_layers=[1, 2], vector_path=vpath, strength=1.0)
            cb.setup(model)

            outputs = []
            model.model[1].register_forward_hook(lambda m, inp, out: outputs.append(("layer1", out.clone())))
            model.model[2].register_forward_hook(lambda m, inp, out: outputs.append(("layer2", out.clone())))

            x = torch.zeros(1, 1, 4)
            model.model[1](x)
            model.model[2](x)

            cb.teardown()

            l1_out = next(v for k, v in outputs if k == "layer1")
            l2_out = next(v for k, v in outputs if k == "layer2")
            assert l1_out.allclose(torch.ones(1, 1, 4))
            assert l2_out.allclose(torch.full((1, 1, 4), 2.0))

    def test_per_layer_vector_requires_rows_for_target_layers(self) -> None:
        """2D vector must include rows up to max target layer."""
        model = _make_model(hidden=4)
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "vec.pt"
            vecs = torch.stack([torch.zeros(4), torch.ones(4)])
            torch.save(vecs, vpath)

            cb = ResidualAdd(target_layers=[2], vector_path=vpath, strength=1.0)
            with pytest.raises(ValueError, match="vector rows"):
                cb.setup(model)


# ---------------------------------------------------------------------------
# ResidualAdd — metadata structure
# ---------------------------------------------------------------------------


class TestResidualAddMetadata:
    def test_metadata_has_type_key(self) -> None:
        """metadata dict contains 'type' key."""
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "vec.pt"
            _save_vector(vpath, (8,))
            cb = ResidualAdd(target_layers=[0], vector_path=vpath, strength=0.5)
            cb.setup(model)
            assert "type" in cb.metadata
            cb.teardown()

    def test_metadata_type_value(self) -> None:
        """metadata['type'] is 'ResidualAdd'."""
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "vec.pt"
            _save_vector(vpath, (8,))
            cb = ResidualAdd(target_layers=[0, 1], vector_path=vpath, strength=1.0)
            cb.setup(model)
            assert cb.metadata["type"] == "ResidualAdd"
            cb.teardown()

    def test_metadata_target_layers(self) -> None:
        """metadata['target_layers'] matches the constructor argument."""
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "vec.pt"
            _save_vector(vpath, (8,))
            cb = ResidualAdd(target_layers=[2, 3], vector_path=vpath)
            cb.setup(model)
            assert cb.metadata["target_layers"] == [2, 3]
            cb.teardown()

    def test_metadata_strength(self) -> None:
        """metadata['strength'] matches the constructor argument."""
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "vec.pt"
            _save_vector(vpath, (8,))
            cb = ResidualAdd(target_layers=[0], vector_path=vpath, strength=2.5)
            cb.setup(model)
            assert cb.metadata["strength"] == pytest.approx(2.5)
            cb.teardown()

    def test_metadata_vector_sha256_is_string(self) -> None:
        """metadata['vector_sha256'] is a non-empty string after setup."""
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "vec.pt"
            _save_vector(vpath, (8,))
            cb = ResidualAdd(target_layers=[0], vector_path=vpath)
            cb.setup(model)
            sha = cb.metadata["vector_sha256"]
            assert isinstance(sha, str) and len(sha) > 0
            cb.teardown()

    def test_metadata_vector_sha256_is_deterministic(self) -> None:
        """Same vector file produces the same sha256 digest."""
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "vec.pt"
            _save_vector(vpath, (8,))
            cb1 = ResidualAdd(target_layers=[0], vector_path=vpath)
            cb1.setup(model)
            sha1 = cb1.metadata["vector_sha256"]
            cb1.teardown()

            cb2 = ResidualAdd(target_layers=[0], vector_path=vpath)
            cb2.setup(model)
            sha2 = cb2.metadata["vector_sha256"]
            cb2.teardown()

            assert sha1 == sha2

    def test_metadata_before_setup_raises(self) -> None:
        """Accessing metadata before setup() raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "vec.pt"
            _save_vector(vpath, (8,))
            cb = ResidualAdd(target_layers=[0], vector_path=vpath, strength=1.0)
            # setup() not called — _vector_sha256 is still ""
            with pytest.raises(RuntimeError, match="setup\\(\\)"):
                _ = cb.metadata

    def test_metadata_is_base_intervention_callback(self) -> None:
        """ResidualAdd inherits from BaseInterventionCallback."""
        assert issubclass(ResidualAdd, BaseInterventionCallback)

    def test_metadata_is_intervention_true(self) -> None:
        """ResidualAdd.is_intervention is True."""
        assert ResidualAdd.is_intervention is True


# ---------------------------------------------------------------------------
# LogitBias — hook behaviour
# ---------------------------------------------------------------------------


class TestLogitBiasHook:
    def test_setup_registers_hook_on_lm_head(self) -> None:
        """setup() registers exactly one forward hook on lm_head."""
        model = _make_model()
        cb = LogitBias(token_biases={0: 1.0, 5: -2.0})
        cb.setup(model)
        assert len(model.lm_head._forward_hooks) == 1
        cb.teardown()

    def test_teardown_removes_hook(self) -> None:
        """teardown() removes the registered hook from lm_head."""
        model = _make_model()
        cb = LogitBias(token_biases={0: 1.0})
        cb.setup(model)
        cb.teardown()
        assert len(model.lm_head._forward_hooks) == 0

    def test_teardown_twice_is_safe(self) -> None:
        """Calling teardown() twice must not raise."""
        model = _make_model()
        cb = LogitBias(token_biases={1: 0.5})
        cb.setup(model)
        cb.teardown()
        cb.teardown()  # must not raise

    def test_hook_adds_bias_to_logits(self) -> None:
        """The hook adds the specified bias values to the correct token positions."""
        vocab = 16
        hidden = 8

        token_biases = {2: 10.0, 7: -5.0}

        # Get baseline without hook
        model_base = _make_model(hidden=hidden, vocab=vocab)
        x = torch.zeros(1, 1, hidden)
        with torch.no_grad():
            baseline = model_base.lm_head(x).clone()

        # Get output with hook active (fresh model, same weights via manual copy)
        model_hook = _make_model(hidden=hidden, vocab=vocab)
        # Share weight so outputs are identical modulo the bias
        model_hook.lm_head.weight = model_base.lm_head.weight

        cb = LogitBias(token_biases=token_biases)
        cb.setup(model_hook)

        with torch.no_grad():
            with_bias = model_hook.lm_head(x).clone()

        cb.teardown()

        diff = with_bias - baseline
        assert diff[..., 2].item() == pytest.approx(10.0, abs=1e-4)
        assert diff[..., 7].item() == pytest.approx(-5.0, abs=1e-4)
        # Other tokens should be unchanged
        for i in range(vocab):
            if i not in token_biases:
                assert diff[..., i].item() == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# LogitBias — metadata structure
# ---------------------------------------------------------------------------


class TestLogitBiasMetadata:
    def test_metadata_has_type_key(self) -> None:
        """metadata dict contains 'type' key."""
        cb = LogitBias(token_biases={0: 1.0})
        assert "type" in cb.metadata

    def test_metadata_type_value(self) -> None:
        """metadata['type'] is 'LogitBias'."""
        cb = LogitBias(token_biases={0: 1.0, 3: -2.0})
        assert cb.metadata["type"] == "LogitBias"

    def test_metadata_num_biased_tokens(self) -> None:
        """metadata['num_biased_tokens'] equals len(token_biases)."""
        biases = {0: 1.0, 5: 2.0, 10: -3.0}
        cb = LogitBias(token_biases=biases)
        assert cb.metadata["num_biased_tokens"] == 3

    def test_metadata_bias_sum(self) -> None:
        """metadata['bias_sum'] equals the sum of bias values."""
        biases = {0: 1.0, 5: 2.0, 10: -3.0}
        cb = LogitBias(token_biases=biases)
        assert cb.metadata["bias_sum"] == pytest.approx(0.0)

    def test_metadata_bias_sum_positive(self) -> None:
        biases = {1: 3.5, 2: 1.5}
        cb = LogitBias(token_biases=biases)
        assert cb.metadata["bias_sum"] == pytest.approx(5.0)

    def test_metadata_is_dict(self) -> None:
        """metadata returns a dict."""
        cb = LogitBias(token_biases={1: 0.5})
        assert isinstance(cb.metadata, dict)

    def test_metadata_is_base_intervention_callback(self) -> None:
        """LogitBias inherits from BaseInterventionCallback."""
        assert issubclass(LogitBias, BaseInterventionCallback)

    def test_metadata_is_intervention_true(self) -> None:
        """LogitBias.is_intervention is True."""
        assert LogitBias.is_intervention is True


# ---------------------------------------------------------------------------
# __init__.py re-exports
# ---------------------------------------------------------------------------


def test_interventions_init_exports_residual_add() -> None:
    from mdp.callbacks.interventions import ResidualAdd as RA

    assert RA is ResidualAdd


def test_interventions_init_exports_logit_bias() -> None:
    from mdp.callbacks.interventions import LogitBias as LB

    assert LB is LogitBias
