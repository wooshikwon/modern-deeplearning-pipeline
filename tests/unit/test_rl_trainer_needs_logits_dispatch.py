"""RLTrainer forward 경로의 ``needs_logits`` 분기 검증 (Phase B U2).

U2 — `_train_step_offline` / `_train_step_generation` 두 메서드가 알고리즘의
``needs_logits: ClassVar[bool]`` 선언을 읽어 ``_forward_model`` 호출을 스킵하는지
확인한다. 기존 DPO/GRPO/PPO 기본 경로 (needs_logits=True)가 byte-identical로
유지되는지, 그리고 WeightedNTP 류의 `needs_logits=False + needs_hidden_states=True`
fused-loss 알고리즘에서 forward가 스킵되고 hidden/head 주입만 일어나는지 검증한다.

전략: RLTrainer 전체 초기화(Settings/Recipe/dataloader)가 무거우므로,
``SimpleNamespace``에 필요한 속성만 심고 ``_train_step_offline`` /
``_train_step_generation`` 메서드를 unbound로 바인딩하여 분기 로직만 격리 검증한다.
(기존 ``test_extract_hidden_states_dispatcher.py``의 ``_call_dispatcher`` 패턴과 동일.)
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import torch

from mdp.training import rl_trainer as rl_trainer_module
from mdp.training.losses.base import BaseAlgorithm
from mdp.training.rl_trainer import RLTrainer


# ────────────────────────────────────────────────────────────── #
#  공용 fixture · 헬퍼                                            #
# ────────────────────────────────────────────────────────────── #


class _CallCounter:
    """``_forward_model`` / ``_forward_preference`` / ``_extract_hidden_states_and_head``
    호출 이력을 기록하는 간이 스파이."""

    def __init__(self) -> None:
        self.forward_model_calls: list[dict[str, Any]] = []
        self.forward_preference_calls: list[dict[str, Any]] = []
        self.extract_hidden_calls: list[dict[str, Any]] = []


def _make_trainer_stub(
    algorithm: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    trainable: dict[str, Any] | None = None,
    frozen: dict[str, Any] | None = None,
    forward_model_out: dict[str, Any] | None = None,
    hidden_shape: tuple[int, int, int] = (2, 4, 8),
    head_shape: tuple[int, int] = (16, 8),
) -> tuple[SimpleNamespace, _CallCounter]:
    """RLTrainer stub을 만든다. `_train_step_offline` / `_train_step_generation`
    메서드를 바인딩할 수 있는 최소 속성만 포함.

    shim 제거(fix-c1) 이후 `_features_forward_model` / `extract_hidden_states_and_head`
    는 모듈 레벨 함수로 직접 호출되므로, monkeypatch를 통해 rl_trainer 모듈 네임스페이스에서
    해당 함수를 spy로 교체한다.
    """
    counter = _CallCounter()
    trainable = trainable if trainable is not None else {"policy": object()}
    frozen = frozen if frozen is not None else {}
    forward_model_out = forward_model_out if forward_model_out is not None else {
        "logits": torch.zeros(2, 4, 16)
    }

    def _spy_forward_model(model, batch, role="policy"):
        counter.forward_model_calls.append({"role": role, "model_id": id(model)})
        return dict(forward_model_out)

    def _stub_forward_preference(models, batch):
        counter.forward_preference_calls.append({"model_names": sorted(models.keys())})
        return {
            name: {
                "chosen_logits": torch.zeros(2, 4, 16),
                "rejected_logits": torch.zeros(2, 4, 16),
            }
            for name in models
        }

    def _spy_extract_hidden(model, batch, layer_idx=-1):
        counter.extract_hidden_calls.append({"model_id": id(model)})
        return torch.zeros(*hidden_shape), torch.zeros(*head_shape)

    monkeypatch.setattr(rl_trainer_module, "_features_forward_model", _spy_forward_model)
    monkeypatch.setattr(rl_trainer_module, "extract_hidden_states_and_head", _spy_extract_hidden)

    stub = SimpleNamespace(
        algorithm=algorithm,
        trainable=trainable,
        frozen=frozen,
        amp_dtype=torch.float32,
        amp_enabled=False,
        optimizers={},
        schedulers={},
        scheduler_intervals={},
        scaler=None,
        grad_accum_steps=1,
        grad_clip_norm=None,
        global_step=0,
        policy=trainable.get("policy"),
        _generation_kwargs={},
        _forward_preference=_stub_forward_preference,
    )
    return stub, counter


@contextmanager
def _stub_backward_and_step(monkeypatch: pytest.MonkeyPatch):
    """`backward_and_step`를 성공(True, {}) 반환으로 교체. optimizer 상호작용 회피.

    반환 튜플의 둘째 원소는 grad_norm dict — 스텁은 빈 dict를 돌려 caller의
    unpack을 만족시키고 실제 gradient 측정은 스킵한다.
    """

    def _ok(**kwargs):
        return True, {}

    monkeypatch.setattr(rl_trainer_module, "backward_and_step", _ok)
    yield


# ────────────────────────────────────────────────────────────── #
#  테스트용 Mock algorithm                                        #
# ────────────────────────────────────────────────────────────── #


class _MockDefaultAlgorithm(BaseAlgorithm):
    """기본 경로: needs_logits=True (상속), needs_hidden_states=False.

    DPO/GRPO/PPO가 선언 없이 기본값을 상속하는 패턴과 동일.
    """

    def compute_loss(self, trainable_out, frozen_out, batch):
        logits = trainable_out["policy"]["logits"]
        loss = logits.sum() * 0.0 + torch.tensor(1.0, requires_grad=True)
        return {"policy": loss}


class _MockFusedLossAlgorithm(BaseAlgorithm):
    """WeightedNTP 류: needs_logits=False + needs_hidden_states=True."""

    needs_logits: ClassVar[bool] = False
    needs_hidden_states: ClassVar[bool] = True

    def compute_loss(self, trainable_out, frozen_out, batch):
        # policy_out에 logits 키는 없어야 하고 hidden_states/output_head_weight만 있어야 함
        policy_out = trainable_out["policy"]
        assert "logits" not in policy_out, (
            "needs_logits=False 인데 trainer가 logits을 주입했습니다."
        )
        assert "hidden_states" in policy_out, "hidden_states 주입 누락"
        assert "output_head_weight" in policy_out, "output_head_weight 주입 누락"
        hidden = policy_out["hidden_states"]
        loss = hidden.sum() * 0.0 + torch.tensor(1.0, requires_grad=True)
        return {"policy": loss}


class _MockPreferenceAlgorithm(BaseAlgorithm):
    """Preference 경로 테스트용: needs_logits=False 선언이 있어도 무시되어야 함."""

    needs_logits: ClassVar[bool] = False

    def compute_loss(self, trainable_out, frozen_out, batch):
        # preference 경로에서 chosen/rejected logits이 주입되어야 함
        policy_out = trainable_out["policy"]
        assert "chosen_logits" in policy_out
        assert "rejected_logits" in policy_out
        loss = policy_out["chosen_logits"].sum() * 0.0 + torch.tensor(1.0, requires_grad=True)
        return {"policy": loss}


# ────────────────────────────────────────────────────────────── #
#  _train_step_offline 분기 테스트                                #
# ────────────────────────────────────────────────────────────── #


def _call_offline(stub: SimpleNamespace, batch: dict) -> tuple:
    """`_train_step_offline` unbound 호출."""
    return RLTrainer._train_step_offline(stub, batch, device_type="cpu", batch_idx=0)


class TestOfflineNeedsLogitsTrue:
    """needs_logits=True 기본 경로 — 기존 DPO/GRPO/PPO 동작 보존."""

    def test_non_preference_calls_forward_model_for_trainable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """needs_logits=True + non-preference 배치 → policy `_forward_model` 1회 호출."""
        algo = _MockDefaultAlgorithm()
        stub, counter = _make_trainer_stub(algo, monkeypatch)
        batch = {"input_ids": torch.arange(8).view(2, 4)}

        with _stub_backward_and_step(monkeypatch):
            _call_offline(stub, batch)

        # trainable=policy 1개 → forward_model 1회. frozen 비어있음.
        assert len(counter.forward_model_calls) == 1
        assert counter.forward_model_calls[0]["role"] == "policy"
        # hidden dispatcher 호출 없음 (needs_hidden_states=False)
        assert counter.extract_hidden_calls == []
        # preference 경로 호출 없음
        assert counter.forward_preference_calls == []

    def test_step_logits_returned_from_policy_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """needs_logits=True → policy_out["logits"]가 step_logits로 반환된다."""
        algo = _MockDefaultAlgorithm()
        stub, counter = _make_trainer_stub(algo, monkeypatch)
        batch = {"input_ids": torch.arange(8).view(2, 4)}

        with _stub_backward_and_step(monkeypatch):
            _, step_logits, _ = _call_offline(stub, batch)

        assert step_logits is not None
        assert step_logits.shape == (2, 4, 16)


class TestOfflineNeedsLogitsFalse:
    """needs_logits=False 경로 — WeightedNTP 류 fused-loss."""

    def test_skips_forward_model_for_trainable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """needs_logits=False + needs_hidden=True → policy `_forward_model`는 호출 안 됨."""
        algo = _MockFusedLossAlgorithm()
        stub, counter = _make_trainer_stub(algo, monkeypatch)
        batch = {"input_ids": torch.arange(8).view(2, 4)}

        with _stub_backward_and_step(monkeypatch):
            _call_offline(stub, batch)

        # _forward_model은 policy에 대해 호출되지 않아야 함 (trainable 스킵)
        # frozen도 비어 있으므로 전체 호출 0회
        assert counter.forward_model_calls == []
        # hidden dispatcher는 호출되어야 함 (needs_hidden=True)
        assert len(counter.extract_hidden_calls) == 1
        # preference 경로 호출 없음
        assert counter.forward_preference_calls == []

    def test_still_injects_hidden_and_head(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """needs_logits=False 경로에서도 trainable_out["policy"]에 hidden + head 주입."""
        algo = _MockFusedLossAlgorithm()
        stub, counter = _make_trainer_stub(algo, monkeypatch)
        batch = {"input_ids": torch.arange(8).view(2, 4)}

        # compute_loss 내부의 assert가 hidden/head 주입을 이미 검증한다.
        with _stub_backward_and_step(monkeypatch):
            loss_val, step_logits, _ = _call_offline(stub, batch)

        # step_logits는 None — policy_out에 "logits" 키 없음
        assert step_logits is None
        assert loss_val == pytest.approx(1.0)

    def test_frozen_forward_still_runs_when_nonempty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """needs_logits=False여도 frozen 모델이 존재하면 frozen forward는 실행.

        WeightedNTP critic/shuffled mode에서 frozen value 모델이 필요한 경우를 방어.
        """
        algo = _MockFusedLossAlgorithm()
        frozen_model = object()
        stub, counter = _make_trainer_stub(
            algo,
            monkeypatch,
            frozen={"value": frozen_model},
            forward_model_out={"values": torch.zeros(2, 4)},
        )
        batch = {"input_ids": torch.arange(8).view(2, 4)}

        with _stub_backward_and_step(monkeypatch):
            _call_offline(stub, batch)

        # frozen forward 1회 호출 (role="value"), trainable forward는 스킵
        assert len(counter.forward_model_calls) == 1
        assert counter.forward_model_calls[0]["role"] == "value"


class TestOfflinePreferencePath:
    """Preference 경로는 needs_logits 플래그와 무관하게 chosen/rejected logits forward."""

    def test_preference_path_ignores_needs_logits_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """is_preference=True 경로는 needs_logits=False 선언을 무시한다."""
        algo = _MockPreferenceAlgorithm()
        stub, counter = _make_trainer_stub(algo, monkeypatch)
        batch = {
            "chosen_input_ids": torch.arange(8).view(2, 4),
            "rejected_input_ids": torch.arange(8, 16).view(2, 4),
        }

        with _stub_backward_and_step(monkeypatch):
            _call_offline(stub, batch)

        # preference 경로가 실행됨: _forward_preference가 frozen({})과 trainable({policy})에 각 1회 호출,
        # _forward_model은 0회. trainable 호출에서 "policy"가 등장해야 함.
        assert len(counter.forward_preference_calls) == 2
        model_names_flat = [
            name
            for call in counter.forward_preference_calls
            for name in call["model_names"]
        ]
        assert "policy" in model_names_flat
        assert counter.forward_model_calls == []

    def test_preference_path_with_default_algorithm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """needs_logits=True (기본) + preference 배치 → 기존 DPO 경로와 byte-identical."""

        class _DPOLike(BaseAlgorithm):
            def compute_loss(self, trainable_out, frozen_out, batch):
                policy_out = trainable_out["policy"]
                assert "chosen_logits" in policy_out
                loss = policy_out["chosen_logits"].sum() * 0.0 + torch.tensor(
                    1.0, requires_grad=True
                )
                return {"policy": loss}

        algo = _DPOLike()
        stub, counter = _make_trainer_stub(algo, monkeypatch)
        batch = {
            "chosen_input_ids": torch.arange(8).view(2, 4),
            "rejected_input_ids": torch.arange(8, 16).view(2, 4),
        }

        with _stub_backward_and_step(monkeypatch):
            _, step_logits, _ = _call_offline(stub, batch)

        # preference 경로: step_logits=None (chosen/rejected 분리로 인함).
        # _forward_preference가 frozen({})과 trainable({policy})에 각 1회 호출 = 총 2회.
        assert step_logits is None
        assert len(counter.forward_preference_calls) == 2
        model_names_flat = [
            name
            for call in counter.forward_preference_calls
            for name in call["model_names"]
        ]
        assert "policy" in model_names_flat
        assert counter.forward_model_calls == []


# ────────────────────────────────────────────────────────────── #
#  _train_step_generation 분기 테스트                             #
# ────────────────────────────────────────────────────────────── #


def _call_generation(stub: SimpleNamespace, batch: dict) -> tuple:
    """`_train_step_generation` unbound 호출."""
    return RLTrainer._train_step_generation(stub, batch, device_type="cpu")


class _GenerateStubModel:
    """policy.generate()를 stub하기 위한 간이 모델. config.pad_token_id 제공."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(pad_token_id=0)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        # prompt를 그대로 반환 (rollout 결과)
        return input_ids


def _make_generation_stub(
    algorithm: Any,
    monkeypatch: pytest.MonkeyPatch,
    mini_epochs: int = 1,
) -> tuple[SimpleNamespace, _CallCounter]:
    """generation 경로용 stub. policy가 `generate` 가능해야 함.

    shim 제거(fix-c1) 이후 모듈 레벨 함수를 monkeypatch로 교체한다.
    """
    policy = _GenerateStubModel()
    counter = _CallCounter()

    # mini_epochs 속성 부착 (mock algorithm이 이미 없으면)
    if not hasattr(algorithm, "mini_epochs"):
        algorithm.mini_epochs = mini_epochs

    def _spy_forward_model(model, batch, role="policy"):
        counter.forward_model_calls.append({"role": role, "model_id": id(model)})
        B, S = batch["input_ids"].shape
        return {"logits": torch.zeros(B, S, 16)}

    def _spy_extract_hidden(model, batch, layer_idx=-1):
        counter.extract_hidden_calls.append({"model_id": id(model)})
        B, S = batch["input_ids"].shape
        return torch.zeros(B, S, 8), torch.zeros(16, 8)

    def _stub_compute_rewards(frozen_out, generated_ids, gen_mask):
        B = generated_ids.shape[0]
        return torch.zeros(B)

    monkeypatch.setattr(rl_trainer_module, "_features_forward_model", _spy_forward_model)
    monkeypatch.setattr(rl_trainer_module, "extract_hidden_states_and_head", _spy_extract_hidden)

    stub = SimpleNamespace(
        algorithm=algorithm,
        trainable={"policy": policy},
        frozen={},
        amp_dtype=torch.float32,
        amp_enabled=False,
        optimizers={},
        schedulers={},
        scheduler_intervals={},
        scaler=None,
        grad_accum_steps=1,
        grad_clip_norm=None,
        global_step=0,
        policy=policy,
        _generation_kwargs={},
        _compute_rewards=_stub_compute_rewards,
    )
    return stub, counter


class _GenerationAlgorithmDefault(BaseAlgorithm):
    """needs_logits=True (기본 상속) — PPO/GRPO 류."""

    needs_generation: ClassVar[bool] = True
    mini_epochs: ClassVar[int] = 1

    def compute_loss(self, trainable_out, frozen_out, batch):
        logits = trainable_out["policy"]["logits"]
        loss = logits.sum() * 0.0 + torch.tensor(1.0, requires_grad=True)
        return {"policy": loss}


class _GenerationAlgorithmFused(BaseAlgorithm):
    """needs_logits=False + needs_hidden_states=True (generation 경로 fused)."""

    needs_generation: ClassVar[bool] = True
    needs_logits: ClassVar[bool] = False
    needs_hidden_states: ClassVar[bool] = True
    mini_epochs: ClassVar[int] = 1

    def compute_loss(self, trainable_out, frozen_out, batch):
        policy_out = trainable_out["policy"]
        assert "logits" not in policy_out
        assert "hidden_states" in policy_out
        assert "output_head_weight" in policy_out
        hidden = policy_out["hidden_states"]
        loss = hidden.sum() * 0.0 + torch.tensor(1.0, requires_grad=True)
        return {"policy": loss}


class TestGenerationNeedsLogitsTrue:
    """Generation mini-epoch 내부의 기본 경로 보존 (PPO/GRPO 회귀)."""

    def test_mini_epoch_calls_forward_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """needs_logits=True 기본: rollout old_logits 1회 + mini-epoch trainable 1회 = 2회."""
        algo = _GenerationAlgorithmDefault()
        stub, counter = _make_generation_stub(algo, monkeypatch)
        batch = {
            "input_ids": torch.arange(8).view(2, 4),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
        }

        with _stub_backward_and_step(monkeypatch):
            _call_generation(stub, batch)

        # rollout old_logits (role="policy") + mini-epoch trainable (role="policy")
        # frozen은 비어 있으므로 frozen forward 없음
        roles = [c["role"] for c in counter.forward_model_calls]
        assert roles.count("policy") == 2, (
            f"rollout + mini-epoch에서 policy forward 2회 기대. 실제: {roles}"
        )
        assert counter.extract_hidden_calls == []


class TestGenerationNeedsLogitsFalse:
    """Generation mini-epoch 내부에서 trainable forward를 스킵."""

    def test_mini_epoch_skips_trainable_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """needs_logits=False: rollout old_logits 1회만 남고 mini-epoch trainable 스킵."""
        algo = _GenerationAlgorithmFused()
        stub, counter = _make_generation_stub(algo, monkeypatch)
        batch = {
            "input_ids": torch.arange(8).view(2, 4),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
        }

        with _stub_backward_and_step(monkeypatch):
            _call_generation(stub, batch)

        # rollout old_logits는 여전히 1회 (spec §Out of scope), mini-epoch trainable은 0회
        # frozen forward는 없음 (frozen={})
        roles = [c["role"] for c in counter.forward_model_calls]
        assert roles == ["policy"], (
            f"rollout old_logits 1회만 있어야 합니다. 실제: {roles}"
        )
        # hidden dispatcher는 mini-epoch에서 호출
        assert len(counter.extract_hidden_calls) == 1
