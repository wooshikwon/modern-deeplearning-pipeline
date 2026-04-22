"""Algorithm 계약의 ``needs_logits`` × ``needs_hidden_states`` 2×2 매트릭스 통합 검증.

Phase B U4 — 통합 회귀. `tests/unit/test_rl_trainer_needs_logits_dispatch.py`가
`SimpleNamespace` 스텁으로 분기 로직을 격리 검증했다면, 이 integration test는
**실제 ``BaseAlgorithm`` 서브클래스 + 실제 ``RLTrainer`` 메서드 unbound 바인딩**
조합의 end-to-end 거동을 4개 조합 각각에 대해 검증한다.

설계 의도
---------

spec §설계 원칙이 정의한 2×2 행동성 매트릭스의 4개 조합을 모두 훑어, trainer
forward 분기가 각각의 경우에 호출 패턴(`_forward_model` / `_extract_hidden_states_and_head`)
과 `trainable_out` 컨텐츠를 일관되게 생성하는지 확인한다. "어떤 조합이 지원되는가"를
테스트 레벨에서 명시적 표로 고정하여 향후 계약 확장 시 회귀 시 바로 이 파일이 먼저
깨지도록 한다.

| needs_logits | needs_hidden_states | 실제 사용 사례 | 테스트 클래스 |
|---|---|---|---|
| True  | False | DPO/PPO/GRPO 기본값                   | ``TestLogitsOnlyPath`` |
| True  | True  | hybrid (Phase A 시점 WeightedNTP)     | ``TestLogitsAndHiddenPath`` |
| False | True  | Phase B 목표 (WeightedNTP 현재 상태)  | ``TestHiddenOnlyPath`` |
| False | False | 이론적 무의미 조합 — 방어 계약 검증    | ``TestEmptyContractPath`` |

왜 unit test로 충분하지 않은가
-----------------------------

U2의 unit test는 ``_forward_model`` / ``_extract_hidden_states_and_head``를 람다
스텁으로 치환하고 호출 카운터만 본다. "실제 ``BaseAlgorithm`` 상속 클래스가 실제
트레이너 메서드로부터 ``trainable_out``을 받을 때 어떤 키 구조가 보이는가"는 spy
패턴의 ``compute_loss`` 내부 assert로만 검증 가능한데, 이 맥락이 통합 레벨에서
4개 조합 **모두**에 대해 한꺼번에 살아 있어야 "매트릭스가 계약으로 작동한다"는
점을 바로 읽을 수 있다. 이 파일은 그 **읽기 가이드**이자 회귀 표지.

CPU에서만 구동 가능한 tiny 조합 — 실 모델 없이 `SimpleNamespace` 바인딩으로
`RLTrainer._train_step_offline`만 unbound 호출한다. PEFT/HF 모델과 dispatcher의
실제 연결은 `test_dispatcher_with_real_peft.py`가 담당.
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
#  공용 helper — _make_trainer_stub은 unit test와 동일한 접근을   #
#  따른다 (``_stub_forward_model`` 호출 카운트 수집). integration #
#  레벨의 차이는 "실제 BaseAlgorithm 서브클래스가 호출부"라는 점.#
# ────────────────────────────────────────────────────────────── #


class _CallSpy:
    """``_forward_model`` / ``_extract_hidden_states_and_head`` 호출 기록용 스파이."""

    def __init__(self) -> None:
        self.forward_model_calls: list[dict[str, Any]] = []
        self.extract_hidden_calls: list[dict[str, Any]] = []
        # 각 테스트 클래스가 compute_loss에서 관찰한 trainable_out 구조를 기록한다
        self.observed_policy_out_keys: list[tuple[str, ...]] = []


def _make_stub(
    algorithm: Any,
    spy: _CallSpy,
    *,
    trainable: dict[str, Any] | None = None,
    frozen: dict[str, Any] | None = None,
    hidden_shape: tuple[int, int, int] = (2, 4, 8),
    head_shape: tuple[int, int] = (16, 8),
    logits_shape: tuple[int, int, int] = (2, 4, 16),
) -> SimpleNamespace:
    """RLTrainer-호환 stub 생성. ``_train_step_offline``만 unbound 바인딩할 수 있도록
    필요한 속성만 심는다.

    algorithm은 반드시 ``BaseAlgorithm`` 서브클래스여야 한다 — integration test의
    핵심 계약.
    """
    trainable = trainable if trainable is not None else {"policy": object()}
    frozen = frozen if frozen is not None else {}

    def _fwd(model, batch, role="policy"):
        spy.forward_model_calls.append({"role": role, "model_id": id(model)})
        return {"logits": torch.zeros(*logits_shape)}

    def _fwd_pref(models, batch):
        # non-preference 경로만 다룬다 — preference는 unit test에서 커버.
        raise AssertionError("preference 경로는 이 파일 범위 밖")

    def _extract(model, batch, layer_idx=-1):
        spy.extract_hidden_calls.append({"model_id": id(model)})
        return torch.zeros(*hidden_shape), torch.zeros(*head_shape)

    return SimpleNamespace(
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
        _forward_model=_fwd,
        _forward_preference=_fwd_pref,
        _extract_hidden_states_and_head=_extract,
    )


@contextmanager
def _stub_backward(monkeypatch: pytest.MonkeyPatch):
    """``backward_and_step``를 성공 반환으로 치환. optimizer 실제 호출 회피."""

    def _ok(**kwargs):
        return True

    monkeypatch.setattr(rl_trainer_module, "backward_and_step", _ok)
    yield


def _call_offline(stub: SimpleNamespace, batch: dict) -> tuple:
    return RLTrainer._train_step_offline(stub, batch, device_type="cpu", batch_idx=0)


def _pointwise_batch() -> dict[str, torch.Tensor]:
    """non-preference 배치 — `chosen_input_ids` 없음."""
    return {"input_ids": torch.arange(8).view(2, 4)}


# ────────────────────────────────────────────────────────────── #
#  조합별 Algorithm 구현 — 각 클래스는 BaseAlgorithm을 상속하고   #
#  compute_loss에서 spy에 관찰한 policy_out.keys()를 기록한다.    #
#  이로써 trainer → algorithm 인터페이스가 계약을 지키는지 확인.  #
# ────────────────────────────────────────────────────────────── #


class _LogitsOnlyAlgo(BaseAlgorithm):
    """(needs_logits=True, needs_hidden_states=False) — DPO/PPO/GRPO 기본값.

    BaseAlgorithm의 기본 ``needs_logits=True``를 상속만 받고 별도 override 없음.
    ``needs_hidden_states``는 선언 안 함 → default False (trainer의 getattr).
    """

    def __init__(self, spy: _CallSpy) -> None:
        self._spy = spy

    def compute_loss(self, trainable_out, frozen_out, batch):
        policy_out = trainable_out["policy"]
        self._spy.observed_policy_out_keys.append(tuple(sorted(policy_out.keys())))
        logits = policy_out["logits"]
        loss = logits.sum() * 0.0 + torch.tensor(1.0, requires_grad=True)
        return {"policy": loss}


class _LogitsAndHiddenAlgo(BaseAlgorithm):
    """(needs_logits=True, needs_hidden_states=True) — hybrid 상태.

    Phase A 완료 시점의 WeightedNTP가 일시적으로 존재했던 조합. logits과 hidden
    을 **둘 다** 소비하는 algorithm. 본 spec이 해결하고자 한 redundancy가 바로
    이 조합에서 발생한다 (backbone forward 2회 — `_forward_model` + dispatcher).
    테스트는 계약이 양쪽 모두 활성화됨을 확인한다.
    """

    needs_hidden_states: ClassVar[bool] = True

    def __init__(self, spy: _CallSpy) -> None:
        self._spy = spy

    def compute_loss(self, trainable_out, frozen_out, batch):
        policy_out = trainable_out["policy"]
        self._spy.observed_policy_out_keys.append(tuple(sorted(policy_out.keys())))
        # 계약상 logits + hidden_states + output_head_weight 세 개가 모두 존재해야
        # 함. 이 조합은 redundancy를 유발하지만 semantically 유효.
        loss = (
            policy_out["logits"].sum() * 0.0
            + policy_out["hidden_states"].sum() * 0.0
            + torch.tensor(1.0, requires_grad=True)
        )
        return {"policy": loss}


class _HiddenOnlyAlgo(BaseAlgorithm):
    """(needs_logits=False, needs_hidden_states=True) — Phase B 목표 조합.

    spec의 본래 동기 — WeightedNTP 현재 상태. logits을 소비하지 않으므로 trainer는
    ``_forward_model``을 스킵하고 dispatcher만 호출.
    """

    needs_logits: ClassVar[bool] = False
    needs_hidden_states: ClassVar[bool] = True

    def __init__(self, spy: _CallSpy) -> None:
        self._spy = spy

    def compute_loss(self, trainable_out, frozen_out, batch):
        policy_out = trainable_out["policy"]
        self._spy.observed_policy_out_keys.append(tuple(sorted(policy_out.keys())))
        # 계약: logits 키 없음, hidden_states + output_head_weight만.
        assert "logits" not in policy_out
        loss = (
            policy_out["hidden_states"].sum() * 0.0
            + torch.tensor(1.0, requires_grad=True)
        )
        return {"policy": loss}


class _EmptyContractAlgo(BaseAlgorithm):
    """(needs_logits=False, needs_hidden_states=False) — 이론적 무의미 조합.

    어떤 실제 algorithm도 선언하지 않지만, 계약이 "logits도 hidden도 안 쓴다"는
    선언을 수용할 때 trainer가 어떻게 동작하는지 고정하는 방어적 테스트.

    - trainer는 `_forward_model` / `_extract_hidden_states_and_head`를 모두 스킵.
    - ``trainable_out["policy"]``는 비어 있는 dict ``{}``.
    - algorithm이 이 비어 있는 dict에서 키를 꺼내려 하면 KeyError — 즉 계약이
      무의미 조합을 조용히 허용하지 않고 algorithm 레벨에서 자연 실패한다.
    """

    needs_logits: ClassVar[bool] = False

    def __init__(self, spy: _CallSpy) -> None:
        self._spy = spy

    def compute_loss(self, trainable_out, frozen_out, batch):
        policy_out = trainable_out["policy"]
        self._spy.observed_policy_out_keys.append(tuple(sorted(policy_out.keys())))
        # 의도적으로 아무 키도 없는 dict에서 꺼내기를 시도 — 자연스러운 실패 경로.
        return {"policy": policy_out["logits"]}  # KeyError 발생


# ────────────────────────────────────────────────────────────── #
#  4 조합 통합 테스트                                             #
# ────────────────────────────────────────────────────────────── #


class TestLogitsOnlyPath:
    """(True, False) — DPO/PPO/GRPO 기본값. backbone forward 1회, dispatcher 0회."""

    def test_forward_model_called_once_dispatcher_not_called(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _CallSpy()
        algo = _LogitsOnlyAlgo(spy)
        stub = _make_stub(algo, spy)

        with _stub_backward(monkeypatch):
            _, step_logits = _call_offline(stub, _pointwise_batch())

        # forward 1회 (trainable=policy), dispatcher 0회.
        assert [c["role"] for c in spy.forward_model_calls] == ["policy"]
        assert spy.extract_hidden_calls == []
        # algorithm이 관찰한 policy_out: ("logits",)만.
        assert spy.observed_policy_out_keys == [("logits",)]
        # step_logits는 trainer에서 회수되어 callback에 전파된다.
        assert step_logits is not None
        assert step_logits.shape == (2, 4, 16)


class TestLogitsAndHiddenPath:
    """(True, True) — Phase A 시점 WeightedNTP의 일시 상태.

    trainer가 `_forward_model`과 `_extract_hidden_states_and_head` 양쪽 모두
    호출. redundant하지만 계약 수준에서는 유효한 조합. 본 spec이 해결하고자 한
    redundancy가 여기서 정확히 2회의 backbone 호출로 드러난다.
    """

    def test_both_forward_and_dispatcher_called(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _CallSpy()
        algo = _LogitsAndHiddenAlgo(spy)
        stub = _make_stub(algo, spy)

        with _stub_backward(monkeypatch):
            _, step_logits = _call_offline(stub, _pointwise_batch())

        # forward 1회 + dispatcher 1회 = backbone 경로 2회.
        assert [c["role"] for c in spy.forward_model_calls] == ["policy"]
        assert len(spy.extract_hidden_calls) == 1
        # policy_out에 세 키 모두: logits, hidden_states, output_head_weight.
        assert spy.observed_policy_out_keys == [
            ("hidden_states", "logits", "output_head_weight")
        ]
        # step_logits는 여전히 callback에 전파된다.
        assert step_logits is not None


class TestHiddenOnlyPath:
    """(False, True) — Phase B 목표. WeightedNTP 현재 상태.

    trainer가 `_forward_model`을 **스킵**하고 dispatcher만 호출. ``trainable_out``
    에 logits 키가 없고 hidden_states / output_head_weight만 주입된다.
    """

    def test_forward_model_skipped_dispatcher_called(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _CallSpy()
        algo = _HiddenOnlyAlgo(spy)
        stub = _make_stub(algo, spy)

        with _stub_backward(monkeypatch):
            _, step_logits = _call_offline(stub, _pointwise_batch())

        # forward 0회 (trainable 스킵), dispatcher 1회.
        assert spy.forward_model_calls == []
        assert len(spy.extract_hidden_calls) == 1
        # policy_out에 logits 없음, hidden/head만.
        assert spy.observed_policy_out_keys == [
            ("hidden_states", "output_head_weight")
        ]
        # step_logits는 None — policy_out.get("logits")가 키 없으므로 None.
        assert step_logits is None


class TestEmptyContractPath:
    """(False, False) — 이론적 무의미 조합. 계약이 비대해지지 않는지 확인.

    trainer는 모든 forward 경로를 스킵하고 ``trainable_out["policy"] = {}``만
    남긴다. algorithm이 이 빈 dict에서 logits/hidden을 꺼내려 하면 KeyError가
    자연 발생 — 계약이 "조용한 성공"으로 무의미 조합을 허용하지 않음을 확인.
    """

    def test_all_paths_skipped_policy_out_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _CallSpy()
        algo = _EmptyContractAlgo(spy)
        stub = _make_stub(algo, spy)

        with _stub_backward(monkeypatch):
            # compute_loss에서 `policy_out["logits"]` 접근 → KeyError.
            with pytest.raises(KeyError, match="logits"):
                _call_offline(stub, _pointwise_batch())

        # forward 0회, dispatcher 0회 — 양쪽 모두 스킵.
        assert spy.forward_model_calls == []
        assert spy.extract_hidden_calls == []
        # algorithm은 빈 dict를 관찰했음 — 계약은 무의미 조합에서 조용히 성공하지
        # 않는다.
        assert spy.observed_policy_out_keys == [()]
