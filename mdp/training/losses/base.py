"""BaseAlgorithm — RL alignment loss 알고리즘의 공통 계약.

각 알고리즘(DPO/GRPO/PPO 등)은 Trainer가 forward 경로를 어떻게 구성해야 하는지를
class-level flag로 선언한다. 새 flag를 추가할 때도 같은 패턴을 따른다.

계약 flag 3종
---------------

알고리즘이 Trainer에 요구하는 입력 종류를 선언하는 직교 flag 세 개를 BaseAlgorithm
이 공통으로 선언한다. 모든 flag는 ``ClassVar[bool]`` 로 선언되며, 알고리즘은 필요한
flag만 override한다.

- ``needs_logits: ClassVar[bool] = True``
    알고리즘의 ``compute_loss`` 가 ``trainable_out[<model>]["logits"]`` 키를
    읽는가. 기본값은 **True** — 과거 암묵적 상수 True를 명시화한 것이며,
    DPO/GRPO/PPO 등 logits을 항상 소비하는 알고리즘은 선언 없이 이 기본값을
    상속해 byte-identical 동작을 유지한다.

    ``False`` 로 선언한 알고리즘은 Trainer가 ``_forward_model`` 호출을 스킵해도
    되는 대상이다 (예: hidden + output head로 fused loss를 계산하는 경로).

- ``needs_hidden_states: ClassVar[bool] = False``
    알고리즘의 ``compute_loss`` 가 ``trainable_out[<model>]["hidden_states"]`` /
    ``["output_head_weight"]`` 를 읽는가. 기본값은 **False** — 대부분의 logits 기반
    알고리즘은 hidden state를 직접 소비하지 않는다. ``True`` 로 선언한 알고리즘은
    Trainer가 마지막 layer의 hidden state와 output head weight를 주입한다
    (예: Liger FLCE 기반 fused-loss).

- ``needs_generation: ClassVar[bool] = False``
    알고리즘이 학습 루프 시작 전에 policy rollout(샘플 생성)을 필요로 하는가.
    기본값은 **False** — offline 계열(DPO 등)은 batch에 이미 chosen/rejected가
    포함되어 generation이 불필요하다. ``True`` 로 선언한 알고리즘은 Trainer의
    generation 경로(``_train_step_generation``)를 거쳐 rollout 후 학습한다
    (예: GRPO, PPO).

Trainer 측 조회 패턴 (U2 소비자)
---------------------------------

Trainer는 BaseAlgorithm 상속 여부와 무관하게 duck typing으로 flag를 조회한다::

    needs_hidden = getattr(self.algorithm, "needs_hidden_states", False)
    needs_logits = getattr(self.algorithm, "needs_logits", True)
    needs_gen    = getattr(self.algorithm, "needs_generation", False)

BaseAlgorithm의 ClassVar 선언은 **단일 진실 원천**과 **타입 체커·IDE 지원**을
제공할 뿐이며, ``getattr`` default가 런타임 fallback을 보장한다. 따라서 외부
알고리즘이 BaseAlgorithm을 상속하지 않고 plain class로 flag 일부만 명시하는
패턴도 계속 유효하다(예: ``weighted_ntp.WeightedNTPLoss``).

의미 있는 flag 조합 예시
-------------------------

| needs_logits | needs_hidden_states | needs_generation | 예시          | Trainer 동작                              |
|--------------|---------------------|------------------|---------------|-------------------------------------------|
| True         | False               | False            | DPO           | 기존 forward 1회 (logits)                 |
| True         | False               | True             | GRPO / PPO    | generation → forward 1회                  |
| False        | True                | False            | WeightedNTPLoss (외부) | forward 스킵 + hidden 주입       |

네 번째 조합 ``(needs_logits=True, needs_hidden_states=True)`` 는 계약상 허용되지만,
현재 RLTrainer 구현에서는 ``_forward_model`` + ``_extract_hidden_states_and_head`` 가
두 번의 backbone forward를 유발한다. 실사용 consumer는 아직 없으며, forward 통합은
backlog spec ``spec-training-restructure`` 에서 다룬다.

대칭성
-------

이 flag 선언 패턴은 ``BaseModel._block_classes: ClassVar[set[str] | None]`` /
``BaseStrategy`` / ``BaseHead`` 가 취하는 "base class가 ClassVar default를 선언하고
서브클래스는 필요한 flag만 override" 패턴과 대칭이다.
"""

from __future__ import annotations

from typing import ClassVar


class BaseAlgorithm:
    """RL alignment loss 알고리즘의 공통 기반 클래스.

    하위 알고리즘이 Trainer의 forward 분기 동작을 조정하기 위한 class-level
    선언 공간을 제공한다. 기본 구현은 비어 있으며, 알고리즘은 필요한 flag만
    override한다. 각 flag의 의미와 조합은 모듈 docstring을 참조.
    """

    needs_logits: ClassVar[bool] = True
    needs_hidden_states: ClassVar[bool] = False
    needs_generation: ClassVar[bool] = False
