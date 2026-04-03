# 구조적 마찰 해소 계획

doc-code audit(2026-04-03)에서 식별된 5건의 구조적 마찰. 코드가 기능적으로는 올바르지만, 설계 문서의 자연스러운 설명을 방해하는 지점들이다. 향후 리팩토링 시 참조.

---

## 마찰 1. RL 필드가 Recipe 최상위에 평탄화

### 문제

`algorithm`, `models`는 RL 전용이지만 `optimizer`, `loss`와 같은 레벨에 섞여 있다. 문서에서 Recipe 구조를 설명할 때 "이 필드는 RL에서만 유효하다"는 조건문이 반복된다. `generation`은 RL(GRPO/PPO 텍스트 생성)과 서빙(추론 시 생성 파라미터) 양쪽에서 쓰여 소속이 모호하다.

### 영향 범위 분석

`recipe.algorithm` 접근: 4개 파일 (schema.py, rl_train.py, rl_trainer.py, factory.py)
`recipe.models` 접근: 5개 파일 (위 + test_rl_integration.py)
`recipe.generation` 접근: 2개 파일 (rl_trainer.py, serving/handlers.py)

### 수정 방향

Recipe에 `RLSpec` 서브모델을 추가하고 RL 전용 필드를 이동한다.

```python
# schema.py
class RLSpec(BaseModel):
    algorithm: dict[str, Any]         # _component_ 패턴 (DPO, GRPO, PPO)
    models: dict[str, RLModelSpec]    # 역할별 모델 정의
    generation: GenerationSpec | None = None  # GRPO/PPO 전용

class Recipe(BaseModel):
    ...
    rl: RLSpec | None = None          # None이면 SFT
    generation: GenerationSpec | None = None  # 서빙/추론 전용 (rl.generation과 독립)
```

`generation`은 이원화한다:
- `recipe.rl.generation` — 학습 중 GRPO/PPO 응답 생성 파라미터
- `recipe.generation` — 추론/서빙 시 생성 파라미터 (기존 호환)

### 변경 파일 (6개)

| 파일 | 변경 |
|------|------|
| `settings/schema.py` | `RLSpec` 추가, `algorithm`/`models` 이동, `check_rl_consistency` 수정 |
| `cli/rl_train.py` | `recipe.algorithm` → `recipe.rl.algorithm` (3곳) |
| `training/rl_trainer.py` | `recipe.algorithm` → `recipe.rl.algorithm`, `recipe.models` → `recipe.rl.models` (3곳) |
| `factory/factory.py` | `recipe.models` → `recipe.rl.models` (2곳) |
| `training/rl_trainer.py` | `recipe.generation` → `recipe.rl.generation` (1곳) |
| `tests/e2e/test_rl_integration.py` | `recipe.models` → `recipe.rl.models` (1곳) |

### YAML 호환성

기존 RL Recipe:
```yaml
algorithm:
  _component_: DPO
models:
  policy: ...
```

변경 후:
```yaml
rl:
  algorithm:
    _component_: DPO
  models:
    policy: ...
```

SFT Recipe는 `rl:` 키가 없으므로 영향 없다. RL fixture YAML은 전부 수정 필요.

### 우선순위: **중**

기능에 영향 없음. 문서 구조에만 영향. RL 기능 확장 시(새 알고리즘 추가 등) 함께 처리하면 효율적.

---

## 마찰 2. STRATEGY_MAP 키 비대칭 (`deepspeed` vs `deepspeed_zero3`)

### 문제

ZeRO Stage 2가 `"deepspeed"`, Stage 3가 `"deepspeed_zero3"`. 문서에서 "두 전략은 ZeRO stage만 다르다"고 설명하려면 이름이 대칭이어야 자연스러운데, `deepspeed`에는 stage 번호가 없다.

### 영향 범위 분석

`"deepspeed"` 문자열 참조: 2개 파일

| 파일 | 라인 | 용도 |
|------|------|------|
| `training/_common.py` | 17 | STRATEGY_MAP 키 정의 |
| `cli/list_cmd.py` | 198 | `_STRATEGIES` 표시 목록 |

CompatValidator는 전략 키를 하드코딩하지 않음 (`startswith("fsdp")`만 확인). 테스트 fixture는 `deepspeed_zero3`만 사용. 테스트 코드는 STRATEGY_MAP을 프로그래밍적으로 참조.

### 수정 방향

```python
# _common.py
STRATEGY_MAP: dict[str, str] = {
    "ddp": "mdp.training.strategies.ddp.DDPStrategy",
    "fsdp": "mdp.training.strategies.fsdp.FSDPStrategy",
    "deepspeed_zero2": "mdp.training.strategies.deepspeed.DeepSpeedStrategy",
    "deepspeed_zero3": "mdp.training.strategies.deepspeed.DeepSpeedStrategy",
}
```

### 변경 파일 (2개)

| 파일 | 변경 |
|------|------|
| `training/_common.py:17` | `"deepspeed"` → `"deepspeed_zero2"` |
| `cli/list_cmd.py:198` | `"deepspeed"` → `"deepspeed_zero2"` |

### 하위 호환

사용자 Config YAML에서 `strategy: deepspeed`를 쓰고 있었다면 깨진다. 과도기적으로 양쪽 키를 모두 등록할 수 있다:

```python
"deepspeed": "...",        # deprecated alias
"deepspeed_zero2": "...",  # canonical
"deepspeed_zero3": "...",
```

### 우선순위: **낮음**

변경 범위가 2줄이지만 사용자 YAML 호환성 고려 필요. 다음 major 버전에서 처리.

---

## 마찰 3. `_backward_and_step`이 RLTrainer에만 존재

### 문제

RLTrainer에 `_backward_and_step(losses, force_step)` 메서드가 있고, SFT Trainer는 동일 로직을 `_train_one_epoch` 안에 인라인으로 갖고 있다. 문서에서 backward/step 흐름을 통합 설명하려면 "SFT는 인라인, RL은 메서드"라는 비대칭을 풀어야 한다. `force_step`은 PPO mini-epoch 전용인데 메서드 이름이 범용적이라 SFT에도 있을 것 같은 인상을 준다.

### SFT vs RL 비교

| 측면 | SFT (`trainer.py:414-439`) | RL (`rl_trainer.py:739-767`) |
|------|--------------------------|------------------------------|
| loss 스케일링 | `loss / grad_accum_steps` 인라인 | `loss / accum` (`force_step`이면 1) |
| 모델 수 | 단일 | `dict[str, Tensor]` 다중 |
| scheduler step | `scheduler_interval == "step"` 조건부 | 항상 step |
| NaN 가드 | 인라인 (L418-420) | 메서드 상단 집중 (L742-748) |
| zero_grad | `set_to_none=True` | `set_to_none` 미지정 (기본 False) |

### 수정 방향

`_common.py`에 공유 함수를 추출한다:

```python
def backward_and_step(
    losses: dict[str, torch.Tensor],
    optimizers: dict[str, Optimizer],
    schedulers: dict[str, Any | None],
    scaler: GradScaler,
    trainable_models: dict[str, nn.Module],
    grad_accum_steps: int,
    global_step: int,
    grad_clip_norm: float | None = None,
    force_step: bool = False,
) -> bool:
    """공유 backward + optimizer step. step 실행 여부를 반환한다."""
```

SFT Trainer는 단일 모델/옵티마이저를 dict로 감싸서 전달:

```python
stepped = backward_and_step(
    losses={"model": loss},
    optimizers={"model": self.optimizer},
    schedulers={"model": self.scheduler if self.scheduler_interval == "step" else None},
    ...
)
```

### 변경 파일 (3개)

| 파일 | 변경 |
|------|------|
| `training/_common.py` | `backward_and_step()` 함수 추가 (~30줄) |
| `training/trainer.py` | `_train_one_epoch` 인라인 로직 → `backward_and_step()` 호출로 교체 |
| `training/rl_trainer.py` | `_backward_and_step` 메서드 → `backward_and_step()` 호출로 교체 |

### 조정 필요 사항

- `zero_grad(set_to_none=True)` 통일 — SFT의 `True`가 메모리 효율이 더 좋으므로 RL도 `True`로 통일
- `scheduler_interval` 분기 — 공유 함수에서 `None` scheduler는 step하지 않는 것으로 처리

### 우선순위: **중**

코드 중복 제거 + 문서 설명 단순화. `_common.py` 추출 패턴이 이미 확립되어 있어 자연스럽게 확장 가능.

---

## 마찰 4. Monitoring이 Trainer 서브루틴이면서 독립 모듈

### 문제

`monitoring/baseline.py`는 독립 모듈이지만, 아키텍처 다이어그램에서 Trainer와 동급 박스로 그리면 과대 표현이고, Trainer 하위로 그리면 inference에서도 직접 호출한다는 사실과 맞지 않는다.

### 실제 호출 관계

```
Trainer.train()
  └→ _maybe_compute_baseline() → compute_baseline()    # 학습 후 baseline 생성

inference.py
  ├→ compute_baseline()     # 현재 데이터의 baseline 계산
  └→ compare_baselines()    # 저장된 baseline과 비교 → drift 감지

RLTrainer → (미호출)  # RL 학습에서 monitoring 미통합
```

Monitoring은 **2개의 독립 호출자**(Trainer, inference CLI)가 있으므로 Trainer 서브루틴이 아니다. 다만 RLTrainer에서 미통합인 것은 gap.

### 수정 방향

**이 마찰은 코드가 아닌 문서/다이어그램의 표현 문제다.** 아키텍처 다이어그램에서 Monitoring을 "공유 유틸리티" 레이어에 배치하면 해결된다:

```
┌─ Tier 3: Orchestrator ─┐     ┌─ CLI ─┐
│ TrainPipeline           │     │ inference.py │
│ InferencePipeline       │     └──────────────┘
└────────┬────────────────┘           │
         │                            │
    ┌────▼────────────────────────────▼────┐
    │          Monitoring (공유)             │
    │  compute_baseline / compare_baselines │
    └──────────────────────────────────────┘
```

추가로 RLTrainer에도 `_maybe_compute_baseline()` 메서드를 추가하면 gap이 해소된다.

### 변경 파일

| 파일 | 변경 |
|------|------|
| `drafts/01-architecture.md` | 다이어그램에서 Monitoring 위치 조정 |
| `training/rl_trainer.py` | `_maybe_compute_baseline()` 추가 (Trainer와 동일 패턴, ~15줄) |

### 우선순위: **낮음**

문서 표현 문제가 주. RLTrainer monitoring 통합은 별도 태스크.

---

## 마찰 5. SFT vs RL `train()` 반환 딕셔너리 비대칭

### 문제

| 키 | SFT | RL | TrainResult 스키마 |
|---|---|---|---|
| `metrics` | O (val 메트릭) | O (loss만) | O |
| `training_duration_seconds` | O | O | O (`duration_seconds`로 매핑) |
| `total_steps` | O | O | O |
| `total_epochs` | O | **X** | O |
| `stopped_reason` | O | **X** | O |
| `monitoring` | O (선택) | **X** | O |
| `algorithm` | **X** | O | **X** |
| `checkpoint_dir` | CLI에서 추가 | **X** | O |

RL CLI(`rl_train.py:67`)는 `TrainResult`를 사용하지 않고 raw dict를 `build_result`에 직접 전달한다. SFT CLI(`train.py:106-115`)는 `TrainResult`를 구성한다.

### 수정 방향

1. `TrainResult`에 `algorithm: str | None = None` 추가
2. RLTrainer `train()` 반환에 누락 키 추가:

```python
# rl_trainer.py train() 반환
{
    "metrics": self.last_metrics,          # val 메트릭 포함 (DPO validation 추가됨)
    "training_duration_seconds": ...,
    "total_steps": self.global_step,
    "total_epochs": epoch_counter,         # 추가
    "stopped_reason": stopped_reason,      # 추가 (while 루프 종료 이유 추적)
    "algorithm": type(self.algorithm).__name__,
}
```

3. `rl_train.py`에서 `TrainResult`를 사용하도록 변경:

```python
# rl_train.py — SFT train.py 패턴과 동일
result = TrainResult(
    checkpoint_dir=settings.config.storage.checkpoint_dir,
    output_dir=settings.config.storage.output_dir,
    metrics=train_result.get("metrics", {}),
    total_epochs=train_result.get("total_epochs"),
    total_steps=train_result.get("total_steps"),
    stopped_reason=train_result.get("stopped_reason"),
    duration_seconds=train_result.get("training_duration_seconds"),
    algorithm=train_result.get("algorithm"),
)
```

### 변경 파일 (3개)

| 파일 | 변경 |
|------|------|
| `cli/schemas.py` | `TrainResult`에 `algorithm: str \| None = None` 추가 |
| `training/rl_trainer.py` | `train()` 반환 dict에 `total_epochs`, `stopped_reason` 추가 |
| `cli/rl_train.py` | raw dict → `TrainResult` 구성으로 변경 |

### 우선순위: **높음**

JSON 출력 구조의 일관성은 에이전트 자동화에 직접 영향. `--format json` 출력이 명령에 따라 다른 키를 반환하면 에이전트 파싱 로직이 분기해야 한다. 가장 빠르게 해소할 수 있는 마찰.

---

## 구현 순서 권장

```
[마찰 5] train() 반환 통일        ← 영향 큼, 변경 적음 (3파일)
[마찰 2] strategy 키 대칭화       ← 변경 2줄, 하위 호환 주의
[마찰 3] backward_and_step 공유   ← 중복 제거, _common.py 패턴 활용
[마찰 1] RL 네임스페이스 분리      ← 가장 큰 변경, RL 확장 시 함께
[마찰 4] Monitoring 표현 조정     ← 문서+다이어그램 위주
```
