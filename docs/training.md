# Training Guide

SFT(Supervised Fine-Tuning), RL alignment, 분산 학습, 콜백에 대한 상세 가이드.

## SFT 학습

### 기본 사용법

```bash
mdp train -r recipe.yaml -c config.yaml
```

GPU가 2개 이상이면 자동으로 `torchrun`을 통해 분산 학습이 실행된다.

### 학습 루프 흐름

1. Recipe + Config 로드 → Settings 생성
2. 모델, 데이터, 옵티마이저, 스케줄러, 콜백 생성
3. 분산 전략 설정 (DDP/FSDP/DeepSpeed)
4. Epoch 루프:
   - AMP(Mixed Precision) 활성화
   - Forward → Loss → Backward → Gradient clipping → Step
   - Gradient accumulation 적용
   - 검증 interval에 따라 validation 실행
   - 콜백 호출 (체크포인트, early stopping 등)
5. MLflow에 메트릭 자동 기록

### Fractional Epochs

`epochs`에 소수점을 사용할 수 있다:

```yaml
training:
  epochs: 0.5    # 데이터의 절반만 학습
```

### Max Steps

`epochs` 대신 `max_steps`로 학습량을 지정할 수 있다:

```yaml
training:
  max_steps: 10000
```

둘 다 지정하면 먼저 도달하는 조건에서 종료된다(early-hit). 두 값이 공존하는 것은 의도된 사용 패턴이다 — `epochs`를 안전판으로, `max_steps`를 실제 중단 기준으로 함께 지정하거나, recipe에 `epochs`만 두고 CLI override로 `max_steps`를 주입해도 된다. `Trainer`/`RLTrainer`는 두 값이 모두 set된 채로 진입하면 학습 시작 시점에 INFO 로그를 1회 남겨 정직성을 확보한다("epochs=..., max_steps=... 모두 지정됨. 먼저 도달한 조건에서 종료됩니다"). 하나만 지정된 경우는 무음.

> 구현 근거: `mdp/training/trainer.py` outer break(L388-390), RLTrainer `while` 조건(`mdp/training/rl_trainer.py`의 `while self.global_step < max_steps and not self._stop_requested`). TrainingSpec Field description과 docstring(`mdp/settings/schema.py`)에도 early-hit 규칙이 명시되어 있다.

### Gradient Accumulation

```yaml
training:
  gradient_accumulation_steps: 4
  # 실질 배치 크기 = batch_size × gpus × accumulation_steps
```

### Learning Rate Warmup

```yaml
scheduler:
  _component_: CosineAnnealingLR
  T_max: 100
  warmup_steps: 500          # 절대 스텝 수
  # 또는
  warmup_ratio: 0.1          # 전체의 10% (상호 배타)
```

#### Warmup Tuning — `warmup_start_factor` / `warmup_end_factor`

위 두 필드(`warmup_steps`·`warmup_ratio`)가 **얼마나 길게** warmup할지를 결정한다면, `warmup_start_factor` / `warmup_end_factor`는 **어떤 곡선**으로 warmup할지를 결정한다. 내부적으로 MDP는 `torch.optim.lr_scheduler.LinearLR(start_factor, end_factor, total_iters=warmup_steps)`을 `SequentialLR`로 base scheduler 앞에 감싼다. 두 factor는 MDP 기본값 유지를 전제로 하는 **opt-in** 필드다.

| 필드 | 타입 | 기본값 | 역할 |
|------|------|--------|------|
| `warmup_start_factor` | float (선택) | `1e-8` | warmup step 0의 `base_lr` 멀티플라이어 |
| `warmup_end_factor` | float (선택) | `1.0` | warmup 마지막 step의 `base_lr` 멀티플라이어 |

- **유효 범위**: `0 < warmup_start_factor <= warmup_end_factor <= 1.0`. 위반 시 Trainer / RLTrainer 초기화 단계에서 `ValueError`.
- **`warmup_start_factor = 0` 금지**: PyTorch `LinearLR`이 `start_factor = 0`에서 `ZeroDivisionError`(issue #86454)를 낸다. "0에서 시작"하는 HF 관례를 원하면 `1e-8` 같은 양의 매우 작은 값을 쓴다 — MDP 기본값 `1e-8`이 바로 이 관례의 PyTorch-호환 구현이다.
- **두 필드 생략 시**: `start_factor=1e-8`, `end_factor=1.0`. bf16 mantissa 아래 수치이므로 HF `get_linear_schedule_with_warmup`과 실효적으로 동일.

사용 예시 (base_lr의 10%에서 시작하도록 조정):

```yaml
scheduler:
  _component_: CosineAnnealingLR
  T_max: 100
  warmup_ratio: 0.03
  warmup_start_factor: 0.1    # base_lr × 0.1에서 시작 (MDP 기본 1e-8 대신)
  warmup_end_factor: 1.0
```

**언제 기본값(`1e-8`)을 바꿀까**:

- **소규모 `warmup_steps` (< 10)**: step당 factor 증가폭이 너무 커져 초반 몇 step의 gradient 변화가 과격해질 수 있다. `start_factor=0.1~0.3`이 gradient 완충 효과를 준다.
- **LoRA adapter zero-init**: LoRA는 `B=0`으로 초기화되어 첫 step에서 adapter 출력이 0이므로 "극단적으로 작은 lr에서 시작"할 필요가 거의 없다. 짧은 constant LR이나 완화된 warmup(`start_factor=0.1` 수준)이 문헌상 잘 동작한다.
- **일반 LLM SFT**: MDP 기본값이 LLaMA-3 SFT 실전 스택(LLaMA-Factory, Axolotl, WaveCoder 등)과 수치적으로 동등하므로 **조정 불필요**.

RLTrainer는 `recipe.rl.models.{policy|critic|value|...}.scheduler` 각각이 독립 dict이므로 모델별로 다른 factor를 지정할 수 있다. Trainer와 RLTrainer는 공용 헬퍼(`mdp/training/_schedulers.py`)를 경유하므로 같은 Recipe 필드로 양쪽에서 동일한 LinearLR이 조립된다.

### Loss 함수 선택

**방법 1: 외부 loss 지정** — `model.forward()` → `loss_fn(logits, labels)`
```yaml
loss:
  _component_: CrossEntropyLoss
```

**방법 2: 모델 내장 loss 사용** — `loss:` 섹션 생략 → `model.training_step(batch)` 호출
```yaml
# loss 섹션 생략
# HuggingFace AutoModelFor* 는 내장 loss가 있어 이 방식이 기본
```

### 체크포인트 & Resume

체크포인트 구조:
```
checkpoint-1000/
  ├── model.safetensors (또는 adapter_model.safetensors)
  ├── optimizer.pt
  ├── scheduler.pt
  └── trainer_state.json
```

Resume 설정:
```yaml
# config.yaml
job:
  resume: auto       # 최신 체크포인트에서 자동 재개
  # resume: disabled  # 항상 처음부터
  # resume: ./checkpoints/checkpoint-1000  # 특정 체크포인트
```

Mid-epoch resume도 지원된다 — `trainer_state.json`의 `step_in_epoch` 기준으로 배치를 건너뛴다.

---

## RL Alignment 학습

### 기본 사용법

```bash
mdp rl-train -r rl-recipe.yaml -c config.yaml
```

### 내장 알고리즘

| 알고리즘 | 필요 모델 | 생성 필요 | 용도 |
|---------|----------|---------|------|
| **DPO** | policy + reference | 아니오 | 선호 최적화 |
| **GRPO** | policy + reference + reward | 예 (K개 응답) | 그룹 상대 최적화 |
| **PPO** | policy + reference + value + reward | 예 | 근접 정책 최적화 |

> **외부 알고리즘 주입**: RLTrainer는 `rl.algorithm._component_`로 임의의 loss 클래스를 받는다. `compute_loss(trainable_out, frozen_out, batch) -> {"policy": Tensor}` 규약과 `needs_generation`/`mini_epochs` 속성만 맞추면 외부 패키지의 loss 클래스를 그대로 주입할 수 있다 (예: `weighted_ntp.algorithms.WeightedNTPLoss`).

### DPO 학습 예시

```yaml
name: dpo-alignment
task: text_generation

data:
  dataset:
    _component_: HuggingFaceDataset
    source: argilla/ultrafeedback
    split: train
    fields: {prompt: prompt, chosen: chosen, rejected: rejected}
  collator:
    _component_: PreferenceCollator
    tokenizer: model-name
    max_length: 2048
  dataloader:
    batch_size: 2

training:
  max_steps: 5000
  precision: bf16
  gradient_accumulation_steps: 4

rl:
  algorithm:
    _component_: DPO
    beta: 0.1

  models:
    policy:
      _component_: AutoModelForCausalLM
      pretrained: hf://meta-llama/Llama-3-8B
      adapter: {_component_: LoRA, r: 16, alpha: 32}
      optimizer: {_component_: AdamW, lr: 5e-6}
      scheduler:
        _component_: CosineAnnealingLR
        T_max: 5000
        warmup_ratio: 0.1
    reference:
      _component_: AutoModelForCausalLM
      pretrained: hf://meta-llama/Llama-3-8B
```

### GRPO 학습 예시

```yaml
rl:
  algorithm:
    _component_: GRPO
    beta: 0.01

  models:
    policy:
      _component_: AutoModelForCausalLM
      pretrained: hf://model
      optimizer: {_component_: AdamW, lr: 1e-5}
    reference:
      _component_: AutoModelForCausalLM
      pretrained: hf://model
    reward:
      _component_: AutoModelForCausalLM
      pretrained: hf://reward-model

  generation:
    max_new_tokens: 512
    temperature: 0.8
    group_size: 4              # 프롬프트당 K개 응답 생성
```

### RL 핵심 규칙

- `optimizer`가 있는 모델만 학습됨 (optimizer 생략 = frozen)
- Frozen 모델: `requires_grad=False`, `eval()` 모드
- 각 모델의 optimizer/scheduler가 독립적으로 관리됨
- MLflow 아티팩트에는 policy 모델만 저장됨 (LoRA 시 어댑터만 ~50MB)

---

## 분산 학습

### 자동 감지

GPU가 2개 이상이면 자동으로 분산 학습이 활성화된다. Config에서 전략을 지정하지 않으면 기본 DDP가 사용된다.

### 전략 비교

| 전략 | GPU 수 | 용도 | 메모리 효율 |
|------|--------|------|------------|
| **DDP** | 2~8 | 모델이 단일 GPU에 들어갈 때 | 낮음 (전체 복사) |
| **FSDP** | 2+ | 대형 모델 | 높음 (파라미터 샤딩) |
| **DeepSpeed ZeRO-2** | 2+ | 대형 모델 | 높음 (gradient + optimizer 샤딩) |
| **DeepSpeed ZeRO-3** | 2+ | 초대형 모델 | 매우 높음 (전체 샤딩 + CPU offload) |

### FSDP 설정

```yaml
compute:
  gpus: auto
  distributed:
    strategy: fsdp
    sharding_strategy: FULL_SHARD    # FULL_SHARD | SHARD_GRAD_OP | HYBRID_SHARD
    mixed_precision: true
    precision: bf16
    cpu_offload: false
```

> FSDP + QLoRA는 비호환이다. QLoRA는 DDP 또는 DeepSpeed를 사용한다.

### DeepSpeed 설정

```yaml
compute:
  gpus: auto
  distributed:
    strategy: deepspeed_zero3
```

### Expert Parallelism (MoE)

Mixture-of-Experts 모델에서 Expert Parallelism을 활성화한다:

```yaml
compute:
  distributed:
    strategy: ddp
    expert_parallel:
      ep_size: 4               # EP 그룹당 GPU 수
```

---

## 콜백

### 학습 콜백

콜백은 학습 루프의 8개 지점에서 호출된다:

```
on_train_start → on_epoch_start → (on_batch_start → on_batch_end)* → on_epoch_end
→ on_validation_start → on_validation_end → on_train_end
```

### ModelCheckpoint

```yaml
- _component_: ModelCheckpoint
  dirpath: ./checkpoints
  monitor: val_loss
  mode: min                   # min (loss) 또는 max (accuracy)
  save_top_k: 3               # 상위 3개만 유지
  every_n_steps: 500          # 스텝 기반 저장 (검증과 무관)
  strict: false               # 기본값. true이면 monitor 미매칭 시 즉시 ValueError
```

저장 구조:
- `latest` → 가장 최근 체크포인트 symlink
- `best` → 모니터링 메트릭 기준 최고 체크포인트 symlink

**`strict` 옵션의 의미**: validation이 반환한 metric dict에 `monitor` 키가 없을 때의 동작을 선택한다. `strict: false`(기본)는 WARNING 로그(사용 가능한 metric key 목록 포함)를 남기고 저장을 skip — 기존 동작 유지. `strict: true`는 첫 validation에서 즉시 `ValueError`로 학습을 중단시킨다. 사용 시점은 recipe의 monitor 이름 오타나 head/model이 기대한 metric을 반환하지 않는 상황을 즉시 감지하고 싶을 때다. 대규모 자동 스윕(auto-research 등)에서 "학습은 성공, 산출물은 0개"인 silent failure를 막는 방어선으로 유용하다.

`strict`는 콜백의 `critical` 속성과 **이름만 유사하고 의미가 다르다**. `critical`은 콜백에서 발생한 예외를 trainer가 재전파할지 결정하는 공용 스위치이고, `strict`는 ModelCheckpoint 내부의 monitor 미매칭 처리 방식을 결정한다. 둘은 독립적으로 조합 가능하다.

**`TrainResult.checkpoints_saved`**: 학습 종료 후 결과 JSON에 실제 저장된 체크포인트 개수가 `checkpoints_saved: int | None`으로 포함된다. 상위 오케스트레이터가 MLflow artifact 조회 없이도 "이 run이 산출물을 남겼는가"를 1차 판정할 수 있다. `0`이면 monitor 미매칭 등으로 인한 silent failure를 즉시 탐지 가능.

### EarlyStopping

```yaml
- _component_: EarlyStopping
  monitor: val_loss
  patience: 5
  mode: min
  min_delta: 0.001             # 최소 개선량
```

`patience` 동안 메트릭이 개선되지 않으면 학습을 조기 종료한다.

### EMACallback

```yaml
- _component_: EMACallback
  decay: 0.999
```

Exponential Moving Average — 매 스텝마다 모델 가중치의 이동 평균을 유지한다.

---

## MLflow Logging Conventions

MDP는 학습 중 모든 수치·속성을 MLflow에 기록할 때 **값의 성격**을 기준으로 저장소를 결정한다. "같은 개념은 같은 저장소로, 같은 키로, 같은 시점에, 같은 API로" — Trainer와 RLTrainer는 이 규약을 공유하기 위해 `mdp/training/_mlflow_logging.py` 공용 헬퍼를 동일하게 호출한다. 콜백이 아닌 **builtin 경로**에 로깅을 녹여 넣어, "로깅을 위해 콜백을 추가해야 하는가"라는 질문이 원천적으로 사라진다.

### 왜 이 분류가 필요한가 — 업계 관례와 실증 사례

현재 MLflow 공식 API 자체가 이 분류를 강제한다. `mlflow.log_param`은 "한 run에서 같은 key를 두 번 쓰면 `MlflowException`"이라는 계약이며([mlflow#7402](https://github.com/mlflow/mlflow/issues/7402)), 내부적으로 값을 `str`로 캐스팅해 수치 dashboard에 플로팅되지 않는다. 반면 `mlflow.log_metric(..., step=...)`은 시그니처 자체에 `step` 인자가 first-class로 포함된다. 두 API는 "param은 1회 스냅샷, metric은 시계열"이라는 계약을 구조적으로 분리한다.

주요 프레임워크 4종은 모두 같은 결론에 도달한다. PyTorch Lightning의 `LearningRateMonitor`는 `log_metrics`로만 param group별 LR을 흘리고 `log_param`/`log_hyperparams` 경로를 쓰지 않는다. HuggingFace Transformers의 `MLflowCallback`은 `TrainingArguments.to_dict()`만 `log_params`로 1회 기록하고 이후 매 log step의 `logs["learning_rate"]`를 `log_metrics`로 흘린다. MosaicML Composer의 `LRMonitor`는 `lr-{OptimizerClass}/group{idx}` 키로, LitGPT는 `learning_rate` 키로 — 형식만 다를 뿐 모두 step 축 metric이다.

MDP도 과거엔 이 분류가 없었다. Trainer의 `_log_mlflow_params()`가 `self.optimizer.param_groups[0]["lr"]` 스냅샷을 `params/learning_rate`로 박고 있었고, RLTrainer 역시 같은 스냅샷을 `params/policy_lr`로 기록했다. Warmup scheduler(`start_factor=1e-8`)를 쓰는 run에서는 run 시작 순간 optimizer의 LR이 `base_lr × 1e-8`이기 때문에, 실제로 MLflow UI의 `params/policy_lr`에 `2e-12` 같은 황당한 값이 박혔다. Recipe는 `lr: 2e-4`로 올바르게 선언했고 scheduler도 정상 동작하고 있었으나, warmup step 0 스냅샷이 스스로를 "recipe 선언값"처럼 가장해 사용자가 학습을 중단할 뻔한 실제 사례(weighted-ntp Phase 3 Baseline, 2026-04-18)가 이 분류의 필요성을 증명했다.

### 분류 규약 — 3범주

| 범주 | 정의 | 저장소 API | 호출 시점 |
|------|------|-----------|----------|
| **static** | run 시작 시 확정되어 종료까지 변하지 않는 값 | `mlflow.log_params` | `log_static_params(recipe, settings)` — run 시작 시 1회 |
| **dynamic** | step 또는 epoch 축을 따라 변하는 값 | `mlflow.log_metric(..., step=...)` | `log_step_metrics` / `log_epoch_metrics` — 매 경계 |
| **summary** | run 종료 시점에 확정되는 집계/상태 값 | `mlflow.set_tag` + `mlflow.log_metrics` + (`log_dict`, `log_artifacts`) | `log_summary(...)` — run 종료 직전 |

각 범주가 기록하는 키 목록:

**static (`params/*`)**
`task`, `model_class` 또는 `policy_class`, `algorithm`(RL), `batch_size`, `epochs`, `max_steps`, `precision`, `gradient_accumulation_steps`, `learning_rate_init`, `adapter_component`, `adapter_r`, `strategy`, `dataset_source`, `pretrained`.

`learning_rate_init`은 Recipe의 선언값 그대로다 — SFT는 `recipe.optimizer.lr`, RL은 `recipe.rl.models.policy.optimizer.lr`. 값의 출처가 **Recipe 객체**이지 optimizer 인스턴스가 아니라는 점이 핵심이다. `_init` 접미사는 "recipe 선언값이며 warmup/scheduler 적용 이전"임을 키 이름에 명시한다.

**dynamic (`metrics/*`, step 축)**
`learning_rate` (또는 multi-group slash 변형), `momentum`, `weight_decay`, `train_loss`, `val_*`.

`collect_optimizer_state`가 `optimizer.param_groups` 전체를 순회하며 각 group의 `lr`·`momentum`·`weight_decay`를 뽑아낸다. Warmup이 활성화된 run에서 step 0의 `metrics/learning_rate`는 `base_lr × warmup_start_factor` 값부터 시작해 선형으로 증가하며, `metrics/learning_rate` 시계열을 보면 scheduler의 전체 궤적이 그대로 드러난다. 이 값이 "실제로 적용된 LR"이라는 진실이다.

**summary (`tags/*`, `metrics/final_*`, `artifacts/*`)**
tag: `stopped_reason`, `checkpoints_saved`, `best_checkpoint`. metric: `training_duration_seconds`, `total_steps`, `final_{k}` (각 `last_metrics` 항목에 `final_` prefix). artifact: `config.json` (sanitized), best checkpoint 디렉토리, policy adapter (RL).

`final_*` metric은 run이 종료된 "마지막 값"을 쿼리 가능한 형태로 박아 두는 용도다. 시계열에서 뽑지 않고도 MLflow UI의 run 비교 테이블에서 `final_val_loss` 컬럼만으로 run 간 최종 성능을 정렬할 수 있다.

### Multi-group Optimizer 네이밍 규칙

`param_groups`가 여러 개인 optimizer(예: weighted-ntp의 `CriticValueModel`이 LoRA lr=4e-5 + value head lr=4e-4로 2-group 구성)에서는 group별로 키가 분기된다.

| 축 조합 | 키 형태 | 구체 예 |
|--------|---------|--------|
| single-group, single-optimizer | `learning_rate` | `metrics/learning_rate` |
| single-group, multi-optimizer (RL policy+critic 각 single-group) | `learning_rate/{opt_name}` | `metrics/learning_rate/policy`, `metrics/learning_rate/critic` |
| multi-group, single-optimizer (unnamed) | `learning_rate/group_{idx}` | `metrics/learning_rate/group_0`, `metrics/learning_rate/group_1` |
| multi-group, single-optimizer (named) | `learning_rate/{group_name}` | `metrics/learning_rate/lora`, `metrics/learning_rate/head` |
| multi-group, multi-optimizer | `learning_rate/{opt_name}/group_{idx_or_name}` | `metrics/learning_rate/policy/lora` |

`param_group`에 `"name"` 키가 있으면 숫자 인덱스 대신 이름이 쓰인다 — LoRA/value head 분리 같은 실사례에서 MLflow UI 가독성이 크게 좋아진다. Slash 구분자는 MLflow UI에서 계층 그룹으로 표시되며, Composer(`lr-Adam/group0`)·Lightning(`Adam/pg1`) 관례와 일치한다.

`momentum`·`weight_decay` 도 동일 규칙을 따른다. SGD는 `param_group["momentum"]`을 갖지만 AdamW는 갖지 않으므로, `collect_optimizer_state`는 키가 실제 존재할 때만 해당 항목을 반환한다 — optimizer 종류를 caller가 몰라도 자연스럽게 분기된다.

### Trainer ↔ RLTrainer 대칭 타임라인

| 시점 | Trainer (`mdp train`) | RLTrainer (`mdp rl-train`) |
|------|----------------------|--------------------------|
| `train()` 진입, MLflow run 시작 | `log_static_params(recipe, settings)` | 동일 |
| 매 grad_accum 경계 (loop 내) | `log_step_metrics(self._optimizer_dict(), self.global_step, extra={"train_loss": actual_loss})` | `log_step_metrics(self.optimizers, self.global_step, extra={"loss": step_loss})` |
| Epoch 경계 (Trainer만 보유) | `log_epoch_metrics(self._optimizer_dict(), epoch, extra={"epoch_train_loss": ..., **val_metrics})` | (해당 없음 — RL은 step 축만) |
| Validation end | (extra에 `val_*` 포함) | 동일 |
| run 종료 직전 (`finally` 블록) | `log_summary(training_duration_seconds=..., total_steps=..., stopped_reason=..., final_metrics=self.last_metrics, checkpoint_stats=aggregate_checkpoint_stats(self.callbacks), ...)` | 동일 (RLTrainer의 `final_metrics=self.last_metrics`는 U3에서 복구된 대칭) |

Trainer는 단일 `self.optimizer` 하나만 갖고 RLTrainer는 policy·critic 복수 optimizer를 `self.optimizers` dict로 갖는다. 공용 헬퍼가 양쪽에서 동일하게 동작하려면 동일 시그니처가 필요한데, Trainer는 `_optimizer_dict()` 메서드가 `{"policy": self.optimizer}`로 포장해 RLTrainer의 dict 시그니처와 맞춘다. 이 작은 래퍼 하나 덕분에 공용 헬퍼 함수는 Trainer·RLTrainer 양쪽에서 같은 코드로 호출된다.

### `_mlflow_logging.py` 모듈의 5개 함수 시그니처

```python
def collect_optimizer_state(
    optimizers: Mapping[str, torch.optim.Optimizer],
) -> dict[str, float]:
    """param_groups 전체 순회, slash 네이밍 규칙으로 lr/momentum/weight_decay dict 반환. 순수 함수."""

def log_static_params(recipe: Recipe, settings: Settings) -> None:
    """Run 시작 시 1회. Recipe 선언값만 log_params로. optimizer 인스턴스 상태는 읽지 않음(원칙 2)."""

def log_step_metrics(
    optimizers: Mapping[str, torch.optim.Optimizer],
    step: int,
    extra: Mapping[str, float] | None = None,
) -> None:
    """Step 경계 호출. collect_optimizer_state 결과 + extra를 한 번의 mlflow.log_metrics로 기록."""

def log_epoch_metrics(
    optimizers: Mapping[str, torch.optim.Optimizer],
    epoch: int,
    extra: Mapping[str, float] | None = None,
) -> None:
    """Epoch 경계 호출. log_step_metrics와 동일 구조, step=epoch으로 MLflow에 전달."""

def log_summary(
    *,  # keyword-only
    training_duration_seconds: float,
    total_steps: int,
    stopped_reason: str,
    final_metrics: Mapping[str, float] | None,
    checkpoint_stats: tuple[int, Path | None, str] | None,
    sanitized_config: Mapping[str, Any] | None,
    artifact_dirs: Iterable[tuple[Path, str]] = (),
) -> None:
    """Run 종료 직전 호출. log_metrics(summary+final_*) + set_tag(stopped_reason/checkpoints_saved/best_checkpoint) + log_dict(config) + log_artifacts(dirs)."""
```

모든 함수는 `mlflow.active_run()`이 `None`이면 no-op이다. DDP rank 가드(`_is_main_process`)는 caller(Trainer/RLTrainer) 측에서 이미 걸려 있으므로 모듈 내부에서 별도 가드하지 않는다. thread-safe할 필요도 없다 — rank-0에서만 호출되는 전제.

`log_summary`는 keyword-only 시그니처로 인자 순서 혼동을 막는다. `checkpoint_stats`는 `_common.aggregate_checkpoint_stats(callbacks)` 반환 튜플(`(count, best_path, monitor_hint)`)을 그대로 넘기면 된다 — [spec-trainer-robustness-fixes](https://github.com/user/repo/)의 공용 헬퍼와 자연스럽게 연결된다.

### "원칙 2: 이중 기록 아닌 역할 분리" 상세

업계 관례를 얕게 모방하면 "HuggingFace처럼 초기 LR을 param + 실제 LR을 metric으로 이중 기록"이라는 결론으로 쉽게 흘러간다. 그러나 MDP는 **역할을 명확히 분리**하는 쪽을 택했다. 이유는 두 가지다.

첫째, "초기 LR"의 의미가 출처에 따라 다르다. HF가 param에 넣는 `learning_rate`는 `TrainingArguments`의 선언값(= Recipe의 `optimizer.lr` 필드)이지 warmup step 0의 optimizer 상태가 아니다. Recipe가 출처이므로 static이 맞다. 반면 과거 MDP의 `_log_mlflow_params()`는 optimizer 인스턴스를 읽어 `param_groups[0]["lr"]`을 스냅샷했다 — 이는 warmup 적용 후 값이며, scheduler 구성에 따라 `0.033 × lr` 또는 `1e-8 × lr` 등 어떤 값이든 될 수 있다. **출처가 다르면 의미가 다르다**. 본 spec은 param에 넣을 때는 Recipe의 선언값을 직접 읽어 기록한다. Optimizer 인스턴스 상태는 더 이상 param으로 나가지 않는다.

둘째, metric이 이미 완전하다. Scheduler가 적용된 실제 LR은 매 step마다 metric으로 기록되므로 step 0 값도 당연히 포함된다. Param에 초기 LR을 중복으로 넣는 이점은 MLflow UI의 params 패널에서 recipe 선언값을 빠르게 확인하는 것뿐이며, 그 이점은 `params/learning_rate_init`이라는 **이름 분리**로 얻는다 — "init" 접미사가 "Recipe 선언값이며 warmup 이전"을 명시해 warmup-induced 혼란을 원천 차단한다. 이름이 다르므로 의미도 다르고, 두 값이 의미적으로 다르므로 이중 기록이 아니라 분리된 기록이다.

HuggingFace는 두 값이 어차피 같은 의미(Recipe 선언값)라서 이중 기록해도 충돌이 없다. MDP는 과거에 두 값이 다른 의미(Recipe 선언값 vs warmup step 0 스냅샷)였기 때문에 혼란이 발생했다. 역할 분리가 MDP의 해법이다.

### 마이그레이션 — 기존 대시보드·스크립트

spec-logging-consistency 적용 이후 **더 이상 생성되지 않는 키**:

| 구 키 | 폐지 이유 | 대체 키 |
|------|----------|---------|
| `params/learning_rate` | warmup step 0 optimizer 스냅샷이 Recipe 선언값으로 오인되는 구조적 결함 | `params/learning_rate_init` + `metrics/learning_rate` |
| `params/policy_lr` | 동일 (RL측 비대칭 네이밍) | 위와 동일 |

기존 run과 비교가 필요한 경우:
- Recipe 선언값을 비교하려면 새 run은 `params/learning_rate_init`, 구 run은 `params/learning_rate`(또는 `params/policy_lr`)가 비교 대상. 구 run의 값은 warmup scheduler가 쓰인 경우 선언값이 아닐 수 있음에 유의.
- 실제 적용된 LR을 비교하려면 양쪽 모두 `metrics/learning_rate` step 0 값을 본다. 구 run도 metric 축에는 (부분적으로) 적용된 LR이 남아 있다.

`stopped_reason`·`checkpoints_saved`·`best_checkpoint` tag와 `training_duration_seconds`·`total_steps` summary metric은 **변경되지 않는다**. 이들은 spec-trainer-robustness-fixes에서 이미 제대로 설계된 경로라 본 spec의 범위 밖이다.

### 로깅 서브시스템 흐름

```
Recipe YAML ── 파싱 ──→ Settings ──┐
                                    │
                                    ▼
Trainer.train() / RLTrainer.train()  (rank-0 guard: _is_main_process)
                                    │
             ┌──────────────────────┴──────────────────────┐
             ▼                                              ▼
run 시작:                                         step/epoch 경계:
  log_static_params(recipe, settings)             log_step_metrics(optimizers, step, extra)
  ↓                                                log_epoch_metrics(optimizers, epoch, extra)
  mlflow.log_params({task, batch_size,            ↓
    learning_rate_init, adapter_*, ...})           collect_optimizer_state(optimizers)
                                                   → {learning_rate[/group_*], momentum, ...}
                                                   merged with extra (loss, val_*)
                                                   ↓
                                                   mlflow.log_metrics(merged, step=...)

             run 종료 직전:
             log_summary(*, training_duration_seconds, total_steps,
                          stopped_reason, final_metrics, checkpoint_stats, ...)
             ↓
             mlflow.log_metrics({duration, total_steps, final_*})
             mlflow.set_tag(stopped_reason, checkpoints_saved, best_checkpoint)
             mlflow.log_dict(sanitized_config, "config.json")
             mlflow.log_artifacts(...)
```

모든 화살표는 rank-0에서만 흐른다. `mlflow.active_run()`이 `None`이면 공용 헬퍼가 전부 no-op으로 빠진다 — caller가 rank 가드를 빼먹거나 run이 아직 시작되지 않은 초기화 경로에서 호출해도 부작용이 없다.

---

## 드리프트 모니터링

학습 종료 시 baseline을 저장하고, 추론 시 데이터 분포 변화를 감지한다.

```yaml
monitoring:
  enabled: true
  baseline:
    max_batches: 100          # baseline 계산에 사용할 배치 수
  drift:
    method: jensen_shannon    # 또는 kullback_leibler
    threshold: 0.1
```

감지 항목: 엔트로피 드리프트, 클래스 분포 드리프트, 임베딩 중심 드리프트.
심각도: none → watch → alert → retrain.

---

## Graceful Shutdown

학습 루프가 외부 시그널(사용자 Ctrl+C, 상위 `timeout` 명령, K8s eviction 등)로 중단될 때, MLflow run이 zombie(RUNNING) 상태로 남지 않도록 `Trainer.train()`·`RLTrainer.train()`이 SIGTERM·SIGINT를 내부에서 처리한다.

### 동작 원칙

- 학습 시작 시점에 `train()`이 SIGTERM·SIGINT handler를 설치한다. handler는 `_stop_requested = True` flag만 세우고, step 경계에서 `_should_stop()`이 이 flag를 감지하여 loop break로 이어진다. 배치 중간에 gradient 훼손 없이 "다음 step 이후 graceful shutdown"으로 전환된다
- finally 블록이 실행되어 `strategy.cleanup()` → `on_train_end` 콜백 → `_log_mlflow_summary`가 순차 실행되므로, MLflow run에 `final_*` metric과 `stopped_reason` tag가 정상 기록된다
- finally는 nested try/finally로 구조화되어 있어, `on_train_end` 콜백(ModelCheckpoint, EMA 등)이 예외를 던져도 `_log_mlflow_summary`는 독립적으로 실행되고, 어떤 예외가 발생해도 handler 복원은 최종적으로 보장된다
- 첫 시그널만 `_stop_signal_name`에 기록된다. SIGTERM 수신 후 사용자가 Ctrl+C를 연타해도 `stopped_reason` tag가 SIGTERM 기준으로 유지된다

### timeout과 조합 시 보장사항

```bash
# 2시간 wall-clock 상한으로 학습 실행
timeout 2h mdp train -r recipe.yaml -c config.yaml --format json
```

- 2시간 후 timeout이 SIGTERM을 보내면 `Trainer.train()`이 현재 step 완료 후 break
- MLflow run은 `FINISHED` 상태로 마감 (zombie RUNNING 아님)
- `stopped_reason` MLflow tag에 `signal_term`이 기록됨
- 결과 JSON의 `stopped_reason`·`checkpoints_saved` 필드로 상위 오케스트레이터가 run을 판정 가능

### stopped_reason 리터럴

MLflow tag와 `TrainResult.stopped_reason` JSON 필드에 기록되는 값들:

| `mdp train` (Trainer) | `mdp rl-train` (RLTrainer) |
|---|---|
| `completed` | `completed` |
| `early_stopped` | `early_stopping` |
| `max_steps_reached` | `max_steps` |
| `signal_term` | `signal_term` |
| `signal_int` | `signal_int` |

> Trainer와 RLTrainer의 early/max 리터럴 표기 불일치는 의도된 호환성 유지다. 외부 오케스트레이터가 이미 기존 리터럴을 가정한 경우 깨지지 않도록 새 리터럴(`signal_term`/`signal_int`)만 대칭으로 추가했다. 향후 spec에서 통일 검토.

### 분산 학습에서의 안전성

torchrun은 SIGTERM을 자식 프로세스 전체에 전파한다. 각 rank의 main thread가 독립적으로 handler를 설치하므로 같은 step 경계에서 동시에 break하고 collective 직전에 loop를 탈출한다 — NCCL deadlock이 발생하지 않는다. 일부 rank만 종료되는 경로는 torchrun 전파 가정이 깨질 때만 발생하며, 이 경우는 `NCCL_TIMEOUT`과 함께 고려한다.

### 주의: inference 경로는 handler 없음

본 문서의 graceful shutdown 설명은 **train 경로 전용**이다. `mdp inference` / `mdp generate` 경로에는 현재 signal handler가 설치되어 있지 않다. 추론 중 timeout이 걸리면 출력 파일이 부분 저장 상태로 남을 수 있으므로, 장시간 배치 추론은 `--batch-size`·데이터셋 분할을 조정하여 wall-clock을 예측 가능하게 관리한다. (후속 spec 후보)

---

## 에러 복구 전략

MDP는 프로세스 내 복구를 시도하지 않는다. 대신:

1. `ModelCheckpoint`로 주기적 저장
2. `job.resume: auto`로 최신 체크포인트에서 재시작
3. `torchrun --max_restarts`로 분산 환경에서 자동 재시작
4. MLflow/모니터링 실패는 경고만 출력하고 학습은 계속
5. SIGTERM/SIGINT는 위 Graceful Shutdown 경로로 처리되어 `stopped_reason=signal_term|signal_int` tag와 함께 정상 마감

---

## 주의사항

1. **loss 생략 시**: HuggingFace AutoModelFor\*는 내장 loss가 있어 생략 가능. 커스텀 모델은 `training_step()` 구현 또는 `loss:` 지정 필수
2. **DDP에서 drop_last**: `false`로 설정하면 GPU별 배치 크기 불일치로 gradient 동기화 실패. 기본 `true` 유지
3. **대형 모델 torch_dtype**: 미지정 시 fp32로 로드되어 OOM. `float16` 또는 `bfloat16` 지정 필수
4. **gradient_accumulation**: `batch_size × gpus × accumulation_steps = 실질 배치 크기`. accumulation 변경 시 학습률 조정 필요
5. **warmup_steps/warmup_ratio**: 상호 배타. 하나만 지정
6. **bf16 precision**: Ampere+ GPU(A100, RTX 3090+) 필요
