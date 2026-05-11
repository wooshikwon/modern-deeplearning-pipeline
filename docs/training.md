# Training Guide

SFT(Supervised Fine-Tuning), RL alignment, 분산 학습, 콜백에 대한 상세 가이드.

## Trainer / RLTrainer 상속 구조

`Trainer`(SFT)와 `RLTrainer`(RL)는 모두 `BaseTrainer(ABC)`를 상속한다. `BaseTrainer`는 두 trainer가 중복 구현하던 lifecycle infrastructure를 단일 위치에 집중시킨다.

```
BaseTrainer(ABC)           — mdp/training/_base.py
├── Trainer(BaseTrainer)   — mdp/training/trainer.py   (mdp train)
└── RLTrainer(BaseTrainer) — mdp/training/rl_trainer.py (mdp rl-train)
```

`BaseTrainer`가 소유하는 것:
- 공통 wrapper: `_fire`, `_move_to_device`, `_should_stop`, `_estimate_total_steps`
- OOM / memory_history wrapper (`_progress_log.py` free function 래핑)
- System logging wrapper: `_fmt_eta`, `_log_step_progress`, `_log_run_banner`
- MLflow lifecycle wrapper: `_start_mlflow_run`, `_log_mlflow_params`, `_log_mlflow_summary`
- Checkpoint state hooks: `_checkpoint_state()` (abstract), `_load_checkpoint_state(state)`

`Trainer` / `RLTrainer`가 보유하는 것:
- `train()` main loop (epoch/step 루프, 콜백 발화, OOM 처리)
- Step 실행 메서드 (`_train_step_*`, `_forward_preference`, 등)
- Validation 루프 (`_validate`, `_run_rl_validation`, 등)
- `_checkpoint_state()` / `_checkpoint_model_slots()` / `_load_checkpoint_state()` 구현
- `_maybe_resume()` 구현
- `_collect_mlflow_params()` 구현

### Runtime control-plane 호출 경로

학습 CLI의 상위 흐름은 `RunPlan -> AssemblyPlan -> ExecutionEngine`으로
고정되어 있다.

```
mdp train / mdp rl-train
  -> SettingsLoader.load_training_settings(...)
  -> RunPlanBuilder.training(...)
  -> RuntimeLauncher
  -> runtime worker
  -> ExecutionEngine.run(run_plan)
  -> AssemblyPlanner.from_run_plan(...)
  -> AssemblyMaterializer(...)
  -> Trainer.from_bundle(...).train()
     또는 RLTrainer.from_bundle(...).train()
```

`SettingsLoader.load_training_settings()`는 recipe/config source loading,
override 적용, env 치환, schema validation을 수행하고 `Settings`를 반환한다.
`SettingsLoader`는 runtime planning wrapper가 아니며 command intent를 만들지
않는다. `RunPlanBuilder.training(...)`이 이 `Settings`에 validation scope,
callback config path, source path, command intent, distributed intent를 더해
`RunPlan`을 만든다. runtime source of truth는 `RunPlan`이다.

`AssemblyPlan`은 모델/데이터/전략/콜백/trainer 조립 그래프다. 실제 모델,
optimizer, dataloader, callback instance는 담지 않는다. 이 경계 때문에
torchrun worker는 logging bootstrap, process group 초기화, Liger patch 적용을
끝낸 뒤 안전하게 materialization을 수행할 수 있다.

`ExecutionEngine`은 SFT/RL dispatch의 소유자다.
`mdp.runtime.training.run_training(run_plan: RunPlan)`은 `Settings`에서
`RunPlan`을 만들지 않는다. Liger patch를 적용한 뒤 이미 검증된 `RunPlan`으로
`ExecutionEngine`에 진입하는 current-process helper이며,
`_torchrun_entry.run_training()`은 torchrun worker adapter로서 이 runtime
helper에 위임한다. 공식 runtime
작성 경로는 `Trainer.from_bundle(...)` / `RLTrainer.from_bundle(...)`이고,
직접 생성자 `Trainer(...)` / `RLTrainer(...)`는 이미 materialized component를
주입하는 stable low-level loop API다.

### Checkpoint I/O 호출 경로

체크포인트 저장·복원의 파일 I/O는 `mdp/training/_checkpoint.py`의 `CheckpointManager`가 담당한다. Trainer/RLTrainer는 저장 시점과 resume 진입점을 소유하고, callback은 manager에 model slot과 trainer state를 넘긴다.

```
# Save
ModelCheckpoint.save_checkpoint(...)
  -> trainer._checkpoint_model_slots()
  -> CheckpointManager.save(...)

# Resume
trainer._maybe_resume()
  -> CheckpointManager.load(...)
  -> trainer._load_checkpoint_state(...)
```

FSDP 환경에서는 `_checkpoint.gather_fsdp_state_dict(model)`이 all-rank collective로 full state dict를 수집한다.
학습 종료 후 MLflow `model/` artifact는 checkpoint manager가 아니라
`ServingArtifactManager(mode="mlflow_snapshot")`가 작성한다. 배포용 package는
`mdp export` 경로의 `deployment_export` mode가 별도로 담당한다.

### Feature Extractor 호출 경로

`_extract_hidden_states_and_head` dispatcher는 `mdp/training/_features.py`
stateless free function으로 분리되어 있다. `RLTrainer`는 이를 직접 호출하며,
얇은 bound method wrapper(`self._extract_hidden_states_and_head`)는 trainer
low-level API를 직접 다루는 호출자를 위해 유지된다.

```python
from mdp.training._features import extract_hidden_states_and_head

# RLTrainer 내부: needs_hidden_states=True 경로
hidden, head_weight = extract_hidden_states_and_head(
    self.trainable["policy"], batch, layer_idx=-1
)
```

`_features.py`는 stateless이므로 향후 inference callback에서도 trainer 인스턴스 없이 재사용 가능하다.

---

## SFT 학습

### 기본 사용법

```bash
mdp train -r recipe.yaml -c config.yaml
```

GPU가 2개 이상이고 `compute.distributed.strategy`가 명시되어 있으면 `torchrun`을 통해 분산 학습이 실행된다.

### 학습 루프 흐름

1. Recipe + Config 로드 -> `RunPlan` 생성
2. RuntimeLauncher가 single-process 또는 torchrun parent launch 결정
3. worker에서 logging, process group, Liger patch 순서로 runtime setup
4. `AssemblyPlan` 생성 후 모델, 데이터, 옵티마이저, 스케줄러, 콜백 bundle materialization
5. `ExecutionEngine`이 `Trainer` 또는 `RLTrainer`로 dispatch
6. Epoch 루프:
   - AMP(Mixed Precision) 활성화
   - Forward → Loss → Backward → Gradient clipping → Step
   - Gradient accumulation 적용
   - 검증 interval에 따라 validation 실행
   - 콜백 호출 (체크포인트, early stopping 등)
7. MLflow에 메트릭 자동 기록

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

**방법 2: 모델 내장 loss 사용** — `loss:` 섹션 생략 → `model.forward()` output의 `loss` 사용
```yaml
# loss 섹션 생략
# HuggingFace AutoModelFor* 는 labels가 있으면 outputs.loss를 반환하므로 이 방식이 기본
```

`loss:`를 지정하면 external supervised criterion이 loss를 소유한다. 이때 모델이
forward output에 `loss`를 함께 반환해도 Trainer는 `loss_fn(logits, labels)`만
사용하고 native loss는 무시한다. `loss:`를 생략하면 Trainer는 forward output의
Tensor `loss`를 요구한다.

raw timm/torchvision-style vision 모델처럼 `forward(x) -> logits`만 제공하는
모델은 `loss:`를 지정하면 그대로 사용할 수 있다. Trainer의 forward adapter가
batch의 `pixel_values`를 positional `x`로 전달하고, external criterion이
`logits`와 `labels`로 loss를 계산한다. `loss:`를 생략하려면 raw 모델을 감싼
custom wrapper가 `forward()` output에 `loss`를 반환해야 한다.

### 체크포인트 & Resume

새 체크포인트 구조:
```
checkpoint-1000/
  ├── manifest.json
  ├── model.safetensors (또는 adapter_model.safetensors / adapter_model.bin)
  ├── optimizer.pt
  ├── scheduler.pt
  ├── scaler.pt                 # AMP 사용 시
  ├── recipe.yaml               # snapshot이 있으면 기록
  └── trainer_state.json
```

`manifest.json`은 checkpoint layout의 source of truth다. `layout_version`, `kind`(`sft`/`rl`), `global_step`, `saved_at`, trainer state 파일, recipe/config snapshot 파일, scaler 파일, 그리고 모델별 role/format/path/trainable/optimizer/scheduler 경로를 기록한다. RL checkpoint는 `policy/`, `reference/`, `reward/`, `critic/`, `value/` 같은 model slot별 하위 디렉토리를 사용할 수 있다.

Checkpoint compatibility boundary: 권장 작성 경로는 항상 `manifest.json`
기반 layout이다. `manifest.json`이 없는 checkpoint는 read-only compatibility
reader가 best-effort로 읽는다. 이 경우 `trainer_state.json`과 `scaler.pt`만
공통 I/O 계층에서 복원하고, 모델/optimizer/scheduler weight 주입은
manifestless resume reader의 파일명 관례(`adapter_model.safetensors`,
`adapter_model.bin`, `model.safetensors`, `model.pt`, `optimizer.pt`,
`scheduler.pt`)를 따른다.

`best`와 `latest`는 checkpoint 디렉토리의 symlink다. `latest`는 가장 최근 저장본을 가리키고, `best`는 validation monitor가 있을 때 가장 좋은 metric checkpoint를 가리킨다. Export와 MLflow summary에서 checkpoint 하나를 고를 때는 `best`가 우선이고 없으면 `latest`를 사용한다.

Strategy checkpoint boundary:
- No strategy: manager가 모델 state dict 또는 `save_pretrained()` 결과를 직접 저장한다.
- DDP: `checkpoint_capability`가 manager 지원을 선언한다. 일반 모델은 rank 0이 full state dict를 저장하지만, PEFT/LoRA 모델은 strategy `unwrap()`으로 내부 모델을 확인한 뒤 `save_pretrained()` adapter layout을 우선한다.
- FSDP: full state dict 저장이 all-rank collective이므로 모든 rank가 save 경로에 진입하고 rank 0만 파일과 manifest를 쓴다.
- DeepSpeed ZeRO: 현재 unsupported다. ZeRO engine checkpoint는 별도 engine-contract spec 범위이며 DDP/FSDP checkpoint처럼 복원 가능하다고 가정하지 않는다.

Resume 설정:
```yaml
# config.yaml
job:
  name: core4-h100-20260511  # 선택. 있으면 checkpoint_dir/{recipe.name}/{job.name} 사용
  resume: auto       # 최신 체크포인트에서 자동 재개
  # resume: disabled  # 항상 처음부터
  # resume: ./checkpoints/checkpoint-1000  # 특정 체크포인트
```

`job.name`을 지정하면 `storage.checkpoint_dir` 아래에 recipe/job namespace가 붙는다.
예를 들어 `checkpoint_dir: ./storage/checkpoints/runs`, recipe `wntp_taw`,
job `core4-h100`이면 실제 checkpoint root는
`./storage/checkpoints/runs/wntp_taw/core4-h100`이다. Resume 시 manifest의
`recipe_name`이 현재 recipe와 다르면 restore 전에 실패하여 다른 recipe의
`latest`를 잘못 잡는 것을 차단한다.

SFT는 mid-epoch resume을 지원한다 — `trainer_state.json`의 `step_in_epoch` 기준으로 배치를 건너뛴다. RLTrainer는 현재 `global_step`과 epoch counter를 복원하지만 `step_in_epoch` 기반 batch skip은 수행하지 않는다.

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

> **외부 알고리즘 주입**: RLTrainer는 `rl.algorithm._component_`로 임의의 loss 클래스를 받는다. `compute_loss(trainable_out, frozen_out, batch) -> {"policy": Tensor}` 규약과 `needs_generation`/`needs_hidden_states`/`needs_logits` / `mini_epochs` 속성만 맞추면 외부 패키지의 loss 클래스를 그대로 주입할 수 있다 (예: `weighted_ntp.algorithms.WeightedNTPLoss`). 계약 flag와 구현 패턴은 `docs/extending.md` "커스텀 RL 알고리즘" 섹션을 참조한다.

### Algorithm Interface — forward 분기 로직

RLTrainer는 매 step에서 알고리즘이 선언한 3개 flag를 `getattr` duck typing으로 조회해 forward 경로를 재구성한다. `mdp/training/losses/base.py`의 `BaseAlgorithm`이 선언하는 ClassVar 3개(`needs_logits=True` / `needs_hidden_states=False` / `needs_generation=False`)가 단일 진실 원천이고, 알고리즘은 필요한 flag만 override한다. 계약 상세와 조합별 예시는 `docs/extending.md`의 "커스텀 RL 알고리즘" 섹션에 있고, 이 항목은 **trainer 내부 분기 순서**를 기술한다.

`_train_step_offline` / `_train_step_generation` 공통 순서 (pointwise / mini-epoch 분기):

1. `needs_logits = getattr(algorithm, "needs_logits", True)`로 flag 조회. default True이므로 선언 누락 = 기존 forward 경로 유지.
2. `needs_logits=True`면 `self.trainable`의 각 모델에 대해 `_forward_model`을 실행하여 `trainable_out[<name>]["logits"]`을 채운다.
3. `needs_logits=False`면 `trainable_out = {name: {} for name in self.trainable}`로 빈 dict만 초기화. logits forward를 스킵하여 `(B, S, V)` tensor와 backbone activation 중복을 제거.
4. `needs_hidden_states=True`면 `_extract_hidden_states_and_head(self.trainable["policy"], batch)` dispatcher를 호출하여 `trainable_out["policy"]`에 `"hidden_states"` / `"output_head_weight"` 키를 setdefault로 주입.
5. `algorithm.compute_loss(trainable_out, frozen_out, batch)`로 loss dict를 얻는다.

**Preference 경로(DPO 등)**는 chosen/rejected logits 둘 다 필수이므로 `needs_logits=False` 선언이 있어도 의도적으로 무시한다. `is_preference` 분기가 먼저 `_forward_preference`로 흘러 위 게이트의 영향권 밖이다.

**Frozen 모델 forward는 언제나 실행된다**. `self.frozen`에 등록된 모델(reference, value, reward 등)은 algorithm의 flag와 독립적으로 `_forward_model` 경로를 탄다. frozen skip 여부까지 계약에 끌어올리는 것은 "현재 `needs_logits=False`를 선언하는 fused-loss 계열은 대체로 frozen도 안 쓴다"는 경험적 관찰이 확립되기 전까지 out-of-scope.

**generation 경로의 `old_logits` forward**는 `needs_logits` 게이트와 무관하게 실행된다. PPO/GRPO가 old_log_probs 계산에 사용하는 rollout 단계 forward이므로 flag로 스킵할 수 없다.

#### Memory-Efficient CE Helper — `compute_per_token_ce_chunked_from_hidden`

`(needs_logits=False, needs_hidden_states=True)` 경로의 알고리즘은 hidden states와 output head weight만으로 per-token CE를 계산해야 한다. MDP는 이 용도로 아래 free function을 제공한다.

**위치**: `mdp/training/losses/_ce_helpers.py`

**시그니처**:

```python
from mdp.training.losses._ce_helpers import compute_per_token_ce_chunked_from_hidden

compute_per_token_ce_chunked_from_hidden(
    hidden_states: Tensor,    # (B, S, H)
    head_weight: Tensor,      # (V, H)
    labels: Tensor,           # (B, S)
    chunk_size: int,
    ignore_index: int = -100,
) -> Tensor                   # (B, S-1)
```

**동작**: hidden states를 sequence 축(`S-1` dim)으로 `chunk_size`만큼 나누어 각 chunk 안에서만 `hidden_chunk @ head_weight.T`를 계산한다. 각 chunk의 forward 계산은 `torch.utils.checkpoint.checkpoint(use_reentrant=False)`로 감싸져 backward 시점까지 `(B, chunk, V)` 중간 텐서가 autograd graph에 남지 않는다. 이를 통해 어떤 시점에도 `(B, S, V)` 전체 logits tensor가 HBM에 materialize되지 않는다.

**대표 소비자**: `WeightedNTPLoss` (`weighted_ntp.algorithms.WeightedNTPLoss`). `needs_logits=False, needs_hidden_states=True` 플래그를 선언하고 `self.loss_chunk_size`를 `chunk_size` 인자로 전달한다.

**FLCE 경로 보류 이유**: Liger FLCE는 `bf16 + reduction="none" + 큰 num_valid` 조합에서 backward gradient가 전부 0으로 수치 붕괴하는 결함이 확정됐다 (sincere-gnat-917). 업스트림 패치가 1년 이상 미해결이므로 MDP helper로 대체한다.

#### 실측 효과 — needs_logits=False 도입 전후

`weighted_ntp` sanity run(H200 4× DDP, bs=32, Llama-3-8B, gradient_checkpointing + Liger FLCE)에서 Phase A(`needs_hidden_states=True`만 도입) → Phase B(`needs_logits=False` 추가)로 옮기면서:

| 지표 | Phase A (c4~c6) | Phase B (c7) | 변화 |
|---|---:|---:|:---:|
| peak_memory_gb | 114.25 | 60.51 | **−47%** |
| duration (10 step sanity) | 212.3s | 137.6s | **−35%** |
| loss (bit-level) | 0.7557 | 0.7557 | **동일** |

peak 감소는 `(B, S, V)` logits tensor(약 15.6 GiB @ bf16)와 그에 딸린 LlamaForCausalLM backbone activation(약 16 GiB)이 trainer에서 더 이상 생성되지 않아, batch 1 backward 시점의 grad_accum 누적 peak에서 동시에 사라진 결과다. 수학적 등가성은 algorithm이 이미 Phase A에서 `policy_out.get("logits")` None-safe 경로와 `use_flce` gate로 logits 미사용 경로를 구현해둔 덕에 유지된다.

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

GPU가 2개 이상이어도 `compute.distributed.strategy`가 명시되어야 분산 학습이 활성화된다. `compute.gpus: auto`만으로는 torchrun을 시작하지 않는다.

### 전략 비교

| 전략 | GPU 수 | 용도 | 메모리 효율 |
|------|--------|------|------------|
| **DDP** | 2~8 | 모델이 단일 GPU에 들어갈 때 | 낮음 (전체 복사) |
| **FSDP** | 2+ | 대형 모델 | 높음 (파라미터 샤딩) |

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

> FSDP + QLoRA는 비호환이다. 현재 안정 경로는 DDP다. 지원 전략 경계는 [Runtime Contracts](runtime-contracts.md#strategy-capability)를 참조한다.

### Expert Parallelism (MoE)

Mixture-of-Experts 모델에서 Expert Parallelism을 활성화한다:

```yaml
compute:
  distributed:
    strategy: ddp
    moe:
      enabled: true
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

콜백은 `--callbacks <yaml>` CLI 파일로만 주입한다. Recipe에는 `callbacks:` 필드가 없다.

### EarlyStopping (training.early_stopping 1급 필드)

EarlyStopping은 Recipe의 `training.early_stopping` 필드로 지정한다. `--callbacks`로 직접 지정하는 경로는 aliases에서 제거되어 막혀 있다.

```yaml
training:
  early_stopping:
    monitor: val_loss
    patience: 5
    mode: min
    min_delta: 0.001             # 최소 개선량
```

`patience` 동안 메트릭이 개선되지 않으면 학습을 조기 종료한다. Pydantic 타입 검증을 통해 YAML 로딩 시점에 잘못된 값(음수 patience 등)을 즉시 차단한다.

### EMACallback (training.ema 1급 필드)

동일하게 Recipe의 `training.ema` 필드로 지정한다.

```yaml
training:
  ema:
    decay: 0.9999
    update_after_step: 0
    update_every: 1
```

Exponential Moving Average — 매 스텝마다 모델 가중치의 이동 평균을 유지하고, `on_train_end`에서 EMA 가중치를 모델에 복사하여 최종 평가/저장 대상으로 사용한다.

### ModelCheckpoint

`--callbacks <yaml>`로 주입한다.

```yaml
# callbacks.yaml
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

---

## Observability

Training writes structured JSON results, MLflow params/metrics/tags, rank-aware text logs, and graceful shutdown status through the shared observability contract. The canonical reference is [Observability](observability.md).

Training-specific hooks in this guide:

- `ModelCheckpoint` contributes to `checkpoints_saved`.
- `Trainer` and `RLTrainer` both report `stopped_reason`.
- `monitoring.log_every_n_steps`, `monitoring.memory_history`, and `monitoring.verbose` configure training log behavior.
- `mdp inference` and `mdp generate` do not currently install the training signal handler.


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

## Failure And Recovery

Training handles SIGTERM/SIGINT at step boundaries and records the outcome in `stopped_reason`; see [Observability](observability.md#graceful-shutdown). Resume behavior is configured through `job.resume` and checkpoint layout is described above in this guide.

MDP does not hide process-level failures with automatic in-process recovery. Agents should use checkpoints, `job.resume`, and the structured result fields described in [Observability](observability.md#error-recovery).


## 주의사항

1. **loss 생략 시**: HuggingFace AutoModelFor\*는 labels가 있으면 `outputs.loss`를 반환하므로 생략 가능. 커스텀 SFT 모델은 forward output에 Tensor `loss`를 반환하거나 `loss:`를 지정한다
2. **DDP에서 drop_last**: `false`로 설정하면 GPU별 배치 크기 불일치로 gradient 동기화 실패. 기본 `true` 유지
3. **대형 모델 torch_dtype**: 미지정 시 fp32로 로드되어 OOM. `float16` 또는 `bfloat16` 지정 필수
4. **gradient_accumulation**: `batch_size × gpus × accumulation_steps = 실질 배치 크기`. accumulation 변경 시 학습률 조정 필요
5. **warmup_steps/warmup_ratio**: 상호 배타. 하나만 지정
6. **bf16 precision**: Ampere+ GPU(A100, RTX 3090+) 필요
