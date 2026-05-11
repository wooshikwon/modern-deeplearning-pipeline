# Configuration Guide

MDP는 **Recipe** + **Config** + **Callbacks** 세 파일로 설정을 관리한다.

## 설계 원칙

| 파일 | 역할 | 관심사 |
|------|------|--------|
| **Recipe** | 실험 정의 | 무엇을 학습할지 — 모델, 데이터, 하이퍼파라미터 |
| **Config** | 인프라 설정 | 어디서 실행할지 — GPU, 분산 전략, MLflow, 체크포인트 |
| **Callbacks** | 관측/개입 | 체크포인트 저장, 로깅, early stopping, 분석 |

동일 Recipe를 다른 Config로 실행하면 같은 실험을 다른 환경에서 재현할 수 있다.

---

## `_component_` 패턴

MDP의 모든 pluggable component는 동일한 `_component_` 문법을 사용한다:

```yaml
optimizer:
  _component_: AdamW          # 내장 alias → torch.optim.AdamW
  lr: 0.001
  weight_decay: 0.01

# 커스텀 컴포넌트도 동일 문법
optimizer:
  _component_: my_package.MyOptimizer   # 점(.) 포함 = 풀 import 경로
  lr: 0.001
```

**해석 규칙**:
- 점(`.`)이 없으면 `mdp/aliases.yaml`에서 풀 경로로 치환
- 점이 있으면 그대로 `importlib`으로 import
- 나머지 키는 `kwargs`로 생성자에 전달
- 중첩된 `_component_` dict도 재귀적으로 해석

YAML은 `_component_` dict 문법을 유지하지만, schema validation 이후 내부 runtime은 typed component envelope를 사용한다. 따라서 `_component_` 누락, 문자열이 아닌 `_component_`, non-model component의 `pretrained`, `target`/`target_modules` 또는 `save`/`modules_to_save` 동시 지정은 component constructor까지 내려가기 전에 실패한다.

내장 alias 목록은 `mdp list callbacks`, `mdp list strategies` 등으로 조회할 수 있다.

**`_component_`를 쓰지 않는 영역**: `training`, `generation`, `data.dataloader`, `metadata`, `evaluation` — 이들은 순수 설정값 묶음이다.

### Closed zone / open zone

Schema validation separates MDP-owned settings from user component kwargs.
Closed zones reject unknown fields immediately, while open zones keep arbitrary
kwargs under a typed `_component_` envelope.

| Zone | YAML area | Rule |
|---|---|---|
| Closed | `training`, `generation`, `rl.generation`, `data.dataloader`, `metadata`, `evaluation`, `monitoring` | Unknown fields are errors. Example: `training.val_check_units` fails because the supported field is `val_check_unit`. |
| Closed | `config.compute`, `config.mlflow`, `config.storage`, `config.serving`, `config.job` | Infrastructure typos are errors before runtime starts. |
| Open | `model`, `head`, `adapter`, `data.dataset`, `data.val_dataset`, `data.collator`, `data.sampler` | `_component_` or model `pretrained` route is validated; remaining keys are passed as component kwargs. |
| Open | `optimizer`, `scheduler`, `loss`, `rl.algorithm`, `rl.models.*`, callback YAML items | Component envelope is validated; component-specific kwargs remain open. |

YAML loader also rejects empty files, non-mapping Recipe/Config roots, non-list callback roots, and duplicate keys with a YAML dot-path in the error message.

### Schema migration table

| Legacy/error form | Current form |
|---|---|
| `data.source` at `data` top level | Move it under `data.dataset.source`; dataset-specific fields live inside `data.dataset`. |
| `callbacks:` in Recipe | Put callback blocks in a separate file and pass `--callbacks callbacks.yaml`. |
| Catalog `adapter: qlora` shorthand | Generated/public Recipe YAML uses `adapter: {_component_: QLoRA, ...}`. Catalog defaults should store the expanded block. |
| `model.pretrained: timm://...` plus `_component_` | Use protocol-specific model routing. `timm://` and `ultralytics://` pretrained routes omit `_component_`; generic/HF routes may include `_component_`. |
| `target` plus `target_modules` | Choose one: semantic `target` or raw backend `target_modules`. The same rule applies to `save` and `modules_to_save`. |

---

## Recipe YAML

### 전체 구조

```yaml
name: experiment-name                   # 실험 이름
task: text_generation                   # 태스크 (9종)
metadata:
  author: name
  description: 설명

model:                                  # 모델 (_component_ 패턴)
  _component_: AutoModelForCausalLM
  pretrained: hf://meta-llama/Llama-3-8B
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2

head:                                   # 출력 head 교체 (선택)
  _component_: ClassificationHead
  _target_attr: head                    # 교체할 모델 속성명
  num_classes: 10

adapter:                                # 파라미터 효율 학습 (선택, 생략 시 full fine-tuning)
  _component_: LoRA
  r: 16
  alpha: 32
  target_modules: [q_proj, v_proj]

data:                                   # 데이터 파이프라인
  dataset:                              # 학습 Dataset (_component_ 패턴)
    _component_: HuggingFaceDataset
    source: dataset-name
    split: train
    fields: {text: text_column}
    tokenizer: model-name
    max_length: 2048
  val_dataset:                          # 검증 Dataset (선택, 생략 시 검증 비활성)
    _component_: HuggingFaceDataset
    source: dataset-name
    split: validation
    fields: {text: text_column}
  collator:                             # Collator (_component_ 패턴)
    _component_: CausalLMCollator
    tokenizer: model-name
    max_length: 2048
  sampler:                              # Sampler (선택 — length-bucketed sampling 활성)
    _component_: LengthGroupedBatchSampler
    bucket_size: 256                    # 기본값: batch_size * 8
    shuffle_buckets: true
  dataloader:                           # DataLoader 설정 (순수 설정값)
    batch_size: 4
    num_workers: 4
    drop_last: true

training:                               # 학습 루프 설정 (순수 설정값)
  epochs: 3                             # epochs 또는 max_steps 중 하나 필수 (float 허용)
  max_steps: 10000                      # epochs와 동시 지정 가능 — early-hit 규칙 적용
  precision: bf16                       # fp32 | fp16 | bf16
  gradient_accumulation_steps: 4
  gradient_clip_max_norm: 1.0
  gradient_checkpointing: false
  val_check_interval: 1.0               # 검증 주기
  val_check_unit: epoch                 # epoch | step
  compile: false                        # torch.compile 활성화

optimizer:                              # 옵티마이저 (_component_ 패턴)
  _component_: AdamW
  lr: 3e-4

scheduler:                              # 스케줄러 (선택)
  _component_: CosineAnnealingLR
  T_max: 100
  warmup_steps: 500                     # 또는 warmup_ratio: 0.1 (상호 배타)

loss:                                   # 외부 supervised criterion (선택, 생략 시 forward output loss 사용)
  _component_: CrossEntropyLoss

evaluation:                             # 평가 메트릭
  metrics: [Accuracy, F1Score]

monitoring:                             # 드리프트 모니터링 (선택)
  enabled: false
  baseline: {max_batches: 100}
  drift: {method: jensen_shannon, threshold: 0.1}
```

### Training Duration (epochs × max_steps)

`training.epochs`와 `training.max_steps`는 상호 배타가 아니라 **공존 가능**하다. Pydantic validator `check_training_duration`는 "최소 하나 필수"만 강제한다.

| 설정 | 동작 |
|---|---|
| `epochs`만 지정 | 지정한 epoch 수까지 학습 |
| `max_steps`만 지정 | 지정한 전역 step 수까지 학습 |
| 둘 다 지정 | **먼저 도달한 조건에서 종료** (early-hit). 학습 시작 시 INFO 로그 1회 ("epochs=..., max_steps=... 모두 지정됨. 먼저 도달한 조건에서 종료됩니다") |

둘 다 지정하는 사용 패턴은 의도된 설계다. 예를 들어 `epochs: 3`을 안전판으로 두고 `max_steps: 200`으로 실제 중단 기준을 지정하거나, recipe에 `epochs`만 두고 CLI `--override training.max_steps=200`으로 주입해도 된다.

### ModelCheckpoint 콜백 옵션

```yaml
# callbacks.yaml
- _component_: ModelCheckpoint
  dirpath: ./checkpoints        # 생략 시 config.storage.checkpoint_dir 사용
  monitor: val_loss             # metric 이름
  mode: min                     # min (loss) | max (accuracy)
  save_top_k: 3                 # 상위 k개만 유지 (나머지 삭제)
  every_n_steps: 500            # 설정 시 매 N step마다 저장 (검증과 무관)
  strict: false                 # 기본. true이면 첫 validation에서 monitor 미매칭 시 ValueError
```

콜백 YAML은 `--callbacks callbacks.yaml`로 전달한다. Recipe에는 `callbacks:` 필드가 없다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `dirpath` | `config.storage.checkpoint_dir` | 체크포인트 루트 디렉토리. 명시 시 storage override를 무시 |
| `monitor` | `"val_loss"` | validation metric 이름. `save_top_k`·`best` symlink 갱신 기준 |
| `mode` | `"min"` | `"min"` 또는 `"max"`. loss 계열은 min, accuracy/f1/auc는 max |
| `save_top_k` | `3` | 상위 k개 체크포인트만 유지 |
| `every_n_steps` | `None` | 정수 지정 시 step 기반 저장 활성화 (validation과 독립) |
| `strict` | `False` | `True`이면 monitor 미매칭 시 즉시 `ValueError` — silent skip 차단 |

`strict`는 자동 스윕 환경에서 "학습은 성공, 산출물은 0개"인 silent failure를 즉시 감지하는 방어선이다. 자세한 동작과 `critical`과의 차이는 [Training Guide](training.md)의 ModelCheckpoint 섹션 참조.

### Task별 필수 fields

| Task | 필수 fields |
|------|------------|
| `text_generation` | `{text}` |
| `text_classification` | `{text, label}` |
| `token_classification` | `{text, token_labels}` |
| `seq2seq` | `{text, target}` |
| `image_classification` | `{image, label}` |
| `object_detection` | `{image, label}` |
| `semantic_segmentation` | `{image, label}` |
| `image_generation` | `{image, text}` |
| `feature_extraction` | (없음) |

### 내장 Dataset 클래스

| 클래스 | 용도 |
|--------|------|
| `HuggingFaceDataset` | HF Hub 또는 로컬(csv/json/parquet/imagefolder) |
| `ImageClassificationDataset` | ImageFolder + augmentation |

### 내장 Collator 클래스

| 클래스 | 용도 | 출력 키 |
|--------|------|---------|
| `CausalLMCollator` | 언어 모델 (GPT, Llama 등) | `input_ids, labels` |
| `PreferenceCollator` | DPO 선호 학습 | `chosen_input_ids, rejected_input_ids, ...` |
| `Seq2SeqCollator` | Encoder-Decoder (T5 등) | `input_ids, labels` |
| `ClassificationCollator` | 텍스트 분류 | `input_ids, labels` |
| `TokenClassificationCollator` | 토큰 분류 (NER 등) | `input_ids, labels` |
| `VisionCollator` | 비전 태스크 | `pixel_values, labels` |

### 내장 Sampler 클래스 (선택)

`data.sampler`는 optional. 미지정 시 기존 동작(`distributed=True` → `DistributedSampler` 자동, 아니면 `shuffle=True`)을 100% 보존한다. 지정 시 train DataLoader에 `batch_sampler=...`로 주입되며, `data.dataloader`의 `batch_size`/`shuffle`/`drop_last`는 sampler 책임으로 위임된다 (DataLoader 표준 계약).

| 클래스 | 용도 | 적용 환경 |
|--------|------|----------|
| `LengthGroupedBatchSampler` | Length-bucketed batching (padding 토큰 절감) | Single-GPU |
| `DistributedLengthGroupedBatchSampler` | Length-bucketed + DDP rank partition (megabatch) | DDP/FSDP |

**주요 인자 (single-GPU `LengthGroupedBatchSampler`)**:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `bucket_size` | `batch_size * 8` | bucket 한 단위 크기. 작을수록 randomness 보존, 클수록 padding 절감 |
| `shuffle_buckets` | `true` | epoch 내부에서 batch 순서를 한 번 더 셔플 (학습 안정성) |
| `length_fn` | `null` | dataset이 `__getlength__`를 구현하지 않을 때만 명시. `Callable[[sample], int]` |
| `seed` | `0` | 결정적 셔플의 base seed. 실제 generator seed는 `seed + epoch` |
| `drop_last` | `false` | 마지막 미완성 batch 처리 |

**Distributed 환경에서는 `DistributedLengthGroupedBatchSampler`를 명시한다**. 단일 클래스로 distributed 분기를 캡슐화하므로 `DistributedSampler`로 외부 wrapping은 권장하지 않는다 — wrapping 시 rank 간 길이 정렬이 깨진다. `num_replicas`/`rank`는 명시하지 않으면 `torch.distributed`에서 자동 조회한다. distributed sampler에서 `bucket_size`는 인자로 받지만 실제 동작에는 사용되지 않으며 분할 단위는 megabatch(`num_replicas * batch_size`)로 고정된다.

**Recipe YAML 예시 — Single-GPU**:

```yaml
data:
  dataset:
    _component_: HuggingFaceDataset
    source: my-dataset
    tokenizer: meta-llama/Llama-3-8B
    max_length: 2048
  collator:
    _component_: CausalLMCollator
    tokenizer: meta-llama/Llama-3-8B
  sampler:
    _component_: LengthGroupedBatchSampler
    bucket_size: 256
  dataloader:
    batch_size: 32
```

**Recipe YAML 예시 — DDP/FSDP**:

```yaml
data:
  sampler:
    _component_: DistributedLengthGroupedBatchSampler
    # bucket_size는 distributed에서 사용되지 않음 — 생략 권장
    # num_replicas, rank는 torch.distributed에서 자동 조회
  # 그 외 dataset/collator/dataloader는 single-GPU와 동일
```

`__getlength__` Protocol을 구현하지 않은 커스텀 dataset에 length-bucketed sampler를 적용하려면 `length_fn`을 명시한다 — 자세한 구현 패턴은 [Extending MDP](extending.md)의 "커스텀 Sampler" 섹션 참조.

### 어댑터 설정

**LoRA**:
```yaml
adapter:
  _component_: LoRA
  r: 16                          # rank
  alpha: 32                      # scaling factor
  dropout: 0.05
  target_modules: [q_proj, v_proj]
```

**QLoRA** (양자화 + LoRA):
```yaml
adapter:
  _component_: QLoRA
  r: 64
  alpha: 128
  quantization:
    bits: 4                      # 4bit 또는 8bit
  modules_to_save: [lm_head, embed_tokens]
```

현재 QLoRA adapter 구현은 HuggingFace `device_map="auto"` 양자화 로딩을 사용한다. Trainer/RLTrainer는 `hf_device_map`이 붙은 모델을 학습 경로에서 거부하므로, QLoRA는 학습 안정 경로로 문서화하지 않는다. LoRA 또는 Prefix Tuning을 사용한다.

**Prefix Tuning**:
```yaml
adapter:
  _component_: PrefixTuning
  r: 16                          # num_virtual_tokens
```

### RL 학습 설정 (rl-train 전용)

`rl:` 키 아래에 알고리즘과 역할별 모델을 정의한다:

```yaml
rl:
  algorithm:
    _component_: DPO
    beta: 0.1

  models:
    policy:                                 # 학습 대상
      _component_: AutoModelForCausalLM
      pretrained: hf://model-name
      optimizer: {_component_: AdamW, lr: 1e-5}
      scheduler: {_component_: CosineAnnealingLR, T_max: 5000}
      adapter: {_component_: LoRA, r: 16}   # 선택
    reference:                              # frozen (optimizer 생략)
      _component_: AutoModelForCausalLM
      pretrained: hf://model-name

  generation:                               # GRPO/PPO 전용
    max_new_tokens: 512
    temperature: 0.8
    group_size: 4                           # GRPO K개 응답
```

내장 알고리즘: `DPO`, `GRPO`, `PPO`. 외부 알고리즘은 `rl.algorithm._component_`에 풀 경로(`weighted_ntp.algorithms.WeightedNTPLoss` 등)를 지정하면 주입된다. RL recipe에서는 top-level `loss:`를 사용하지 않는다. RL objective는 항상 `rl.algorithm.compute_loss(trainable_out, frozen_out, batch)`가 소유한다.

---

## Config YAML

### 전체 구조

```yaml
compute:
  target: local                             # local (기본)
  gpus: auto                                # auto | 1 | [0, 1, 2, 3]
  distributed:
    strategy: ddp                           # ddp | fsdp | auto. deepspeed* is currently fail-fast
    # 전략별 추가 설정은 kwargs로 전달

mlflow:
  tracking_uri: ./mlruns                    # MLflow URI
  experiment_name: my-experiment

storage:
  checkpoint_dir: ./checkpoints
  checkpoint_every_n_steps: 500             # reserved; step save는 ModelCheckpoint.every_n_steps 사용
  output_dir: ./outputs

serving:                                    # mdp serve 전용 (선택 — 생략 시 기본값)
  # --- 실제 동작하는 필드 ---
  max_batch_size: 1                         # PredictHandler 동적 배칭 상한
  batch_window_ms: 50.0                     # 동적 배칭 시간 창 (ms)
  device_map: auto                          # auto | balanced | sequential — from_pretrained(device_map=)로 전달
  max_memory: {"0": "24GiB", "1": "40GiB"}  # GPU별 상한. Config YAML에서는 dict, CLI --max-memory는 JSON 문자열
  # --- Reserved (schema·validator 존재, serve.py 미사용) ---
  backend: torchserve                       # 현재 mdp serve는 uvicorn+FastAPI 고정. validator가 vllm+비호환 task만 차단.
  model_repository: null                    # 향후 TorchServe 통합용 예약 필드
  instance_count: 1                         # 향후 multi-worker 대비 예약 필드

job:
  name: null                                # 선택. 지정 시 checkpoint_dir/{recipe.name}/{name} 사용
  resume: auto                              # disabled | auto | 체크포인트 경로
  max_retries: 0
```

로컬 단일 GPU 학습이면 Config를 최소화해도 기본값으로 동작한다:

```yaml
compute:
  gpus: 1
mlflow:
  tracking_uri: ./mlruns
storage:
  checkpoint_dir: ./checkpoints
```

### 분산 전략 설정

**DDP** (2~8 GPU, 모델이 단일 GPU에 들어갈 때):
```yaml
compute:
  gpus: auto
  distributed:
    strategy: ddp
```

**FSDP** (모델이 단일 GPU를 초과할 때):
```yaml
compute:
  gpus: auto
  distributed:
    strategy: fsdp
    sharding_strategy: FULL_SHARD
    mixed_precision: true
    precision: bf16
```

지원 전략의 런타임 경계는 [Runtime Contracts](runtime-contracts.md#strategy-capability)를 참조한다.

`compute.distributed` is a closed config block. Unknown keys such as `stratgey` fail before runtime starts. The `strategy` slot accepts either a string alias or a `_component_` block:

```yaml
compute:
  gpus: auto
  distributed:
    strategy:
      _component_: DDPStrategy
      backend: gloo
```

Do not repeat the same strategy kwarg in both places. For example, `strategy.backend` and top-level `distributed.backend` together are rejected because the effective constructor value would be ambiguous.

**MoE Expert Parallelism**:
```yaml
compute:
  distributed:
    strategy: ddp
    moe:
      enabled: true
      ep_size: 4
```

---

## Callbacks YAML

`--callbacks` 옵션으로 별도 파일을 지정한다. Recipe에는 `callbacks:` 섹션이 없다.

```yaml
# callbacks.yaml
- _component_: ModelCheckpoint
  monitor: val_loss
  mode: min
  save_top_k: 3
  every_n_steps: 500
```

학습률 로깅은 Trainer builtin 경로가 처리하므로 별도 콜백이 필요 없다. 로깅 키와 MLflow 규약은 [Observability](observability.md)를 참조한다.

`--callbacks` 파일이 지정되지 않으면 callback은 주입되지 않는다. EarlyStopping/EMA처럼 학습 결과 자체에 영향을 주는 동작은 Recipe의 `training.*` 1급 필드를 사용한다.

### 내장 콜백

| 콜백 | 설명 |
|------|------|
| `ModelCheckpoint` | 체크포인트 저장 (best/latest, top-k) |

`EarlyStopping`과 EMA는 callback alias가 아니라 Recipe의 `training.early_stopping` / `training.ema` 필드로 설정한다.

---

## 런타임 Override

CLI에서 Recipe/Config 필드를 덮어쓸 수 있다. 문법과 예시는 [CLI Reference](cli-reference.md#global-options)를 참조한다.

---

## Validation

학습 시작 전 3단 검증이 실행되어 설정 오류를 사전에 차단한다:

1. **Catalog 검증**: 필드 타입, 필수 키 존재 여부
2. **Business 검증**: task-head 호환성, adapter 제약, RL/data 설정, component import warning
3. **Compat 검증**: GPU/distributed 설정, DeepSpeed fail-fast 경계, serving backend 호환성, FSDP + QLoRA 비호환, MoE 설정

모든 에러가 한 번에 수집·보고되므로 하나씩 고칠 필요 없이 전체 문제를 파악할 수 있다.

---

## 환경 변수

Config에서 환경 변수를 참조할 수 있다:

```yaml
mlflow:
  tracking_uri: "${MLFLOW_URI:./mlruns}"    # 환경 변수 또는 기본값
```
