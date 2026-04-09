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

내장 alias 목록은 `mdp list callbacks`, `mdp list strategies` 등으로 조회할 수 있다.

**`_component_`를 쓰지 않는 영역**: `training`, `generation`, `data.dataloader`, `metadata`, `evaluation` — 이들은 순수 설정값 묶음이다.

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
  dataloader:                           # DataLoader 설정 (순수 설정값)
    batch_size: 4
    num_workers: 4
    drop_last: true

training:                               # 학습 루프 설정 (순수 설정값)
  epochs: 3                             # epochs 또는 max_steps 중 하나 필수 (float 허용)
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

loss:                                   # 손실 함수 (선택, 생략 시 model.training_step() 사용)
  _component_: CrossEntropyLoss

callbacks:                              # 콜백 리스트
  - _component_: ModelCheckpoint
    monitor: val_loss
    mode: min
  - _component_: EarlyStopping
    monitor: val_loss
    patience: 3

evaluation:                             # 평가 메트릭
  metrics: [Accuracy, F1Score]

monitoring:                             # 드리프트 모니터링 (선택)
  enabled: false
  baseline: {max_batches: 100}
  drift: {method: jensen_shannon, threshold: 0.1}
```

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

지원 알고리즘: `DPO`, `weighted-NTP`, `GRPO`, `PPO`

---

## Config YAML

### 전체 구조

```yaml
compute:
  target: local                             # local (기본)
  gpus: auto                                # auto | 1 | [0, 1, 2, 3]
  distributed:
    strategy: ddp                           # ddp | fsdp | deepspeed_zero3 | auto
    # 전략별 추가 설정은 kwargs로 전달

mlflow:
  tracking_uri: ./mlruns                    # MLflow URI
  experiment_name: my-experiment

storage:
  checkpoint_dir: ./checkpoints
  output_dir: ./outputs

serving:                                    # mdp serve 전용
  max_batch_size: 1
  batch_window_ms: 50.0
  device_map: auto                          # auto | balanced | sequential
  max_memory: '{"0": "24GiB", "1": "40GiB"}'

job:
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

**DeepSpeed ZeRO-3** (극대형 모델):
```yaml
compute:
  gpus: auto
  distributed:
    strategy: deepspeed_zero3
```

**MoE Expert Parallelism**:
```yaml
compute:
  distributed:
    strategy: ddp
    expert_parallel:
      ep_size: 4
```

---

## Callbacks YAML

`--callbacks` 옵션으로 별도 파일을 지정하거나, Recipe의 `callbacks:` 섹션에 직접 작성한다.

```yaml
# callbacks.yaml
- _component_: ModelCheckpoint
  monitor: val_loss
  mode: min
  save_top_k: 3
  every_n_steps: 500

- _component_: EarlyStopping
  monitor: val_loss
  patience: 3

- _component_: EMACallback
  decay: 0.999

- _component_: LearningRateMonitor

- _component_: ProgressBar
```

**병합 규칙**: `--callbacks` 파일이 지정되면 Recipe의 `callbacks:` 섹션을 override한다.

### 내장 콜백

| 콜백 | 설명 |
|------|------|
| `ModelCheckpoint` | 체크포인트 저장 (best/latest, top-k) |
| `EarlyStopping` | 메트릭 정체 시 조기 종료 |
| `EMACallback` | Exponential Moving Average |
| `LearningRateMonitor` | 학습률 로깅 |
| `ProgressBar` | Rich 진행 바 |

---

## 런타임 Override

CLI에서 Recipe/Config 필드를 덮어쓸 수 있다:

```bash
mdp train -r recipe.yaml -c config.yaml \
  --override training.epochs=5 \
  --override data.dataloader.batch_size=8 \
  --override config.storage.checkpoint_dir=./ckpts
```

- `config.` 접두사: Config 필드
- 그 외: Recipe 필드
- dot notation으로 중첩 접근 가능

---

## Validation

학습 시작 전 3단 검증이 실행되어 설정 오류를 사전에 차단한다:

1. **Catalog 검증**: 필드 타입, 필수 키 존재 여부
2. **Business 검증**: task-head 호환성, adapter 제약 (QLoRA + head 금지), forward contract
3. **Compat 검증**: precision 호환 (bf16 → Ampere+ GPU), FSDP + QLoRA 비호환

모든 에러가 한 번에 수집·보고되므로 하나씩 고칠 필요 없이 전체 문제를 파악할 수 있다.

---

## 환경 변수

Config에서 환경 변수를 참조할 수 있다:

```yaml
mlflow:
  tracking_uri: "${MLFLOW_URI:./mlruns}"    # 환경 변수 또는 기본값
```
