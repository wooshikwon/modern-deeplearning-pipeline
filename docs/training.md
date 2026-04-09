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

둘 다 지정하면 먼저 도달하는 조건에서 종료된다.

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

### 지원 알고리즘

| 알고리즘 | 필요 모델 | 생성 필요 | 용도 |
|---------|----------|---------|------|
| **DPO** | policy + reference | 아니오 | 선호 최적화 |
| **weighted-NTP** | policy + critic | 아니오 | 토큰 수준 선호 |
| **GRPO** | policy + reference + reward | 예 (K개 응답) | 그룹 상대 최적화 |
| **PPO** | policy + reference + value + reward | 예 | 근접 정책 최적화 |

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
  every_n_steps: 500           # 스텝 기반 저장 (검증과 무관)
```

저장 구조:
- `latest` → 가장 최근 체크포인트 symlink
- `best` → 모니터링 메트릭 기준 최고 체크포인트 symlink

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

## 에러 복구 전략

MDP는 프로세스 내 복구를 시도하지 않는다. 대신:

1. `ModelCheckpoint`로 주기적 저장
2. `job.resume: auto`로 최신 체크포인트에서 재시작
3. `torchrun --max_restarts`로 분산 환경에서 자동 재시작
4. MLflow/모니터링 실패는 경고만 출력하고 학습은 계속

---

## 주의사항

1. **loss 생략 시**: HuggingFace AutoModelFor\*는 내장 loss가 있어 생략 가능. 커스텀 모델은 `training_step()` 구현 또는 `loss:` 지정 필수
2. **DDP에서 drop_last**: `false`로 설정하면 GPU별 배치 크기 불일치로 gradient 동기화 실패. 기본 `true` 유지
3. **대형 모델 torch_dtype**: 미지정 시 fp32로 로드되어 OOM. `float16` 또는 `bfloat16` 지정 필수
4. **gradient_accumulation**: `batch_size × gpus × accumulation_steps = 실질 배치 크기`. accumulation 변경 시 학습률 조정 필요
5. **warmup_steps/warmup_ratio**: 상호 배타. 하나만 지정
6. **bf16 precision**: Ampere+ GPU(A100, RTX 3090+) 필요
