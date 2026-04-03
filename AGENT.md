# MDP -- Modern Deeplearning Pipeline

YAML 설정으로 딥러닝 모델의 학습, 추론, 서빙을 수행하는 CLI 도구.

## Commands

| Command | Purpose |
|---------|---------|
| `mdp init <name> --task <task> --model <model>` | 프로젝트 스캐폴딩 + Recipe 템플릿 생성 |
| `mdp train -r recipe.yaml -c config.yaml` | SFT 학습 |
| `mdp rl-train -r rl-recipe.yaml -c config.yaml` | RL alignment 학습 (DPO, weighted-NTP, GRPO, PPO) |
| `mdp inference --run-id <id> --data <path>` | 배치 추론 + 평가 |
| `mdp estimate -r recipe.yaml` | GPU 메모리 추정 + 전략 추천 |
| `mdp export --run-id <id> --output <dir>` | adapter merge + 서빙용 패키징 (`--checkpoint`로 로컬 체크포인트도 가능) |
| `mdp serve --run-id <id>` | REST API 서빙 |
| `mdp list models\|tasks\|callbacks\|strategies` | 카탈로그 조회 |

모든 명령은 `--format json` 옵션을 지원한다. 각 명령의 상세 인자는 `mdp <command> --help`로 확인.

## Agent Discovery Flow

```bash
mdp list tasks                                    # 지원 태스크 조회
mdp list models --task text_generation            # 호환 모델 조회
mdp init my_project --task text_generation --model llama3-8b  # 스캐폴딩
# Recipe의 ??? 필드를 채운 후:
mdp estimate -r recipe.yaml --format json         # 메모리 추정
mdp train -r recipe.yaml -c config.yaml --format json  # 학습
mdp inference --run-id <id> --data test.jsonl --format json  # 추론
```

`mdp init`이 생성하는 Recipe에서 `???`로 표시된 필드만 채우면 된다: `data.source`, `data.fields.*`, `metadata.*`

---

## Two-File System

- **Recipe** (실험 정의): 무엇을 학습할지 — 모델, 데이터, 하이퍼파라미터
- **Config** (인프라 설정): 어디서 실행할지 — GPU, 분산 전략, MLflow, 체크포인트 경로

동일 Recipe를 다른 Config로 실행하면 같은 실험을 다른 환경에서 재현할 수 있다.

---

## Recipe YAML — 구조

```
name: str (실험 이름)
task: str (9종: image_classification, object_detection, semantic_segmentation,
           text_classification, token_classification, text_generation,
           seq2seq, image_generation, feature_extraction)

model:                          # 모델 아키텍처 + pretrained 가중치
  class_path: str               # Python 클래스 경로 (예: transformers.AutoModelForCausalLM)
  pretrained: str               # URI (hf://, timm://, ultralytics://, local://)
  torch_dtype: str              # float32 | float16 | bfloat16
  attn_implementation: str      # eager | sdpa | flash_attention_2
  init_args: dict               # 모델 생성자 추가 인자

head:                           # 출력 head 교체 (AutoModelFor* 사용 시 생략 가능)
  _component_: str              # Head alias (ClassificationHead, CausalLMHead 등)
  _target_attr: str             # 모델에서 교체할 속성명 (예: head, classifier, fc)
  ...                           # Head 생성자 인자 (num_classes, hidden_dim 등)

adapter:                        # 파라미터 효율 학습 (생략 = full fine-tuning)
  method: str                   # lora | qlora | prefix_tuning
  r: int                        # LoRA rank 또는 prefix 토큰 수
  alpha: int                    # LoRA alpha (→ PEFT lora_alpha로 자동 매핑)
  dropout: float                # LoRA dropout (→ PEFT lora_dropout로 자동 매핑)
  target_modules: list | str    # 적용 대상 모듈 (기본 "all_linear")
  quantization:                 # QLoRA 전용
    bits: 4 | 8
  modules_to_save: list         # freeze하지 않을 모듈 (예: [lm_head, embed_tokens])

data:                           # 데이터 파이프라인
  source: str                   # HF Hub 이름 또는 로컬 파일/디렉토리 경로
  fields: {role: column}        # 역할→컬럼 매핑 (image, text, label, chosen, rejected 등)
  split: str                    # 학습 split (기본 "train")
  val_split: str | null         # 검증 split ("auto" = 자동 추론, null = 비활성화, 문자열 = 직접 지정)
  tokenizer:                    # 언어 태스크용 (비전에서는 무시됨)
    pretrained: str
    max_length: int
  augmentation:                 # 비전 태스크용 torchvision DSL
    train/val:
      steps: [{type: str, params: dict}, ...]
  dataloader:
    batch_size: int
    num_workers: int

training:                       # 학습 루프 설정
  epochs: int                   # epochs 또는 max_steps 중 하나 필수
  max_steps: int
  precision: str                # fp32 | fp16 | bf16
  gradient_accumulation_steps: int
  gradient_clip_max_norm: float
  gradient_checkpointing: bool
  val_check_interval: int | float  # validation 주기
  val_check_unit: str           # "epoch" (기본) 또는 "step"
  compile: bool | str           # torch.compile 모드

optimizer:                      # _component_ 패턴
  _component_: str              # AdamW, Adam, SGD 또는 풀 경로
  lr: float
  ...

scheduler:                      # _component_ 패턴 (선택)
loss:                           # _component_ 패턴 (선택 — 생략 시 model.training_step() 사용)
callbacks: [...]                # _component_ 패턴 리스트
evaluation:
  metrics: [...]                # _component_ 패턴 리스트

# RL 전용 (mdp rl-train)
algorithm:                      # _component_ 패턴 (DPO, GRPO, PPO)
models:                         # 역할별 모델 정의
  policy: {model: ..., optimizer: ..., scheduler: ...}
  reference: {model: ...}       # frozen (optimizer 없음)
  value: {model: ..., optimizer: ...}  # PPO 전용
  reward: {model: ...}          # frozen
generation:                     # GRPO/PPO 전용
  max_new_tokens: int
  temperature: float
  group_size: int               # GRPO K개 응답

metadata:
  author: str
  description: str
```

> 필드별 타입·기본값·검증 규칙은 `mdp init`이 생성하는 Recipe 템플릿의 주석에 포함되어 있다.

---

## Config YAML — 구조

```
compute:
  target: str                   # local (기본)
  gpus: int | "auto"
  distributed:
    strategy: str               # ddp | fsdp | deepspeed_zero2 | deepspeed_zero3
    ...                         # 전략별 추가 옵션

mlflow:
  tracking_uri: str             # 기본 ./mlruns
  experiment_name: str

storage:
  checkpoint_dir: str
  output_dir: str

serving:                        # mdp serve 전용
  backend: str                  # torchserve | vllm | onnx
  max_batch_size: int
  batch_window_ms: float

job:
  name: str
  resume: str                   # disabled | auto | 체크포인트 경로
  max_retries: int
```

> 로컬 단일 GPU 학습이면 Config 전체를 생략해도 기본값으로 동작한다.

---

## `_component_` Pattern

임의의 Python 클래스를 선택해야 하는 영역(optimizer, scheduler, loss, head, callback, algorithm)에 사용:

```yaml
optimizer:
  _component_: AdamW        # 내장 alias
  lr: 0.001

optimizer:
  _component_: my_package.MyOptimizer  # 커스텀 (점 포함 = 풀 경로)
  lr: 0.001
```

`_component_`를 사용하지 않는 영역: `model` (class_path+pretrained), `data` (고정 스키마), `adapter` (method 분기), `training` (고정 스키마)

내장 alias 목록은 `mdp list callbacks`, `mdp list strategies` 등으로 조회.

> **DualEncoderHead 참고**: CLIP/SigLIP 학습 시 모델의 `training_step()`에서 `self.head.forward_pair(image_features, text_features)`를 직접 호출해야 한다. `forward()`는 추론 시 image projection만 수행한다.

---

## Validation Rules

학습 시작 전에 3단 검증(Catalog → Business → Compat)이 모든 에러를 수집하여 한 번에 보고한다.

### Task-Head Compatibility

| Task | Allowed Head | AutoModel (head 생략) |
|------|-------------|----------------------|
| image_classification | ClassificationHead | AutoModelForImageClassification |
| object_detection | DetectionHead | AutoModelForObjectDetection |
| semantic_segmentation | SegmentationHead | AutoModelForSemanticSegmentation |
| text_classification | ClassificationHead | AutoModelForSequenceClassification |
| token_classification | TokenClassificationHead, ClassificationHead | AutoModelForTokenClassification |
| text_generation | CausalLMHead | AutoModelForCausalLM |
| seq2seq | Seq2SeqLMHead | AutoModelForSeq2SeqLM |
| image_generation | (head 생략 권장) | 모델 내장 |
| feature_extraction | ClassificationHead, DualEncoderHead | (head 생략 가능) |

> `vision_language`는 별도 task가 아니다. multimodal은 `text_generation` 또는 `feature_extraction` + `fields: {image, text}`로 표현.

### Adapter Constraints

- `qlora` → `quantization.bits` 필수, `torch_dtype`는 `bfloat16` 또는 `float16`
- `lora`/`qlora`/`prefix_tuning` → `r` 필수
- `prefix_tuning`에서 `alpha`, `dropout`, `target_modules`는 자동 무시됨

### Forward Contract

- `loss` 지정 시: `model.forward(batch)` → `dict(logits=...)` 반환 필수
- `loss` 생략 시: `model.training_step(batch)` → `Tensor` (스칼라 loss) 반환 필수

### Precision & Distributed

- `bf16` → Ampere+ GPU (A100, RTX 3090+) 필요
- `flash_attention_2` → Ampere+ GPU 필요
- `fsdp` + `qlora` → **비호환** (대안: DDP, DeepSpeed ZeRO-3)
- 멀티 GPU 시 `distributed.strategy` 필수

### Data Constraints

- `data.source` 필수
- `data.tokenizer` → 언어 태스크에서만 유효 (비전 태스크에서 설정 시 경고)
- `data.fields` → task별 `required_fields`와 일치해야 함
- `streaming: true` + multimodal(vision+language) → 미지원

---

## Common Pitfalls

1. **loss 생략 + training_step 미구현**: HuggingFace AutoModelFor*는 내장 loss가 있어 생략 가능. 커스텀 모델은 반드시 지정하거나 `training_step()` 구현.

2. **DDP에서 drop_last: false**: GPU별 배치 크기가 달라져 gradient 동기화 실패. `drop_last: true` (기본) 유지.

3. **LoRA target_modules 불일치**: 모델에 없는 모듈명 → 에러. 정확한 이름은 모델의 `named_modules()` 출력 확인.

4. **gradient_accumulation과 배치 크기**: `batch_size * gpus * accumulation_steps = 실질 배치 크기`. accumulation을 늘리면 학습률도 조정 필요.

5. **warmup_steps와 warmup_ratio 동시 지정**: 상호 배타. 하나만 사용.

6. **대형 모델 torch_dtype 미지정**: fp32로 로드되어 OOM. 반드시 `float16` 또는 `bfloat16` 지정.

7. **vision 태스크에 tokenizer**: 무시되지만 불필요한 로딩 발생. 제거 권장.

8. **`_component_` 영역에서 `type:` 키 사용 금지**: `type`은 YAML 예약어가 아니지만 augmentation DSL과 혼동됨. `_component_` 패턴은 `_component_:` 키만 사용.

---

## Working Examples

### Vision: ViT LoRA Image Classification

**Recipe** (`recipes/vit-lora-cifar10.yaml`):

```yaml
name: vit-lora-cifar10
task: image_classification

model:
  class_path: timm.create_model
  pretrained: timm://vit_base_patch16_224

head:
  _component_: ClassificationHead
  num_classes: 10

adapter:
  method: lora
  r: 16
  alpha: 32
  target_modules: ["qkv", "proj"]

data:
  source: cifar10
  fields:
    image: img
    label: label
  augmentation:
    train:
      steps:
        - {type: RandomResizedCrop, params: {size: [224, 224]}}
        - {type: RandomHorizontalFlip}
        - {type: ToDtype, params: {dtype: float32, scale: true}}
        - {type: Normalize, params: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}}
    val:
      steps:
        - {type: Resize, params: {size: [256, 256]}}
        - {type: CenterCrop, params: {size: [224, 224]}}
        - {type: ToDtype, params: {dtype: float32, scale: true}}
        - {type: Normalize, params: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}}
  dataloader:
    batch_size: 64

training:
  epochs: 30
  precision: bf16
  gradient_clip_max_norm: 1.0

optimizer:
  _component_: AdamW
  lr: 0.001
  weight_decay: 0.01

scheduler:
  _component_: CosineAnnealingLR
  T_max: 30

loss:
  _component_: CrossEntropyLoss

callbacks:
  - _component_: EarlyStopping
    monitor: val_loss
    patience: 5
  - _component_: ModelCheckpoint
    monitor: val_accuracy
    mode: max

metadata:
  author: agent
  description: ViT-Base LoRA fine-tuning on CIFAR-10
```

**Config** (`configs/local-1gpu.yaml`):

```yaml
compute:
  gpus: 1

mlflow:
  tracking_uri: ./mlruns
  experiment_name: vit-cifar10

storage:
  checkpoint_dir: ./checkpoints/vit-lora-cifar10
  output_dir: ./outputs/vit-lora-cifar10
```

### Language: LLM QLoRA Fine-tuning

**Recipe** (`recipes/llm-qlora.yaml`):

```yaml
name: llm-qlora-chat
task: text_generation

model:
  class_path: transformers.AutoModelForCausalLM
  pretrained: hf://Qwen/Qwen2.5-7B
  torch_dtype: bfloat16

adapter:
  method: qlora
  r: 64
  alpha: 128
  quantization:
    bits: 4
  modules_to_save: [lm_head, embed_tokens]

data:
  source: HuggingFaceH4/ultrachat_200k
  fields:
    text: text
  tokenizer:
    pretrained: Qwen/Qwen2.5-7B
    max_length: 2048
  dataloader:
    batch_size: 4

training:
  max_steps: 1000
  precision: bf16
  gradient_accumulation_steps: 4
  gradient_checkpointing: true

optimizer:
  _component_: bitsandbytes.optim.AdamW8bit
  lr: 0.0002
  weight_decay: 0.01

scheduler:
  _component_: CosineAnnealingLR
  T_max: 1000

callbacks:
  - _component_: ModelCheckpoint
    every_n_steps: 200

metadata:
  author: agent
  description: Qwen 7B QLoRA on UltraChat
```

**Config** (`configs/local-deepspeed.yaml`):

```yaml
compute:
  gpus: auto
  distributed:
    strategy: deepspeed_zero3

mlflow:
  tracking_uri: ./mlruns
  experiment_name: llm-qlora

storage:
  checkpoint_dir: ./checkpoints/llm-qlora
  output_dir: ./outputs/llm-qlora
```

---

## CLI JSON Output

`--format json` 사용 시 stdout에 구조화된 JSON, stderr에 로그가 출력된다.

### Common Wrapper

```json
{
  "status": "success | error",
  "command": "train | inference | estimate | ...",
  "timestamp": "ISO-8601",
  "error": null | {"type": "ValidationError|RuntimeError", "message": "...", "details": {...}}
}
```

### Error Recovery

| error.type | 대응 |
|-----------|------|
| `ValidationError` | Recipe/Config YAML 수정 |
| `RuntimeError` | Config 조정 (batch_size 축소, precision 변경, GPU 수 조정) |

### Train Result (추가 필드)

```json
{
  "checkpoint_dir": "...",
  "metrics": {"val_loss": 0.12, "val_accuracy": 0.95},
  "total_epochs": 27,
  "total_steps": 12600,
  "stopped_reason": "early_stopped | completed | max_steps_reached",
  "duration_seconds": 3600.5,
  "monitoring": {"drift_detected": false, "severity_level": "none"}
}
```

### Inference Result (추가 필드)

```json
{
  "output_path": "./output/predictions.parquet",
  "task": "text_generation",
  "monitoring": {"drift_detected": true, "severity_level": "watch", "alerts": [...]}
}
```

> 전체 JSON 스키마는 `mdp/cli/schemas.py`의 Pydantic 모델 참조.

---

## Custom Model Plugin

별도 레지스트리 없이 Python 파일 생성만으로 커스텀 모델을 사용할 수 있다:

```python
# my_models/custom.py
from mdp.models.base import BaseModel

class MyModel(BaseModel):
    def forward(self, batch: dict) -> dict:
        ...  # 최소 "logits" 키 반환

    def training_step(self, batch: dict) -> torch.Tensor:
        ...  # loss 섹션 생략 시 호출됨

    def validation_step(self, batch: dict) -> dict[str, float]:
        ...  # 메트릭 이름→값 dict

    # 선택:
    def generate(self, batch, **kwargs) -> dict: ...
    def configure_optimizers(self) -> dict | None: ...
```

Recipe에서 `model.class_path: my_models.custom.MyModel` 지정. `PYTHONPATH`에 프로젝트 루트 포함 필요.
