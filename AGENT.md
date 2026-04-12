# MDP -- Modern Deeplearning Pipeline

YAML 설정으로 딥러닝 모델의 학습, 추론, 서빙을 수행하는 CLI 도구.

## Commands

| Command | Purpose |
|---------|---------|
| `mdp init <name> --task <task> --model <model>` | 프로젝트 스캐폴딩 + Recipe 템플릿 생성 |
| `mdp train -r recipe.yaml -c config.yaml` | SFT 학습. `--callbacks <yaml>` 으로 Recipe 콜백 override 가능 |
| `mdp rl-train -r rl-recipe.yaml -c config.yaml` | RL alignment 학습 (DPO, weighted-NTP, GRPO, PPO). `--callbacks <yaml>` 지원 |
| `mdp inference --run-id <id> --data <path>` | 배치 추론 + 평가. `--pretrained <uri>`, `--dtype`, `--trust-remote-code`, `--attn-impl`, `--callbacks <yaml>`, `--save-output`, `--batch-size` (기본 32), `--max-length` (기본 512) |
| `mdp generate --run-id <id> --prompts <jsonl> -o <out>` | autoregressive 생성. `--pretrained <uri>`, `--dtype`, `--trust-remote-code`, `--attn-impl`, `--callbacks <yaml>`, `--max-new-tokens` (기본 256), `--temperature` (기본 1.0), `--top-p` (기본 1.0), `--top-k` (기본 50), `--do-sample`, `--batch-size` (기본 1) |
| `mdp estimate -r recipe.yaml` | GPU 메모리 추정 + 전략 추천 |
| `mdp export --run-id <id> --output <dir>` | adapter merge + 서빙용 패키징 (`--checkpoint`로 로컬 체크포인트도 가능) |
| `mdp serve --run-id <id>` | REST API 서빙 |
| `mdp list models\|tasks\|callbacks\|strategies` | 카탈로그 조회 |

모든 명령은 `--format json` 옵션을 지원한다. `mdp train`/`rl-train`/`inference`/`generate`는 `--override KEY=VALUE` 옵션을 공통으로 지원하여 Recipe/Config의 필드를 런타임에 덮어쓸 수 있다 (예: `--override training.epochs=0.1 --override data.dataloader.batch_size=8`). 각 명령의 상세 인자는 `mdp <command> --help`로 확인.

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

`mdp init`이 생성하는 Recipe에서 `???`로 표시된 필드만 채우면 된다: `data.dataset.source`, `head.num_classes`(해당 시), `metadata.*`

---

## Two-File System + Callbacks

- **Recipe** (실험 정의): 무엇을 학습할지 — 모델, 데이터, 하이퍼파라미터
- **Config** (인프라 설정): 어디서 실행할지 — GPU, 분산 전략, MLflow, 체크포인트 경로
- **Callbacks** (부가 동작): 체크포인트, 분석, 스티어링, 로깅 — `--callbacks <yaml>` 로 별도 파일 제공

동일 Recipe를 다른 Config로 실행하면 같은 실험을 다른 환경에서 재현할 수 있다. 콜백도 독립 파일로 분리하면 동일 실험에 다른 분석 동작을 조합할 수 있다.

### `--pretrained` 모델 소스

`inference`와 `generate`에서 `--run-id`, `--model-dir`, `--pretrained` 중 하나를 지정한다 (상호 배타):

```bash
# 학습된 모델 (기존 방식)
mdp inference --run-id abc123 --data test.jsonl
# 오픈소스 모델 직접 로드 (Recipe 없이)
mdp inference --pretrained hf://meta-llama/Meta-Llama-3-8B --data prompts.jsonl
# 토크나이저 명시 (자동 추론이 안 될 때)
mdp generate --pretrained hf://model --prompts p.jsonl --tokenizer other-tokenizer
# dtype/attention/remote-code 지정 (BFloat16 모델의 numpy 변환 에러 방지)
mdp inference --pretrained hf://Qwen/Qwen2.5-7B --data test.jsonl \
  --dtype float32 --trust-remote-code --attn-impl sdpa --device-map auto
```

### `--callbacks` 파일

train, rl-train, inference, generate 4개 커맨드에서 `--callbacks <yaml>` 옵션을 지원한다. 파일 형식은 Recipe의 `callbacks:` 섹션과 동일:

```yaml
# analysis.yaml
- _component_: my_project.HiddenStateLogger
  layers: [20, 40, 60]
- _component_: ModelCheckpoint
  monitor: val_loss
```

**병합 규칙** (학습 커맨드에서):
- Recipe `callbacks:` 있고 `--callbacks` 없음 → Recipe 콜백 사용
- `--callbacks` 있음 → `--callbacks`가 override (CLI 우선)
- 둘 다 없음 → 콜백 없음

### BaseInferenceCallback

추론 콜백은 `BaseInferenceCallback`을 상속하여 3개 hook을 구현한다. 추론 루프는 hidden state/attention을 직접 다루지 않으며, 모든 내부 접근은 콜백이 `setup`에서 등록하는 `register_forward_hook`을 통해 이루어진다:

```python
from mdp.callbacks.base import BaseInferenceCallback

class MyAnalysisCallback(BaseInferenceCallback):
    def setup(self, model, tokenizer=None, **kwargs):
        """추론 시작 전. 모델에 forward hook 등록, 버퍼 준비."""
        self._handle = model.layer[20].register_forward_hook(self._capture)

    def on_batch(self, batch_idx, batch, outputs, **kwargs):
        """매 배치 forward 후. hook이 캡처한 활성화를 처리."""

    def teardown(self, **kwargs):
        """추론 완료 후. 누적 결과 저장, hook 해제."""
        self._handle.remove()
```

```bash
# 오픈소스 모델 + 분석 콜백
mdp inference --pretrained hf://meta-llama/Meta-Llama-3-8B \
  --data prompts.jsonl --callbacks analysis.yaml
```

---

## Recipe YAML — 구조

```
name: str (실험 이름)
task: str (9종: image_classification, object_detection, semantic_segmentation,
           text_classification, token_classification, text_generation,
           seq2seq, image_generation, feature_extraction)

model:                          # _component_ 패턴 — 모델 클래스 + pretrained 가중치
  _component_: str              # alias (AutoModelForCausalLM, AutoModel 등) 또는 풀 경로
  pretrained: str               # URI (hf://, timm://, ultralytics://, local://). 없으면 직접 인스턴스화
  torch_dtype: str              # float32 | float16 | bfloat16 (선택)
  attn_implementation: str      # eager | sdpa | flash_attention_2 (선택)
  ...                           # 나머지는 flat kwargs로 모델 생성자에 전달

head:                           # _component_ 패턴 — 출력 head 교체 (AutoModelFor* 사용 시 생략 가능)
  _component_: str              # Head alias (ClassificationHead, CausalLMHead 등)
  _target_attr: str             # 모델에서 교체할 속성명 (예: head, classifier, fc)
  ...                           # Head 생성자 인자 (num_classes, hidden_dim 등)

adapter:                        # _component_ 패턴 — 파라미터 효율 학습 (생략 = full fine-tuning)
  _component_: str              # LoRA | QLoRA | PrefixTuning 또는 풀 경로
  r: int                        # LoRA rank 또는 prefix 토큰 수
  alpha: int                    # LoRA alpha
  dropout: float                # LoRA dropout
  target_modules: list | str    # 적용 대상 모듈 (기본 "all_linear")
  quantization:                 # QLoRA 전용
    bits: 4 | 8
  modules_to_save: list         # freeze하지 않을 모듈 (예: [lm_head, embed_tokens])

data:                           # 데이터 파이프라인
  dataset:                      # _component_ 패턴 — 학습 Dataset
    _component_: str            # HuggingFaceDataset | ImageClassificationDataset 또는 풀 경로
    source: str                 # HF Hub 이름 또는 로컬 파일/디렉토리
    split: str                  # 기본 "train"
    fields: {role: column}      # 역할→컬럼 매핑 (선택)
    tokenizer: str              # AutoTokenizer pretrained 이름 (언어 태스크)
    max_length: int             # 토큰화 최대 길이
    augmentation: list          # [{type, params}, ...] (비전 전용)
    ...                         # Dataset별 추가 kwargs (subset, streaming, data_files 등)
  val_dataset:                  # _component_ 패턴 — 검증 Dataset (선택; 생략 시 val 비활성)
    _component_: str
    ...
  collator:                     # _component_ 패턴 — 배치 조립기
    _component_: str            # CausalLMCollator | PreferenceCollator | Seq2SeqCollator |
                                #   ClassificationCollator | TokenClassificationCollator | VisionCollator
    tokenizer: str              # 언어 Collator는 tokenizer 필수
    max_length: int
  dataloader:                   # 순수 설정값 (DataLoader kwargs)
    batch_size: int
    num_workers: int
    drop_last: bool

training:                       # 학습 루프 설정
  epochs: float                 # epochs 또는 max_steps 중 하나 필수 (float 허용)
  max_steps: int
  precision: str                # fp32 | fp16 | bf16
  gradient_accumulation_steps: int
  gradient_clip_max_norm: float
  gradient_checkpointing: bool
  val_check_interval: int | float
  val_check_unit: str           # "epoch" (기본) 또는 "step"
  compile: bool | str
  # strategy는 Config.compute.distributed에서 관리 (아래 Config 섹션 참조)

optimizer:                      # _component_ 패턴
  _component_: str              # AdamW, Adam, SGD 또는 풀 경로
  lr: float
  ...

scheduler:                      # _component_ 패턴 (선택)
loss:                           # _component_ 패턴 (선택 — 생략 시 model.training_step() 사용)
callbacks: [...]                # _component_ 패턴 리스트
evaluation:
  metrics: [...]                # metric 이름 문자열 리스트

# RL 전용 (mdp rl-train) — rl: 키 아래에 중첩
rl:
  algorithm:                    # _component_ 패턴 (DPO, GRPO, PPO, ...)
    _component_: str
    ...
  models:                       # 역할별 model dict. 각 역할이 독립된 _component_ dict
    policy:
      _component_: str          # AutoModelForCausalLM 등
      pretrained: str
      adapter: {_component_: LoRA, ...}        # sub-component
      optimizer: {_component_: AdamW, lr: ...} # sub-component (없으면 frozen)
      scheduler: {_component_: ..., ...}
      head: {_component_: ..., ...}            # sub-component
    reference:                  # frozen (optimizer 생략)
      _component_: str
      pretrained: str
    value: {...}                # PPO 전용
    reward: {...}               # frozen
  generation:                   # GRPO/PPO 전용
    max_new_tokens: int
    temperature: float
    group_size: int             # GRPO K개 응답

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
  distributed:                  # 분산 전략 (멀티 GPU 시 필수)
    strategy: str | dict        # ddp | fsdp | deepspeed_zero3 | auto | {_component_: FSDPStrategy, ...}
    # strategy 외 키(offload_optimizer 등)는 strategy 인스턴스에 kwargs로 전달
    # moe: {enabled: true, ep_size: N} — MoE Expert Parallelism 설정

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

Recipe의 모든 pluggable component(`model`, `adapter`, `head`, `optimizer`, `scheduler`, `loss`, `callback`, `data.dataset`, `data.collator`, `rl.algorithm`, `rl.models.*`)가 동일한 문법을 따른다. Config의 `compute.distributed.strategy`도 dict 형태 시 동일한 `_component_` 문법을 지원한다:

```yaml
optimizer:
  _component_: AdamW        # 내장 alias → torch.optim.AdamW
  lr: 0.001

optimizer:
  _component_: my_package.MyOptimizer  # 커스텀 (점 포함 = 풀 경로)
  lr: 0.001
```

`_component_` 값에 점(`.`)이 없으면 `mdp/aliases.yaml`에서 풀 경로로 치환되고, 점이 있으면 그대로 import된다. 따라서 내장이든 외부 패키지든 동일한 문법으로 주입된다.

**`_component_`를 쓰지 않는 영역**:
- 순수 설정값 묶음: `training`(precision/epochs/gradient_* 등), `generation`, `data.dataloader`, `metadata`, `evaluation`
- `pretrained` 특수 키: `model._component_`와 함께 쓰이는 URI 전용 키. Factory가 URI 스킴에 따라 로더를 선택한다

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

- `adapter._component_: QLoRA` → `quantization.bits` 필수, `torch_dtype`는 `bfloat16` 또는 `float16`
- `LoRA`/`QLoRA`/`PrefixTuning` → `r` 필수
- `PrefixTuning`에서 `alpha`, `dropout`, `target_modules`는 자동 무시됨
- `QLoRA` + `head` 조합은 차단됨 (양자화 dtype 불일치)

### Forward Contract

- `loss` 지정 시: `model.forward(batch)` → `dict(logits=...)` 반환 필수
- `loss` 생략 시: `model.training_step(batch)` → `Tensor` (스칼라 loss) 반환 필수

### Precision & Distributed

- `bf16` → Ampere+ GPU (A100, RTX 3090+) 필요
- `flash_attention_2` → Ampere+ GPU 필요
- `config.compute.distributed.strategy: fsdp` + QLoRA → **비호환** (대안: DDP, DeepSpeed ZeRO-3)
- 멀티 GPU 시 `config.compute.distributed.strategy` 필수 (또는 `auto`로 런타임 선택)

### Data Constraints

- `data.dataset._component_` 필수. `data.collator._component_` 필수
- Dataset/Collator 클래스가 자체 `__init__`에서 파라미터 검증(예: `source` 필수, `tokenizer` 필수)을 수행한다 — 프레임워크는 "`_component_` 키가 있는 dict인가"까지만 확인
- `HuggingFaceDataset`의 `streaming: true` + multimodal(vision+language) → Dataset 클래스가 `ValueError`로 차단
- 검증 데이터는 `data.val_dataset` dict를 명시적으로 선언해야 활성화됨 (자동 추론 없음)

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
  _component_: transformers.AutoModel
  pretrained: hf://google/vit-base-patch16-224

head:
  _component_: ClassificationHead
  _target_attr: head
  num_classes: 10
  pooling: cls_token
  dropout: 0.1

adapter:
  _component_: LoRA
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, v_proj]

data:
  dataset:
    _component_: ImageClassificationDataset
    source: cifar10
    split: train
    fields: {image: img, label: label}
    augmentation:
      - {type: RandomResizedCrop, params: {size: [224, 224]}}
      - {type: RandomHorizontalFlip}
      - {type: ToDtype, params: {dtype: float32, scale: true}}
      - {type: Normalize, params: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}}
  val_dataset:
    _component_: ImageClassificationDataset
    source: cifar10
    split: test
    fields: {image: img, label: label}
    augmentation:
      - {type: Resize, params: {size: [256, 256]}}
      - {type: CenterCrop, params: {size: [224, 224]}}
      - {type: ToDtype, params: {dtype: float32, scale: true}}
      - {type: Normalize, params: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}}
  collator:
    _component_: VisionCollator
  dataloader:
    batch_size: 64

training:
  epochs: 30
  precision: bf16
  gradient_clip_max_norm: 1.0
  # strategy는 Config.compute.distributed에서 설정

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
  _component_: AutoModelForCausalLM
  pretrained: hf://Qwen/Qwen2.5-7B
  torch_dtype: bfloat16

adapter:
  _component_: QLoRA
  r: 64
  alpha: 128
  quantization:
    bits: 4
  modules_to_save: [lm_head, embed_tokens]

data:
  dataset:
    _component_: HuggingFaceDataset
    source: HuggingFaceH4/ultrachat_200k
    split: train_sft
    fields: {text: text}
    tokenizer: Qwen/Qwen2.5-7B
    max_length: 2048
  collator:
    _component_: CausalLMCollator
    tokenizer: Qwen/Qwen2.5-7B
    max_length: 2048
  dataloader:
    batch_size: 4

training:
  max_steps: 1000
  precision: bf16
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  # strategy는 Config에서 distributed.strategy: deepspeed_zero3로 설정

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
  "run_id": "abc123...",
  "checkpoint_dir": "...",
  "metrics": {"val_loss": 0.12, "val_accuracy": 0.95},
  "total_epochs": 27,
  "total_steps": 12600,
  "stopped_reason": "early_stopped | completed | max_steps_reached",
  "duration_seconds": 3600.5,
  "monitoring": {"baseline_saved": true, "baseline_path": "mlruns/.../baseline.json"}
}
```

### Inference Result (추가 필드)

```json
{
  "run_id": "abc123...",
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

Recipe에서 `model._component_: my_models.custom.MyModel` 지정. `PYTHONPATH`에 프로젝트 루트 포함 필요.
