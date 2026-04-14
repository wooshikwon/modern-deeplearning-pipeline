# MDP -- Modern Deeplearning Pipeline

YAML 설정으로 딥러닝 모델의 학습, 추론, 서빙을 수행하는 CLI 도구.

## Commands

| Command | Purpose |
|---------|---------|
| `mdp init <name> --task <task> --model <model>` | 프로젝트 스캐폴딩 + Recipe 템플릿 생성 |
| `mdp train -r recipe.yaml -c config.yaml` | SFT 학습. `--callbacks <yaml>` 으로 Recipe 콜백 override 가능 |
| `mdp rl-train -r rl-recipe.yaml -c config.yaml` | RL alignment 학습. 내장 알고리즘: DPO, GRPO, PPO. `_component_` 패턴으로 외부 알고리즘 주입 가능. `--callbacks <yaml>` 지원 |
| `mdp inference` | 배치 추론 + 평가. 모델 소스 3택: `--run-id`, `--model-dir`, `--pretrained <uri>`. 상세 플래그는 아래 참조 |
| `mdp generate` | autoregressive 생성. 모델 소스 3택 동일. 상세 플래그는 아래 참조 |
| `mdp estimate -r recipe.yaml` | GPU 메모리 추정 + 전략 추천 |
| `mdp export --run-id <id> -o <dir>` | adapter merge + 서빙용 패키징. `--checkpoint`로 로컬 체크포인트도 가능 |
| `mdp serve --run-id <id>` | REST API 서빙. `--model-dir`, `--device-map`, `--max-memory` 지원 |
| `mdp list models\|tasks\|callbacks\|strategies` | 카탈로그 조회 |

모든 명령은 `--format json` 옵션을 지원한다. `mdp train`/`rl-train`/`inference`/`generate`는 `--override KEY=VALUE` 옵션을 공통으로 지원하여 Recipe/Config의 필드를 런타임에 덮어쓸 수 있다 (예: `--override training.epochs=0.1 --override data.dataloader.batch_size=8`). 각 명령의 상세 인자는 `mdp <command> --help`로 확인.

### inference 플래그

```
--run-id <id>              MLflow run ID (학습된 모델)
--model-dir <path>         로컬 모델 디렉토리 (mdp export 결과)
--pretrained <uri>         사전학습 모델 URI (hf://, timm://, ultralytics://, local://)
--tokenizer <name>         토크나이저 명시 (--pretrained 시 자동 추론, 명시 가능)
--data <path>              추론 대상 데이터 (HF Hub 이름 또는 로컬 경로, 필수)
--fields role=col ...      필드 매핑 오버라이드 (예: image=img label=class)
--metrics Metric ...       평가 metric (예: Accuracy F1Score)
--callbacks <yaml>         추론 콜백 YAML 파일
--save-output              콜백 전용 모드에서도 DefaultOutputCallback으로 결과 저장
--output-format fmt        결과 포맷: parquet (기본) | csv | jsonl
--output-dir <path>        결과 저장 디렉토리 (기본 ./output)
--device-map <strategy>    multi-GPU 분산 배치: auto | balanced | sequential
--dtype <type>             모델 로딩 dtype: float32 | float16 | bfloat16
--trust-remote-code        HF 모델의 remote code 신뢰
--attn-impl <impl>         어텐션 구현: flash_attention_2 | sdpa | eager
--batch-size N             pretrained 추론 배치 크기 (기본 32)
--max-length N             토큰화 최대 길이 (기본 512)
```

### generate 플래그

```
--run-id / --model-dir / --pretrained   모델 소스 (inference와 동일)
--tokenizer <name>         토크나이저 명시
--prompts <jsonl>          프롬프트 JSONL 파일 (필수)
--prompt-field <name>      JSONL에서 프롬프트 텍스트 필드명 (기본 "prompt")
-o, --output <path>        출력 JSONL 경로 (기본 ./generated.jsonl)
--max-new-tokens N         생성 최대 토큰 수 (기본 256)
--temperature F            샘플링 temperature (기본 1.0)
--top-p F                  nucleus sampling p (기본 1.0)
--top-k N                  top-k sampling (기본 50)
--do-sample                샘플링 사용 여부
--num-samples N            프롬프트당 생성 횟수 (기본 1)
--batch-size N             배치 크기 (기본 1)
--callbacks <yaml>         추론 콜백 YAML 파일
--device-map <strategy>    multi-GPU 분산 배치
--dtype / --trust-remote-code / --attn-impl   모델 로딩 옵션 (inference와 동일)
```

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
# 로컬 export 모델
mdp inference --model-dir ./exported-model --data test.jsonl
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
        target = dict(model.named_modules())["model.layers.20"]
        self._handle = target.register_forward_hook(self._capture)

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
  _component_: str              # alias (AutoModelForCausalLM 등) 또는 풀 경로
  pretrained: str               # URI (hf://, timm://, ultralytics://, local://). 없으면 직접 인스턴스화
  torch_dtype: str              # float32 | float16 | bfloat16 (선택)
  attn_implementation: str      # eager | sdpa | flash_attention_2 (선택)
  ...                           # 나머지는 flat kwargs로 모델 생성자에 전달

head:                           # _component_ 패턴 — 출력 head 교체 (AutoModelFor* 사용 시 생략 가능)
  _component_: str              # Head alias (ClassificationHead, CausalLMHead 등)
  slot: str                     # semantic head slot (head.cls, head.lm 등). family별 실제 속성명으로 자동 번역
  _target_attr: str             # (하위 호환) 모델에서 교체할 속성명 직접 지정 (예: head, classifier, fc)
  ...                           # Head 생성자 인자 (num_classes, hidden_dim 등)
  # slot과 _target_attr 중 하나만 사용. slot은 semantic, _target_attr은 raw. 둘 다 없으면 head 교체 없음.

adapter:                        # _component_ 패턴 — 파라미터 효율 학습 (생략 = full fine-tuning)
  _component_: str              # LoRA | QLoRA | PrefixTuning 또는 풀 경로
  r: int                        # LoRA rank 또는 prefix 토큰 수
  alpha: int                    # LoRA alpha
  dropout: float                # LoRA dropout
  target: list | str            # semantic 적용 대상 (attn.q, mlp.gate 등). family별 실제 모듈명으로 자동 번역
  target_modules: list | str    # (하위 호환) 적용 대상 모듈 raw name 직접 지정 (기본 "all-linear" — PEFT 표준)
  quantization:                 # QLoRA 전용
    bits: 4 | 8
  save: list                    # semantic freeze 제외 모듈 (head.lm, embed.token 등). 실제 이름으로 자동 번역
  modules_to_save: list         # (하위 호환) freeze하지 않을 모듈 raw name 직접 지정 (예: [lm_head, embed_tokens])
  # target과 target_modules는 동시 지정 금지 (ValueError). save와 modules_to_save도 마찬가지.
  # 둘 다 없으면 PEFT 자동 매핑에 위임 (target_modules=None).

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
    batch_size: int             # 기본 32
    num_workers: int            # 기본 4
    drop_last: bool             # 기본 true
    pin_memory: bool            # 기본 true
    persistent_workers: bool    # 기본 true (num_workers=0이면 자동 false)
    prefetch_factor: int        # 기본 2 (num_workers=0이면 자동 null)

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

optimizer:                      # _component_ 패턴
  _component_: str              # AdamW, Adam, SGD 또는 풀 경로
  lr: float
  ...

scheduler:                      # _component_ 패턴 (선택)
  _component_: str              # CosineAnnealingLR, StepLR, LinearLR 등
  warmup_steps: int             # 절대 warmup 스텝 (선택)
  warmup_ratio: float           # 비율 warmup (warmup_steps와 상호 배타)
  interval: str                 # "step" (기본) 또는 "epoch"
  ...

loss:                           # _component_ 패턴 (선택 — 생략 시 model.training_step() 사용)
callbacks: [...]                # _component_ 패턴 리스트

evaluation:
  metrics: [...]                # metric 이름 문자열 리스트 (Accuracy, F1Score 등)

generation:                     # 서빙/추론 전용 (rl.generation과 독립)
  max_new_tokens: int           # 기본 256
  temperature: float            # 기본 1.0
  top_p: float                  # 기본 1.0
  top_k: int                    # 기본 50
  do_sample: bool               # 기본 true
  num_beams: int                # 기본 1
  repetition_penalty: float     # 기본 1.0

monitoring:                     # 데이터 분포 모니터링
  enabled: bool                 # 기본 false
  baseline: {}
  drift: {}

# RL 전용 (mdp rl-train) — rl: 키 아래에 중첩
rl:
  algorithm:                    # _component_ 패턴. 내장: DPO, GRPO, PPO. 풀 경로로 외부 알고리즘 주입 가능
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
    value: {...}                # PPO 또는 외부 알고리즘용 (frozen)
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
environment:                    # 환경 메타데이터
  name: str                     # 기본 "local"

compute:
  target: str                   # local (기본) — mdp는 로컬/torchrun 실행만 지원
  gpus: int | "auto"
  distributed:                  # 분산 전략 (멀티 GPU 시 필수)
    strategy: str | dict        # ddp | fsdp | deepspeed_zero3 | auto | {_component_: FSDPStrategy, ...}
    accelerators: str           # (선택) "A100" 등 — estimate 커맨드의 VRAM 추정용
    moe:                        # MoE Expert Parallelism
      enabled: bool
      ep_size: int              # Expert Parallel degree

# 원격·클라우드 오케스트레이션(SSH job 제출, SkyPilot 런칭 등)은 mdp의 책임이 아니다.
# 사용자가 이미 실행 환경(로컬 머신 / SSH로 접속한 원격 서버 / 클라우드 컨테이너) 안에 있다고
# 가정한다. 원격 실행을 자동화하려면 SkyPilot, Ray, SLURM, K8s 등 전용 오케스트레이터를 쓴다.

mlflow:
  tracking_uri: str             # 기본 ./mlruns
  experiment_name: str

storage:
  checkpoint_dir: str           # 기본 ./checkpoints
  checkpoint_every_n_steps: int # 체크포인트 저장 간격 (선택)
  output_dir: str               # 기본 ./outputs

serving:                        # mdp serve 전용
  backend: str                  # torchserve | vllm
  model_repository: str
  max_batch_size: int           # 기본 1
  instance_count: int           # 기본 1
  batch_window_ms: float        # 기본 50.0ms
  device_map: str               # auto | balanced | sequential
  max_memory: {gpu_id: size}    # GPU별 메모리 한도

job:
  name: str
  resume: str                   # auto (기본) | never | always
  max_retries: int              # 기본 0
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
- 순수 설정값 묶음: `training`, `generation`, `monitoring`, `data.dataloader`, `metadata`, `evaluation`

### 모델 로딩 규칙 (`_component_` + `pretrained` 조합)

Factory는 `_component_` 유무를 먼저 확인하고, duck typing으로 HF 클래스와 커스텀 클래스를 구분한다:

| `_component_` | `pretrained` | 동작 |
|:-:|:-:|---|
| 없음 | 있음 | **PretrainedResolver가 클래스 추론**: `config.architectures[0]`에서 HF 클래스 결정 |
| 있음 (`from_pretrained` 보유) | 있음 | **HF 프로토콜**: `klass.from_pretrained(identifier, **kwargs)` |
| 있음 (`from_pretrained` 없음) | 있음 | **생성자 호출**: `klass(pretrained=uri, **kwargs)` — 커스텀 BaseModel용 |
| 있음 | 없음 | **생성자 호출**: `klass(**kwargs)` — 랜덤 초기화 |

커스텀 모델(`from_pretrained` 없음)은 `pretrained` 유무와 무관하게 **BaseModel 상속 필수**.

내장 alias 목록은 `mdp list callbacks`, `mdp list strategies` 등으로 조회.

> **DualEncoderHead 참고**: CLIP/SigLIP 학습 시 모델의 `training_step()`에서 `self.head.forward_pair(image_features, text_features)`를 직접 호출해야 한다. `forward()`는 추론 시 image projection만 수행한다.

### Semantic Module Routing

`target`, `slot`, `save` 필드는 모델 family에 독립적인 의미적 이름(dot-path)을 사용한다. Factory가 모델 로딩 후 `family_routing` 모듈을 통해 실제 모듈명으로 자동 번역하므로, 동일한 Recipe로 다른 family의 모델을 사용할 수 있다.

**사용 예시**:

```yaml
# Semantic (권장) — family에 독립적
adapter:
  _component_: LoRA
  target: [attn.q, attn.v]       # Llama → [q_proj, v_proj], BERT → [query, value], T5 → [q, v]

# Wildcard 축약
adapter:
  target: [attn.*, mlp.*]        # family의 모든 attention + MLP 모듈

# Escape hatch (하위 호환) — raw name 직접
adapter:
  target_modules: [q_proj, v_proj]  # 기존 방식 그대로 작동
```

**Semantic 네임스페이스**:

| Prefix | 이름 | 설명 |
|--------|------|------|
| `attn.` | `q`, `k`, `v`, `o`, `qkv` | Attention projections |
| `mlp.` | `gate`, `up`, `down`, `fc1`, `fc2`, `gate_up` | MLP/FFN layers |
| `head.` | `cls`, `lm`, `det`, `seg` | 출력 head |
| `embed.` | `token`, `pos` | Embedding layers |
| `conv.` | `dw` | Convolution layers |

**Family별 번역 예시**:

| Semantic | Llama | BERT | GPT-2 | ViT(HF) | ViT-timm | ConvNeXt |
|----------|-------|------|-------|---------|----------|----------|
| `attn.q` | q_proj | query | - | query | - | - |
| `attn.qkv` | - | - | c_attn | - | qkv | - |
| `attn.o` | o_proj | attention.output.dense | attn.c_proj | attention.output.dense | proj | - |
| `mlp.fc1` | - | intermediate.dense | c_fc | intermediate.dense | fc1 | fc1 |
| `mlp.fc2` | - | *미지원* | mlp.c_proj | *미지원* | fc2 | fc2 |
| `head.cls` | - | classifier | - | classifier | head | head |
| `head.lm` | lm_head | - | lm_head | - | - | - |
| `conv.dw` | - | - | - | - | - | conv_dw |

> **ViT(HF) vs ViT-timm**: HF ViT(`config.model_type="vit"`, family=`vit`)는 BERT-style 모듈명을 사용한다. timm ViT(family=`vit_timm`)는 timm 고유 모듈명을 사용한다. HF Swin(family=`swin`)도 BERT-style이며, timm Swin은 `swin_timm` family로 분리되어 있다.

> **BERT-style `mlp.fc2` 미지원**: BERT/ViT(HF)/Swin(HF) 및 이들의 alias(roberta, dinov2, segformer)에서 `mlp.fc2`는 semantic target으로 사용할 수 없다. PEFT의 suffix 매칭에서 `"output.dense"`가 `"attention.output.dense"`까지 매칭하여 의도하지 않은 attention 모듈에도 LoRA가 적용되기 때문이다. MLP output에 LoRA를 적용하려면 `target_modules: [output.dense]`로 raw name을 직접 지정하되, attention output도 함께 매칭됨을 인지해야 한다.

dot(`.`)이 없는 이름은 raw name으로 취급되어 번역 없이 PEFT에 직접 전달된다. `target`과 `target_modules`를 동시에 지정하면 ValueError로 차단된다.

---

## Validation Rules

학습 시작 전에 3단 검증(Catalog → Business → Compat)이 모든 에러를 수집하여 한 번에 보고한다.

### 검증 스코프 (어느 명령에서 어느 검증이 도는가)

| 명령 | 실행되는 검증 |
|------|--------------|
| `mdp train`, `mdp rl-train` | **전체** — head_task, adapter, rl_models, component_imports, **data_components, task_fields, streaming_distributed**, GPU/strategy 호환성 |
| `mdp estimate`, `mdp inference --run-id`/`--model-dir` | **모델 관련만** — head_task, adapter, rl_models, component_imports |
| `mdp inference --pretrained`, `mdp generate --pretrained` | Recipe 자체가 없으므로 검증 스킵. 런타임 에러로만 잡힘 |

따라서 `mdp estimate`가 통과해도 `data.dataset._component_` 누락 같은 데이터 측 결함은 잡히지 않는다 — 학습 시작 시점에야 발견. **검증 누락을 미리 잡으려면 작은 max_steps로 `mdp train`을 한 번 돌려본다**.

### Task-Head Compatibility

| Task | Allowed Head | AutoModel alias (head 생략) |
|------|-------------|----------------------|
| image_classification | ClassificationHead | (alias 없음 — 풀 경로 사용) |
| object_detection | DetectionHead | (alias 없음) |
| semantic_segmentation | SegmentationHead | (alias 없음) |
| text_classification | ClassificationHead | AutoModelForSequenceClassification |
| token_classification | TokenClassificationHead, ClassificationHead | AutoModelForTokenClassification |
| text_generation | CausalLMHead | AutoModelForCausalLM |
| seq2seq | Seq2SeqLMHead | AutoModelForSeq2SeqLM |
| image_generation | (head 생략 권장) | 모델 내장 |
| feature_extraction | ClassificationHead, DualEncoderHead | (head 생략 가능) |

> vision 태스크(`image_classification`, `object_detection`, `semantic_segmentation`)에는 AutoModel alias가 `aliases.yaml`에 없다. HF 풀 경로(`transformers.AutoModelForImageClassification`)를 사용하거나 `timm://` 프로토콜을 사용한다.
>
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

## Logging & Warning Behavior

### 서드파티 경고 억제

`mdp/__init__.py`에서 의존성 라이브러리의 노이즈를 억제한다:
- `FutureWarning` 전역 필터링 (transformers/datasets의 deprecated API 예고)
- `torch` beta feature `UserWarning` 필터링
- `transformers`, `datasets`, `accelerate`, `bitsandbytes` 로거를 WARNING 레벨로 설정

mdp 자체 로그(`logger = logging.getLogger(__name__)`)는 영향받지 않는다.

### MLflow 에러 처리

- 1회성 호출(`_log_mlflow_params`, `_log_mlflow_summary`): 실패 시 `logger.warning`
- 매 step 호출(`_mlflow_log_metric`): 실패 시 `logger.debug` (MLflow 장애 시 경고 폭주 방지)

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

인라인 예시는 유지하지 않는다. 대신 **실제로 파싱·테스트 검증된 레시피**가 `tests/fixtures/recipes/`에 있다. 에이전트는 이 중 가장 가까운 것을 복사해서 필요한 필드만 수정하거나, `mdp init`이 생성하는 템플릿의 `???` 필드를 채우면 된다.

| 파일 | 태스크 | 특징 |
|------|--------|------|
| `tiny-vision-e2e.yaml` | image_classification | 커스텀 `BaseModel` 최소 예시 (테스트용) |
| `vit-lora-cifar10.yaml` | image_classification | ViT + LoRA + augmentation pipeline |
| `yolo-detection-custom.yaml` | object_detection | `ultralytics://` 프로토콜 + 커스텀 모델 |
| `clip-finetune-custom.yaml` | feature_extraction | CLIP multimodal + DualEncoderHead |
| `gpt2-finetune-text.yaml` | text_generation | LLM SFT 최소 예시 |
| `qwen25-qlora-instruct.yaml` | text_generation | 대형 LLM + QLoRA + bf16 |
| `gpt2-dpo-preference.yaml` | text_generation (RL) | DPO preference 학습 (chosen/rejected) |

이 파일들은 `tests/unit/test_recipe_fixtures.py`에서 `SettingsFactory.for_estimation`으로 파싱·검증되므로 스키마 drift가 발생하면 즉시 깨진다 — 항상 최신 상태가 보장되는 단일 소스.

Config 예시는 단순하다:

```yaml
# 로컬 단일 GPU
compute: {gpus: 1}
mlflow: {experiment_name: my-exp}
storage: {checkpoint_dir: ./checkpoints, output_dir: ./outputs}

# 멀티 GPU + DeepSpeed ZeRO-3
compute:
  gpus: auto
  distributed: {strategy: deepspeed_zero3}
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
    _block_classes = None  # 필수 선언 (아래 참조)

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

### `_block_classes` 필수 선언

`BaseModel` 서브클래스는 `_block_classes` 를 반드시 선언해야 한다 (`forward`, `training_step`, `validation_step`과 동등한 수준의 필수 계약). 미선언 시 **클래스 정의 시점**에 `TypeError`가 발생한다.

`_block_classes`는 모델의 반복 블록 클래스 이름의 `set[str]`이다. FSDP wrap policy, gradient checkpointing 등에서 모델을 효율적인 단위로 분할할 때 사용된다.

| 모델 유형 | `_block_classes` 선언 예시 |
|-----------|--------------------------|
| LLM (decoder-only) | `_block_classes = {"LlamaDecoderLayer"}` |
| VLM (복수 블록) | `_block_classes = {"LlamaDecoderLayer", "CLIPEncoderLayer"}` |
| CNN (ResNet) | `_block_classes = {"Bottleneck"}` |
| 단순 MLP / 반복 블록 없음 | `_block_classes = None` |

HF backbone을 감싸는 커스텀 모델에서는 `_inherit_block_classes()` helper를 `__init__` 마지막에 호출하면 backbone의 블록 클래스 정보를 자동으로 상속받는다 (HF `_no_split_modules` → `_block_classes` 변환).

### 커스텀 모델 + pretrained backbone

커스텀 BaseModel이 외부 pretrained 모델을 backbone으로 사용할 수 있다. `pretrained`는 생성자 인자로 전달된다:

```python
class CriticValueModel(BaseModel):
    _block_classes = None  # placeholder (아래 _inherit_block_classes가 채움)

    def __init__(self, pretrained: str, value_head_dropout: float = 0.2, **kwargs):
        super().__init__()
        model_name = pretrained.removeprefix("hf://")
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_dim = self.backbone.config.hidden_size  # backbone에 적응
        self.value_head = ValueHead(hidden_dim, dropout=value_head_dropout)
        self._inherit_block_classes()
        # → _block_classes = {"LlamaDecoderLayer"} (backbone에서 상속)
```

```yaml
model:
  _component_: my_models.CriticValueModel
  pretrained: hf://meta-llama/Meta-Llama-3-8B
  value_head_dropout: 0.2
```

Factory는 `CriticValueModel`에 `from_pretrained`가 없으므로 **생성자 호출**: `CriticValueModel(pretrained="hf://...", value_head_dropout=0.2)`. backbone의 아키텍처(hidden_size 등)는 불변이며, 커스텀 클래스는 `backbone.config`를 읽어 적응해야 한다.
