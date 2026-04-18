# MDP -- Modern Deeplearning Pipeline

YAML 설정으로 딥러닝 모델의 학습, 추론, 서빙을 수행하는 CLI 도구.

## Commands

| Command | Purpose |
|---------|---------|
| `mdp init <name> --task <task> --model <model>` | 프로젝트 스캐폴딩 + Recipe 템플릿 생성 |
| `mdp train -r recipe.yaml -c config.yaml` | SFT 학습. `--callbacks <yaml>` 으로 콜백 주입 (CLI가 유일한 소스) |
| `mdp rl-train -r rl-recipe.yaml -c config.yaml` | RL alignment 학습. 내장 알고리즘: DPO, GRPO, PPO. `_component_` 패턴으로 외부 알고리즘 주입 가능. `--callbacks <yaml>` 지원 |
| `mdp inference` | 배치 추론 + 평가. 모델 소스 3택: `--run-id`, `--model-dir`, `--pretrained <uri>`. 상세 플래그는 아래 참조 |
| `mdp generate` | autoregressive 생성. 모델 소스 3택 동일. 상세 플래그는 아래 참조 |
| `mdp estimate -r recipe.yaml` | GPU 메모리 추정 + 전략 추천 |
| `mdp export --run-id <id> --output <dir>` | adapter merge + 서빙용 패키징. 소스는 `--run-id` 또는 `--checkpoint <path>` (상호 배타). `--output` 기본값 `./exported-model` |
| `mdp serve --run-id <id>` | REST API 서빙. `--model-dir`, `--device-map`, `--max-memory` 지원 |
| `mdp list models\|tasks\|callbacks\|strategies` | 카탈로그 조회 |

모든 명령은 `--format json` 옵션을 지원한다. `mdp train`/`rl-train`/`inference`/`generate`는 `--override` 옵션을 공통으로 지원하여 Recipe/Config의 필드를 런타임에 덮어쓸 수 있다. 두 가지 문법이 공존한다:

- **KEY=VALUE 형식** (단일 항목 반복): `--override training.epochs=0.1 --override data.dataloader.batch_size=8`
- **JSON dict 형식** (다건 일괄): `--override '{"training.epochs": 0.1, "data.dataloader.batch_size": 8}'`

값은 자동 타입 추론된다 (`null`/`none` → None, `true`/`false` → bool, int → float → JSON(`{...}`, `[...]`) → str). 각 명령의 상세 인자는 `mdp <command> --help`로 확인.

> **멀티 GPU 실행 주의**: `mdp train` / `mdp rl-train`은 `compute.gpus`와 `compute.distributed.strategy` 설정을 읽어 내부적으로 `_torchrun_entry.py`를 통해 멀티 GPU를 직접 관리한다. **절대로 외부 `torchrun`으로 감싸지 말 것** — `torchrun ... mdp rl-train ...` 패턴은 이중 torchrun(`torchrun → MDP → 내부 torchrun`)이 되어 포트 충돌로 실패한다. 올바른 실행: `mdp rl-train -r recipe.yaml -c config.yaml`.

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
--max-new-tokens N         생성 최대 토큰 수
--temperature F            샘플링 temperature
--top-p F                  nucleus sampling p
--top-k N                  top-k sampling
--do-sample                샘플링 사용 여부
--num-samples N            프롬프트당 생성 횟수 (기본 1)
--batch-size N             배치 크기 (기본 1)
--callbacks <yaml>         추론 콜백 YAML 파일
--device-map <strategy>    multi-GPU 분산 배치
--dtype / --trust-remote-code / --attn-impl   모델 로딩 옵션 (inference와 동일)
```

> **생성 파라미터 폴백 순서**: `--max-new-tokens`, `--temperature`, `--top-p`, `--top-k`, `--do-sample`은 CLI 미지정 시 `None`으로 전달되고, Recipe의 `generation:` 섹션 값 → `GenerationSpec` 기본값(`max_new_tokens=256`, `temperature=1.0`, `top_p=1.0`, `top_k=50`, `do_sample=True`) 순으로 폴백한다. `--num-samples`, `--batch-size`는 CLI 레벨에서 각각 1로 기본 적용.

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

- **Recipe** (실험 정의): 무엇을 학습할지 — 모델, 데이터, 하이퍼파라미터. `training.early_stopping`/`training.ema`는 학습 결과에 직접 영향을 주는 동작이므로 Recipe의 1급 필드로 Pydantic 검증을 받는다. Recipe에는 `callbacks:` 필드가 없다
- **Config** (인프라 설정): 어디서 실행할지 — GPU, 분산 전략, MLflow, 체크포인트 경로
- **Callbacks** (부가 동작): 체크포인트, 분석, 스티어링, 로깅 — `--callbacks <yaml>` 로 별도 파일 제공. CLI가 유일한 소스

동일 Recipe를 다른 Config로 실행하면 같은 실험을 다른 환경에서 재현할 수 있다. 콜백도 독립 파일로 분리하면 동일 실험에 다른 분석/개입 동작을 조합할 수 있다.

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

train, rl-train, inference, generate 4개 커맨드에서 `--callbacks <yaml>` 옵션을 지원한다. CLI 파일이 콜백의 유일한 소스이며, Recipe에는 `callbacks:` 필드가 없다:

```yaml
# analysis.yaml
- _component_: my_project.HiddenStateLogger
  layers: [20, 40, 60]
- _component_: ModelCheckpoint
  monitor: val_loss
```

**콜백 소스 규칙**:
- `--callbacks <yaml>` → 파일에서 콜백 로드. 학습 경로에서는 Trainer 생성자에 직접 주입
- `--callbacks` 없음 → 콜백 없음 (EarlyStopping/EMA는 Recipe `training.*` 1급 필드로만 지정)

**Intervention callback 태깅**: `BaseInterventionCallback` (`is_intervention=True`) 서브클래스는 추론 시작 시 `metadata` 프로퍼티가 자동으로 MLflow에 `intervention.{i}.{key}` 태그로 기록된다.

### BaseInferenceCallback / BaseInterventionCallback

추론 콜백은 두 베이스 클래스로 분류된다:

- **`BaseInferenceCallback`**: 읽기 전용 관측. `is_intervention = False`. hidden state 추출, attention 분석 등
- **`BaseInterventionCallback`**: 출력을 바꾸는 개입. `is_intervention = True`. `metadata -> dict` 프로퍼티 구현 필수. 내장: `ResidualAdd`, `LogitBias`

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
# 관측 콜백
mdp inference --pretrained hf://meta-llama/Meta-Llama-3-8B \
  --data prompts.jsonl --callbacks analysis.yaml

# 개입 콜백 (MLflow에 intervention.* tag 자동 기록)
mdp generate --pretrained hf://meta-llama/Meta-Llama-3-8B \
  --prompts prompts.jsonl --callbacks intervention.yaml
```

`mdp list callbacks`로 등록된 콜백의 타입(`[Int]`/`[Obs]`/`[Train]`)을 확인할 수 있다.

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
  early_stopping:               # 학습 조기 종료 (선택). Pydantic 검증.
    monitor: str                # 기본 "val_loss"
    patience: int               # 기본 5
    mode: str                   # "min" (기본) 또는 "max"
    min_delta: float            # 기본 0.0
  ema:                          # 파라미터 지수이동평균 (선택). Pydantic 검증.
    decay: float                # 기본 0.9999
    update_after_step: int      # 기본 0
    update_every: int           # 기본 1

optimizer:                      # _component_ 패턴
  _component_: str              # AdamW, Adam, SGD 또는 풀 경로
  lr: float
  ...

scheduler:                      # _component_ 패턴 (선택)
  _component_: str              # CosineAnnealingLR, StepLR, LinearLR 등
  warmup_steps: int             # 절대 warmup 스텝 (선택)
  warmup_ratio: float           # 비율 warmup (warmup_steps와 상호 배타)
  warmup_start_factor: float    # LinearLR 첫 step 멀티플라이어 (선택, 기본 1e-8)
  warmup_end_factor: float      # LinearLR 마지막 warmup step 멀티플라이어 (선택, 기본 1.0)
  interval: str                 # "step" (기본) 또는 "epoch"
  ...

loss:                           # _component_ 패턴 (선택 — 생략 시 model.training_step() 사용)
# callbacks: 필드는 Recipe에 없다. 콜백은 --callbacks <yaml> CLI로만 주입한다

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
  # --- 실제 동작하는 필드 ---
  max_batch_size: int           # 기본 1. PredictHandler의 동적 배칭 상한
  batch_window_ms: float        # 기본 50.0ms. 동적 배칭 시간 창
  device_map: str               # auto | balanced | sequential (from_pretrained(device_map=))
  max_memory: {gpu_id: size}    # GPU별 메모리 한도 (예: {"0": "24GiB"})
  # --- Reserved (schema·validator는 존재하나 serve.py가 미사용) ---
  backend: str                  # 기본 "torchserve". 현재 mdp serve는 uvicorn+FastAPI 고정.
                                # compat_validator는 backend == "vllm" + 비호환 task를 에러로 잡지만,
                                # 실제 vLLM/TorchServe 백엔드 라우팅은 미구현.
  model_repository: str         # 미사용 (향후 TorchServe 통합 대비 예약)
  instance_count: int           # 기본 1. 미사용 (향후 multi-worker 대비 예약)

> `mdp serve`는 항상 uvicorn + FastAPI로 실행된다. vLLM/TorchServe로 대체 서빙이 필요하면
> `mdp export`로 모델을 패키징한 뒤 해당 런타임에 직접 전달한다.

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

> **Family별 raw name 매핑 표, ViT(HF)↔ViT-timm 구분, BERT-style `mlp.fc2` 미지원 제약** 등의 상세 참조는 `docs/extending.md`의 "Semantic Module Routing 매핑" 섹션을 본다. AGENT.md에는 일상적으로 필요한 네임스페이스까지만 싣는다.

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
- **DDP/FSDP 하에서도 동일 호출 계약이 유지된다.** trainer는 `strategy.unwrap` / `strategy.invoke_custom`을 통해 래핑된 모델의 `training_step` / `validation_step`을 호출하므로, 모델 구현자는 분산 래퍼를 신경 쓰지 않고 plain 메서드로 선언하면 된다. DDP는 autograd hook 기반이라 `.module` 경유 호출만으로 gradient sync가 보존되고, FSDP는 `invoke_custom`이 내부적으로 forward-swap으로 all-gather 훅을 정상 발동시킨다.

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

## Warmup Scheduler Tuning

Recipe의 `scheduler:` 섹션은 `warmup_steps` / `warmup_ratio`로 warmup 길이를 지정하고, 선택적으로 `warmup_start_factor` / `warmup_end_factor`로 warmup 곡선의 시작·끝 `base_lr` 멀티플라이어를 조정한다. 두 factor 모두 **opt-in** — 생략 시 MDP 기본값이 유지돼 기존 Recipe는 수정 없이 동일 동작을 보장한다.

| 필드 | 타입 | 기본값 | 의미 |
|------|------|--------|------|
| `warmup_start_factor` | float (선택) | `1e-8` | warmup step 0의 `base_lr` 멀티플라이어 |
| `warmup_end_factor` | float (선택) | `1.0` | warmup step = `warmup_steps` 시점의 `base_lr` 멀티플라이어 |

유효 범위: `0 < warmup_start_factor <= warmup_end_factor <= 1.0`. 위반 시 Trainer/RLTrainer 초기화에서 `ValueError`. `warmup_start_factor = 0`은 PyTorch `LinearLR` 내부의 `ZeroDivisionError`(issue #86454)를 유발하므로 차단된다 — "0에서 시작" 관례를 원하면 `1e-8` 같은 양의 매우 작은 값을 사용한다.

### 업계 관례 요약

| 프레임워크 | warmup 첫 step 멀티플라이어 |
|---|---|
| PyTorch `LinearLR` | `1/3` (≈0.333, generic default) |
| HuggingFace `get_linear_schedule_with_warmup` | **0** (`current_step / max(1, num_warmup_steps)`, step 0 = 0) |
| MosaicML Composer `LinearWithWarmupScheduler` | **0** |
| DeepSpeed `WarmupLR` | **0** (`warmup_min_lr=0.0`) |
| LLaMA-3 SFT 실전 스택 (LLaMA-Factory, Axolotl, WaveCoder, OpenCodeInterpreter) | **0** (HF 경유) |
| **MDP 기본값** | **1e-8** (HF 관례와 수치 동등한 PyTorch 호환 hack) |

MDP의 `1e-8`은 bf16 mantissa 아래 수치이므로 HF 경로와 실효적으로 동일한 "거의 0에서 시작"을 구현한다. 업계 관례에서 벗어난 값(예: `start_factor=0.1`로 base_lr의 10%에서 시작)이 필요한 특수 레시피(소규모 `warmup_steps` 또는 LoRA adapter zero-init에 따른 초기 gradient 완충 요구 등)에서만 Recipe로 오버라이드한다.

### Recipe 예시

```yaml
scheduler:
  _component_: CosineAnnealingLR
  T_max: 1
  warmup_ratio: 0.03
  warmup_start_factor: 0.1    # base_lr × 0.1에서 시작 (기본 1e-8 대신)
  warmup_end_factor: 1.0      # warmup 끝에서 base_lr 도달 (기본값과 동일)
```

RLTrainer는 `recipe.rl.models.{role}.scheduler`가 모델별로 독립 dict이므로 policy와 critic에 각각 다른 factor를 지정할 수 있다. 두 trainer(Trainer / RLTrainer)는 공용 헬퍼 `mdp/training/_schedulers.py`를 경유하므로 같은 Recipe 필드에서 LinearLR까지 bit-identical하게 전달된다.

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
- 매 step 호출(`log_step_metrics`): 실패 시 `logger.debug` (MLflow 장애 시 경고 폭주 방지)

---

## MLflow Logging Conventions

MDP는 `mdp/training/_mlflow_logging.py` 공용 헬퍼를 통해 Trainer·RLTrainer 양쪽이 동일한 키·시점·API로 MLflow에 기록한다. 값의 성격에 따라 저장소가 기계적으로 결정되며, 같은 개념은 항상 같은 저장소로 간다.

### 3분류 규약 (static / dynamic / summary)

| 범주 | 저장소 API | 대표 키 | 호출 시점 | 의미 |
|------|-----------|---------|-----------|------|
| **static** | `mlflow.log_params` | `task`, `model_class`, `policy_class`, `algorithm`, `batch_size`, `epochs`, `max_steps`, `precision`, `gradient_accumulation_steps`, `learning_rate_init`, `adapter_component`, `adapter_r`, `strategy`, `dataset_source`, `pretrained` | run 시작 시 1회 (`log_static_params`) | 실험을 재현할 때 필요한 "어떻게 실행했는가". Recipe·Settings 선언값 직송 |
| **dynamic** | `mlflow.log_metrics(step=...)` | `learning_rate`, `learning_rate/{group_name_or_idx}`, `momentum`, `weight_decay`, `train_loss`, `val_*` | step/epoch 경계 (`log_step_metrics`·`log_epoch_metrics`) | scheduler로 변하는 "훈련 중 무슨 일이 일어났는가"의 시계열 |
| **summary** | `mlflow.set_tag` + `mlflow.log_metrics` (+ `log_dict`, `log_artifacts`) | tag: `stopped_reason`, `checkpoints_saved`, `best_checkpoint` / metric: `training_duration_seconds`, `total_steps`, `final_{k}` | run 종료 직전 (`log_summary`) | "run이 어떻게 끝났는가"의 단일 스냅샷 |

### Multi-group Optimizer 네이밍

`collect_optimizer_state`가 `optimizer.param_groups` 전체를 순회해 아래 규약으로 metric 키를 생성한다.

| 축 | 키 형태 | 예시 |
|---|---|---|
| single-group, single-optimizer | `learning_rate` | `learning_rate` |
| single-group, multi-optimizer | `learning_rate/{opt_name}` | `learning_rate/policy`, `learning_rate/critic` |
| multi-group, single-optimizer | `learning_rate/group_{idx}` (param_group에 `name` 있으면 `learning_rate/{name}`) | `learning_rate/group_0`, `learning_rate/lora` |
| multi-group, multi-optimizer | `learning_rate/{opt_name}/group_{idx}` (name 있으면 `/{name}`) | `learning_rate/policy/lora` |

`momentum`, `weight_decay`도 해당 param_group에 키가 실제 존재할 때만 같은 규칙으로 추가된다(SGD는 momentum 포함, AdamW는 없음 — optimizer 종류를 caller가 몰라도 자연스럽게 분기). slash 네이밍은 MLflow UI에서 계층 그룹으로 표시되며 Composer(`lr-Adam/group0`)·Lightning(`Adam/pg1`) 업계 관례와 일치한다.

### Trainer ↔ RLTrainer 대칭

| 시점 | Trainer | RLTrainer |
|------|---------|-----------|
| run 시작 | `log_static_params(recipe, settings)` | 동일 |
| 매 grad_accum 경계 | `log_step_metrics(self._optimizer_dict(), global_step, extra={"train_loss": ...})` | `log_step_metrics(self.optimizers, global_step, extra={"loss": ...})` |
| epoch 경계 (Trainer만 epoch 축 보유) | `log_epoch_metrics(self._optimizer_dict(), epoch, extra={"epoch_train_loss": ..., **val_metrics})` | (epoch 개념 없음, step만) |
| run 종료 | `log_summary(final_metrics=self.last_metrics, checkpoint_stats=aggregate_checkpoint_stats(...), ...)` | 동일 (RLTrainer도 `final_*` 블록 보유 — U3 대칭 복구) |

Trainer의 `_optimizer_dict()`가 단일 `self.optimizer`를 `{"policy": ...}`로 포장하여 RLTrainer의 `self.optimizers` dict와 같은 시그니처를 갖춘다 — 공용 헬퍼가 양쪽에서 동일하게 동작.

### 하위 호환 주의

spec-logging-consistency 적용 이후 MLflow run에서 **더 이상 생성되지 않는** 키:

| 구 키 | 폐지 이유 | 대체 키 |
|------|----------|---------|
| `params/learning_rate` | warmup step 0 `optimizer.param_groups[0]["lr"]` 스냅샷이 recipe 선언값으로 오인되는 구조적 결함(원칙 2 위반) | `params/learning_rate_init` (recipe 선언값) + `metrics/learning_rate` (scheduler-adjusted 시계열) |
| `params/policy_lr` | 동일 (RLTrainer 측 비대칭 네이밍 + 동일 결함) | 위와 동일 |

기존 대시보드·스크립트가 `params/learning_rate` 또는 `params/policy_lr`을 조회한다면 `params/learning_rate_init`(정적, Recipe 선언값)로 전환하거나 `metrics/learning_rate` step 0 값(실제 적용된 LR의 시계열 첫 점)을 사용한다. 두 값은 warmup이 활성화된 경우 **의미가 다르다** — `params/learning_rate_init`는 "Recipe에 명시된 base LR", `metrics/learning_rate` step 0은 "warmup_start_factor가 적용된 실제 step 0의 LR".

### 상호 참조

- `stopped_reason`·`checkpoints_saved` tag 적재 규칙은 본 문서 "Train Result" 섹션과 `docs/training.md` "Graceful Shutdown"·"ModelCheckpoint `strict` 옵션" 섹션에 별도 명세(spec-trainer-robustness-fixes)
- Intervention callback의 `intervention.*` MLflow tag 규약은 `docs/training.md` 콜백 섹션에 명세(spec-callback-restructure)
- 공용 헬퍼는 콜백이 아닌 **Trainer builtin** 경로다 — spec-callback-restructure가 `LRMonitor` 콜백을 삭제한 근거("로깅은 builtin")와 일관

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
  "stopped_reason": "early_stopped | completed | max_steps_reached | signal_term | signal_int",
  "duration_seconds": 3600.5,
  "checkpoints_saved": 3,
  "monitoring": {"baseline_saved": true, "baseline_path": "mlruns/.../baseline.json"}
}
```

**`stopped_reason` 리터럴**:

| `mdp train` (SFT) | `mdp rl-train` (RL) | 의미 |
|---|---|---|
| `completed` | `completed` | 루프 정상 종료 (epochs/steps 소진 전) |
| `early_stopped` | `early_stopping` | 콜백(EarlyStopping 등) `should_stop=True` |
| `max_steps_reached` | `max_steps` | `max_steps` 도달 |
| `signal_term` | `signal_term` | SIGTERM 수신 (외부 `timeout`, K8s eviction 등) |
| `signal_int` | `signal_int` | SIGINT 수신 (Ctrl+C) |

Trainer/RLTrainer 표기 불일치(`early_stopped` vs `early_stopping`, `max_steps_reached` vs `max_steps`)는 의도된 호환성 유지다. 기존 오케스트레이터 가정을 깨지 않으려 새 리터럴(`signal_term`/`signal_int`)만 대칭으로 추가했다.

**`checkpoints_saved` 필드**: `int | None`. 실제 디스크에 저장된 체크포인트 개수. `None`은 legacy/unknown, `0` 이상 정수는 실제 개수. 상위 오케스트레이터(auto-research 등)가 MLflow artifact 조회 없이 "이 run이 산출물을 남겼는가"를 판정할 때 사용한다. `0`이면 `ModelCheckpoint.monitor` 미매칭 등으로 인한 silent failure를 즉시 감지 가능.

### Graceful Shutdown

`mdp train` / `mdp rl-train`은 SIGTERM·SIGINT를 `Trainer.train()` / `RLTrainer.train()` 내부에서 처리한다. 외부 `timeout` 명령이나 Ctrl+C로 중단되어도:

- 현재 step 경계에서 break → finally(cleanup + on_train_end + MLflow summary) 실행
- MLflow run이 `FINISHED` 상태로 마감 (zombie RUNNING 아님)
- `stopped_reason` tag에 `signal_term` 또는 `signal_int`, `checkpoints_saved` tag에 저장 개수 기록

```bash
# 2시간 wall-clock 상한 + graceful shutdown
timeout 2h mdp train -r recipe.yaml -c config.yaml --format json
# → MLflow run FINISHED + stopped_reason=signal_term + final_* metric 정상 기록
```

첫 시그널만 `stopped_reason`에 반영된다 (SIGTERM 후 Ctrl+C 연타는 race 없이 SIGTERM 유지). 분산 학습에서는 torchrun이 모든 rank에 SIGTERM을 전파하여 동시에 step 경계 break → NCCL deadlock 없음. 추론 경로(`mdp inference`, `mdp generate`)에는 현재 handler가 **없으므로** 장시간 추론에 timeout을 걸면 출력이 부분 저장될 수 있다.

### ModelCheckpoint `strict` 옵션

`ModelCheckpoint`는 `strict: bool = False` 파라미터를 지원한다. `true`로 설정하면 첫 validation에서 `monitor` 이름이 metric dict에 없을 때 즉시 `ValueError`로 학습을 중단시킨다. 기본 `false`는 WARNING(사용 가능한 metric key 목록 포함) 후 저장 skip — 기존 동작.

```yaml
callbacks:
  - _component_: ModelCheckpoint
    monitor: val_loss
    strict: true    # 자동 스윕에서 monitor 오타를 즉시 감지
```

자동 스윕에서 "학습은 성공 반환, 산출물은 0개"인 silent failure를 막는 방어선. `critical`(콜백 예외 재전파 스위치)과는 이름만 유사하고 의미가 다르다 — 두 속성은 독립적으로 조합 가능.

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
    # └─ 파라미터 그룹별 LR이 필요할 때. {"optimizer": optim_obj} 반환 필수.
    #    None(기본) 또는 dict 형식 외의 반환값은 무시되고 recipe fallback 사용.
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

### `configure_optimizers()` — 파라미터 그룹별 LR

파라미터 그룹마다 다른 LR이 필요할 때 오버라이드한다 (backbone 낮은 LR + head 높은 LR 등).

**반환 형식은 반드시 `{"optimizer": optimizer_객체}` dict이어야 한다.** optimizer를 직접 반환하면 `isinstance(custom, dict)` 체크에서 실패 → 무증상으로 recipe fallback 사용.

정의하면 recipe의 `optimizer:` 전체가 무시된다. `scheduler:`는 recipe에서 그대로 적용.

→ 전체 패턴: `docs/extending.md` — `### configure_optimizers() — 파라미터 그룹별 LR`

### `export()` / `load_from_export()` 계약

단일 네트워크는 기본 구현으로 충분. backbone + 커스텀 head처럼 이질적 구조는 두 메서드를 **반드시 함께** 오버라이드한다 — `export()`만 오버라이드하면 `reconstruct_model()`이 기본 safetensors 로더로 떨어져 실패.

→ 전체 패턴: `docs/extending.md` — `### export() / load_from_export() 계약`
