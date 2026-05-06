# Inference & Serving

배치 추론, 텍스트 생성, 모델 내보내기, REST API 서빙을 다룬다.

## 모델 소스

추론/생성/서빙 명령은 세 가지 모델 소스를 지원한다 (상호 배타). CLI는 먼저 `ModelSourcePlan`을 만들어 `artifact` 경로인지 `pretrained` URI인지 결정하고, 이후 로더는 이 plan을 소비한다.

| 옵션 | 설명 | 용도 |
|------|------|------|
| `--run-id <id>` | MLflow run 아티팩트 | 학습된 모델 로드 |
| `--model-dir <dir>` | 로컬 디렉토리 | export된 모델 또는 체크포인트 |
| `--pretrained <uri>` | 오픈소스 모델 URI | 학습 없이 직접 사용 |

Source precedence and support:
- `--run-id`, `--model-dir`, `--pretrained` 중 정확히 하나만 지정한다.
- `--pretrained`는 `mdp inference`와 `mdp generate`에서만 지원된다. `mdp serve`와 `mdp export`는 artifact/checkpoint source만 받는다.
- `--run-id`는 MLflow `model/` artifact를 로컬로 내려받은 뒤 artifact plan으로 처리한다.
- `--model-dir`는 로컬 artifact, export 결과, 또는 manifest checkpoint 디렉토리로 처리한다. DeepSpeed ZeRO engine checkpoint는 현재 일반 checkpoint source가 아니다.

Pretrained URI 형식:
- `hf://meta-llama/Llama-3-8B` — HuggingFace Hub
- `timm://resnet50` — TIMM
- `ultralytics://yolov8n` — YOLO
- `local://./my-model` — 로컬 경로

---

## 배치 추론

```bash
mdp inference --run-id <id> --data test.jsonl \
  --fields text=text_column label=label_column \
  --metrics Accuracy F1Score \
  --output-format parquet \
  --output-dir ./results
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `--data` | 데이터셋 (HF Hub 이름 또는 로컬 파일) |
| `--fields` | 역할→컬럼 매핑 (`role=column` 형식) |
| `--metrics` | 평가 메트릭 이름 (torchmetrics) |
| `--output-format` | `parquet` / `csv` / `jsonl` |
| `--output-dir` | 결과 저장 디렉토리 |
| `--device-map` | 멀티 GPU 분배 (`auto` / `balanced` / `sequential`) |
| `--callbacks` | 추론 콜백 YAML 파일 |
| `--save-output` | 콜백 전용 모드에서도 DefaultOutputCallback으로 결과 파일을 저장 |

### 오픈소스 모델 직접 추론

Recipe 없이 오픈소스 모델을 바로 사용할 수 있다. `--pretrained`로 지정된 모델은 `config.architectures`에서 클래스를 자동 결정한다. `architectures` 필드가 없는 모델은 에러가 발생한다.

```bash
mdp inference --pretrained hf://meta-llama/Meta-Llama-3-8B \
  --data test.jsonl \
  --fields text=text \
  --metrics Accuracy

# 토크나이저 명시 (자동 추론이 안 될 때)
mdp inference --pretrained hf://model --tokenizer other-tokenizer \
  --data test.jsonl

# dtype/attention/remote-code 지정
mdp inference --pretrained hf://Qwen/Qwen2.5-7B \
  --data test.jsonl --dtype float32 --trust-remote-code --attn-impl sdpa
```

### Pretrained 모델 로딩 옵션

`--pretrained` 사용 시 `from_pretrained()`에 전달되는 추가 옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dtype` | None (모델 기본) | 모델 로딩 dtype (`float32` / `float16` / `bfloat16`). BFloat16 기본 모델이 numpy 변환 에러를 일으킬 때 `float32`를 지정한다 |
| `--trust-remote-code` | False | HuggingFace 모델의 custom code 실행을 허용한다 (Qwen2.5, Gemma-2 등) |
| `--attn-impl` | None (모델 기본) | 어텐션 구현 (`flash_attention_2` / `sdpa` / `eager`) |
| `--device-map` | None | `from_pretrained(device_map=...)`으로 전달. 설정 시 `.to(device)` 호출을 스킵한다 |

Direct-pretrained loading and Recipe-based loading share the same option normalization: CLI dtype/trust-remote-code/attention/device-map values are converted once and then routed only to protocols that accept them.

### 드리프트 감지

학습 시 baseline이 저장되었으면, 추론 시 자동으로 데이터 분포 변화를 감지한다.
`--format json` 출력에 `monitoring.drift_detected`, `severity_level` 등이 포함된다.

---

## 텍스트 생성

```bash
mdp generate --run-id <id> \
  --prompts prompts.jsonl \
  --prompt-field prompt \
  --output generated.jsonl \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --top-p 0.9 \
  --num-samples 3 \
  --batch-size 8
```

### 입력 형식

`prompts.jsonl` — 각 줄이 JSON 객체:
```json
{"prompt": "Explain quantum computing in simple terms."}
{"prompt": "Write a Python function to sort a list."}
```

`--prompt-field`로 프롬프트 텍스트가 담긴 필드명을 지정한다 (기본: `prompt`).

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--max-new-tokens` | 256 | 최대 생성 토큰 수 |
| `--temperature` | 1.0 | 샘플링 온도 |
| `--top-p` | 1.0 | Nucleus sampling |
| `--top-k` | 50 | Top-k sampling |
| `--do-sample` | true | 샘플링 활성화 |
| `--num-samples` | 1 | 프롬프트당 생성 수 |
| `--batch-size` | 1 | 배치 크기 |

Generation kwargs precedence is `MDP defaults < Recipe generation < explicit CLI/serving args`. Batch `mdp generate` and REST serving share the same resolver for the common HuggingFace generation parameters; SSE formatting and request lifecycle remain serving-local.

---

## 모델 내보내기

LoRA/QLoRA 어댑터를 base 모델에 merge하여 배포용 패키지를 생성한다:

```bash
mdp export --run-id <id> --output ./model-production

# 로컬 체크포인트에서도 가능
mdp export --checkpoint ./checkpoints/best --output ./model-production
```

출력 구조:
```
model-production/
  ├── model.safetensors       # merged 전체 모델
  ├── config.json
  ├── tokenizer.json
  ├── special_tokens_map.json
  └── recipe.yaml             # 서빙 메타데이터
```

> MLflow에 저장되는 LoRA 아티팩트는 어댑터만(~50MB) 포함한다. `export`가 on-demand merge를 수행한다.

---

## REST API 서빙

```bash
# MLflow에서 모델 로드
mdp serve --run-id <id> --port 8000

# export된 모델에서 로드
mdp serve --model-dir ./model-production --port 8000 --host 0.0.0.0
```

### 엔드포인트

**`POST /predict`** — 추론 요청

LLM (text_generation/seq2seq):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Explain quantum computing"}'
```
SSE(Server-Sent Events) 스트리밍으로 토큰 단위 응답을 반환한다.

분류 태스크:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is great"}'
```

**`GET /health`** — 서버 상태 확인

### 멀티 GPU 서빙

단일 GPU를 초과하는 대형 모델은 `device_map`으로 멀티 GPU에 분배한다:

```bash
mdp serve --run-id <id> \
  --device-map auto \
  --max-memory '{"0": "24GiB", "1": "40GiB"}'
```

| device_map | 동작 |
|-----------|------|
| `auto` | accelerate가 자동 분배 |
| `balanced` | GPU 간 균등 분배 |
| `sequential` | GPU 0부터 순서대로 채움 |

> **device_map은 추론/서빙 전용.** `device_map`으로 분산 배치된 모델(`hf_device_map` 속성 존재)은 `mdp train`·`mdp rl-train`에서 명시적 guard(`mdp/training/trainer.py`, `mdp/training/rl_trainer.py`)에 의해 거부된다. 멀티 GPU 학습에는 `compute.distributed.strategy`의 DDP/FSDP를 사용한다. `deepspeed*` 전략명은 현재 fail-fast 경계로 남아 있다.

Serving config precedence is `artifact config snapshot < explicit serve CLI`. `SettingsFactory.from_artifact()` uses `config_file` from the artifact manifest when present, otherwise standard config snapshot filenames, otherwise schema defaults. `mdp serve --device-map` and `--max-memory` override artifact-derived serving values at runtime.

### 서빙 아키텍처

- **LLM**: `GenerateHandler` — HuggingFace `TextIteratorStreamer` + SSE 스트리밍. 동시 요청은 lock으로 제한 (1 stream/process)
- **분류/NER**: `PredictHandler` — 동기 추론. `max_batch_size > 1`이면 동적 배칭

### 프로덕션 배포

MDP 서빙은 검증/데모 용도다. 프로덕션 LLM 서빙은:

1. `mdp export`로 모델 패키징
2. vLLM 또는 TGI에 전달 (tensor parallelism, continuous batching)

---

## 추론 콜백

추론 루프는 `forward_fn(batch)` -> 콜백 dispatch -> metric update만 수행한다. 출력 처리(softmax, numpy 변환, 파일 저장)는 `DefaultOutputCallback`이 담당하며, 콜백이 없으면 출력 처리가 일어나지 않는다.

### DefaultOutputCallback

`_postprocess` + `_save_results`를 캡슐화한 기본 출력 콜백. 기존 추론 루프에서 하드코딩되어 있던 출력 처리 로직을 콜백으로 분리하여, 콜백 전용 추론(hidden state 추출 등)에서 불필요한 메모리 소비를 제거한다.

CLI는 모델 소스와 콜백 유무에 따라 DefaultOutputCallback을 자동 주입한다:

| 경로 | 콜백 상태 | DefaultOutputCallback | 이유 |
|------|----------|----------------------|------|
| Recipe (`--run-id`, `--model-dir`) | 무관 | 항상 자동 추가 | Recipe 경로는 결과 저장이 기본 기대 |
| Pretrained | 콜백 없음 | 자동 추가 | 결과 저장이 기본 기대 |
| Pretrained | 사용자 콜백 있음 | 미추가 | 콜백 전용 모드 (메모리 절약) |
| Pretrained | 사용자 콜백 + `--save-output` | 명시 추가 | 사용자가 출력 저장도 원함 |

자동 주입은 CLI 레이어에서 수행되며, `run_batch_inference`는 전달받은 콜백을 그대로 실행하는 순수한 엔진이다.

### BaseInferenceCallback / BaseInterventionCallback 인터페이스

추론 콜백은 두 타입으로 분류된다:

- **`BaseInferenceCallback`** (`is_intervention=False`): 읽기 전용 관측. hidden state 추출, attention 분석 등
- **`BaseInterventionCallback`** (`is_intervention=True`): 출력을 바꾸는 개입. `metadata -> dict` 프로퍼티 구현 필수

```python
from mdp.callbacks.base import BaseInferenceCallback

class ActivationAnalyzer(BaseInferenceCallback):
    def setup(self, model, tokenizer=None, **kwargs):
        """추론 시작 전. forward hook 등록."""
        target = dict(model.named_modules())["model.layers.20"]
        self._handle = target.register_forward_hook(self._capture)
        self._activations = []

    def _capture(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        self._activations.append(h.detach().cpu())

    def on_batch(self, batch_idx, batch, outputs, **kwargs):
        """매 배치 forward 후. 캡처된 활성화 처리."""

    def teardown(self, **kwargs):
        """추론 완료 후. hook 해제, 결과 저장."""
        self._handle.remove()
        torch.save(self._activations, "activations.pt")
```

### Intervention MLflow 태깅

`BaseInterventionCallback` 서브클래스를 `--callbacks`로 주입하면, 추론 시작 시 `apply_intervention_tags`가 자동으로 각 callback의 `metadata` 프로퍼티를 MLflow에 기록한다:

- 활성 MLflow run이 있으면 `mlflow.set_tag("intervention.{i}.{key}", value)` 형태로 tag 기록
- MLflow run이 없으면 stdout/log에만 출력

```yaml
# intervention.yaml
- _component_: ResidualAdd
  target_layers: [20, 21]
  vector_path: ./steer_vector.pt
  strength: 1.5
- _component_: LogitBias
  token_biases: {1234: 2.0, 5678: -1.0}
```

```bash
mdp generate --pretrained hf://meta-llama/Meta-Llama-3-8B \
  --prompts prompts.jsonl --callbacks intervention.yaml
# → MLflow에 intervention.0.type=ResidualAdd, intervention.0.strength=1.5 등 자동 기록
```

### 콜백 YAML 작성

```yaml
# analysis.yaml
- _component_: my_project.ActivationAnalyzer
  layers: [20, 40, 60]
```

### 사용

```bash
# 학습된 모델 + 관측 콜백
mdp inference --run-id <id> --data test.jsonl --callbacks analysis.yaml

# 오픈소스 모델 + 관측 콜백
mdp inference --pretrained hf://meta-llama/Meta-Llama-3-8B \
  --data prompts.jsonl --callbacks analysis.yaml
```

추론 루프는 hidden state/attention을 직접 다루지 않으며, 모든 내부 접근은 콜백의 `register_forward_hook`을 통해 이루어진다.

`mdp list callbacks`로 현재 등록된 콜백의 타입(`[Int]`/`[Obs]`/`[Train]`)을 확인할 수 있다.

---

## Observability

All commands support JSON mode; the canonical output and error schema is [Observability](observability.md).

Training graceful shutdown is also documented there. `mdp inference` and `mdp generate` do not currently install the training signal handler, so timeout/Ctrl+C can leave partial output.
