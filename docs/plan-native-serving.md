# MDP 서빙 재설계

## 현재 상태

### 이미 완료된 것

- **train**: 학습 완료 시 MLflow artifact로 `checkpoint/`와 `model/`을 자동 등록
- **inference**: `--run-id` 기반. MLflow에서 artifact를 받아 모델 재구성 + 추론 + metric 평가

### 해결해야 하는 것

`mdp serve`가 학습 결과를 서빙하지 않는다. `--backend vllm|torchserve`를 필수로 요구하고, 실제로는 외부 프레임워크에 원본 pretrained 모델 이름만 전달하는 런처에 불과하다.

---

## 설계 원칙

1. **MDP가 직접 서빙한다.** recipe.yaml의 fields/task/tokenizer를 읽고 API 엔드포인트를 동적으로 생성한다.
2. **학습과 서빙의 전처리가 동일하다.** 같은 tokenizer, 같은 transform, 같은 fields 매핑.
3. **학습이 끝나면 model/ artifact가 즉시 모든 용도로 사용 가능하다.** MDP 서빙, vLLM 배포, HF Hub 업로드에 별도 변환 단계 없이 바로 사용.
4. **LLM 서빙 품질을 위해 streaming generation과 dynamic batching을 지원한다.**

---

## model/ artifact 포맷

학습 완료 시 MLflow `model/` artifact에 저장되는 포맷이다. **inference, serve, vLLM 배포 모두 이 artifact 하나로 해결한다.**

```
model/
├── model.safetensors      ← 가중치 (LoRA면 merge 완료)
├── config.json            ← HF 모델이면 save_pretrained()이 생성 (vLLM이 읽음)
├── tokenizer.json         ← tokenizer (MDP, vLLM 모두 사용)
├── tokenizer_config.json
├── special_tokens_map.json
├── recipe.yaml            ← MDP 메타데이터 (fields, task, 전처리 설정)
└── serving_meta.json      ← 서빙 메타데이터 (vLLM 호환 여부 등)
```

각 소비자가 필요한 파일만 읽는다:
- **MDP** (inference, serve): `recipe.yaml` + `model.safetensors` + tokenizer + `serving_meta.json`
- **vLLM**: `config.json` + `model.safetensors` + tokenizer
- **HF Hub**: 전체 디렉토리를 그대로 업로드 (recipe.yaml이 포함되어도 무해)

HF 모델이면 `save_pretrained()`로 저장하고, 비-HF 모델(timm, 커스텀)이면 `save_file(state_dict)`로 저장한다. 비-HF 모델은 `config.json`이 없으므로 vLLM으로 배포할 수 없지만, 이는 모델 아키텍처의 제약이지 MDP의 제약이 아니다.

### serving_meta.json

학습 후 실제 모델 상태를 검사한 결과. 프로덕션 배포 가능 여부를 안내한다.

```json
{
  "vllm_available": true,
  "model_class": "LlamaForCausalLM",
  "head_replaced": false,
  "adapter_merged": true
}
```

### `checkpoint/` vs `model/`

| artifact | 용도 |
|----------|------|
| `checkpoint/` | **resume 전용** (adapter 미병합, optimizer.pt 포함) |
| `model/` | **그 외 전부** (inference, serve, vLLM 배포, HF Hub) |

---

## 학습 저장 로직

### `_prepare_serving_model` (trainer.py)

현재 `_export_and_log_model`을 대체한다. 한 번에 모든 용도로 사용 가능한 `model/` artifact를 생성한다.

```python
def _prepare_serving_model(self, checkpoint_dir: Path) -> Path:
    model, settings = reconstruct_model(checkpoint_dir)
    recipe = settings.recipe

    # PEFT adapter 병합
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()

    output_dir = Path(tempfile.mkdtemp())

    # 모델 저장
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)         # → model.safetensors + config.json
    else:
        save_file(model.state_dict(), output_dir / "model.safetensors")

    # tokenizer 저장
    if recipe.data.tokenizer:
        pretrained = recipe.data.tokenizer.get("pretrained")
        if pretrained:
            AutoTokenizer.from_pretrained(pretrained).save_pretrained(output_dir)

    # recipe.yaml 복사
    shutil.copy(checkpoint_dir / "recipe.yaml", output_dir / "recipe.yaml")

    # serving_meta.json
    meta = {
        "vllm_available": hasattr(model, "config") and recipe.head is None,
        "model_class": type(model).__name__,
        "head_replaced": recipe.head is not None,
        "adapter_merged": hasattr(model, "peft_config"),  # merge 전에 있었다면
    }
    (output_dir / "serving_meta.json").write_text(json.dumps(meta))

    return output_dir
```

**별도 export 단계 없음.** 이 함수 하나가 MDP 서빙과 vLLM 배포 모두를 위한 artifact를 만든다.

---

## 서빙 기능

### Streaming Generation (LLM용)

`task == "text_generation"` 또는 `"seq2seq"`일 때, `/predict`가 SSE 스트리밍 응답을 반환한다.

```python
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
thread = threading.Thread(
    target=model.generate,
    kwargs={"input_ids": input_ids, "streamer": streamer, "max_new_tokens": 256},
)
thread.start()

async def stream():
    for token in streamer:
        yield f"data: {json.dumps({'token': token})}\n\n"
    yield "data: [DONE]\n\n"

return StreamingResponse(stream(), media_type="text/event-stream")
```

### Dynamic Batching (비-LLM용)

`task`가 classification, detection 등일 때, 짧은 시간 창(`batch_window`) 내에 도착한 요청을 모아 배치 처리한다.

```python
class BatchScheduler:
    async def submit(self, preprocessed: dict) -> dict:
        future = asyncio.get_event_loop().create_future()
        await self.queue.put((preprocessed, future))
        return await future

    async def _batch_loop(self):
        while True:
            items = [await self.queue.get()]
            # batch_window(50ms) 내 추가 요청 수집
            # 배치로 묶어 model(batch) 1회 실행
            # 결과를 각 future에 분배
```

### 기능 적용 매트릭스

| task | streaming | dynamic batching |
|------|-----------|-----------------|
| text_generation, seq2seq | O | X |
| 그 외 (classification, detection 등) | X | O |

---

## CLI 설계

### `mdp serve`

```bash
mdp serve --run-id abc123 --port 8000 [--host 0.0.0.0]
```

`--backend` 없다. recipe.yaml의 task를 보고 streaming/batching을 자동 결정한다.

### `mdp inference` (이미 구현, artifact 경로만 변경)

```bash
mdp inference --run-id abc123 --data ./test.csv [--metrics Accuracy] [--fields text=content]
```

### 동적 엔드포인트

```
POST /predict  {"text": "Once upon a time"}           # text_generation → SSE 스트림
POST /predict  (multipart/form-data, image: <file>)    # image_classification → JSON
GET  /health   → {"status": "ok", "model": "...", "task": "..."}
```

---

## 내부 흐름

### inference와 serve의 공통 경로

```python
model_dir = mlflow.artifacts.download_artifacts(run_id, "model")
model, settings = reconstruct_model(model_dir)

# tokenizer: artifact에서 로컬 로드
tokenizer = AutoTokenizer.from_pretrained(str(model_dir))  # tokenizer.json이 model_dir에 있음

# transform: recipe에서 val용 로드
transform = build_transforms(recipe.data.augmentation.get("val"))
```

inference와 serve가 동일한 전처리 코드를 공유한다.

### serve 분기

```python
if recipe.task in ("text_generation", "seq2seq"):
    handler = StreamingHandler(model, tokenizer, recipe)
else:
    scheduler = BatchScheduler(model, tokenizer, transform, recipe)
    handler = BatchHandler(scheduler)

app = create_app(handler, recipe)
uvicorn.run(app, host=host, port=port)
```

---

## 코드 변경 상세

### 변경 1: `trainer.py` — `_prepare_serving_model`

현재 `_export_and_log_model`을 대체. `save_pretrained()`(HF 모델) 또는 `save_file()`(비-HF) + recipe.yaml + tokenizer + serving_meta.json을 한 번에 생성. 위의 pseudocode 참조.

`serving/export.py`는 삭제한다. training 저장과 export가 분리되지 않으므로 별도 모듈이 필요 없다.

### 변경 2: `serving/server.py` (신규)

FastAPI 서버. handler 기반 동적 엔드포인트.

```python
def create_app(handler, recipe: Recipe) -> FastAPI:
    app = FastAPI(title=f"MDP: {recipe.name}")

    @app.post("/predict")
    async def predict(request: Request):
        raw = await _parse_request(request, recipe.data.fields, recipe.task)
        return await handler.handle(raw)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": recipe.name, "task": recipe.task}

    return app
```

### 변경 3: `serving/handlers.py` (신규)

```python
class StreamingHandler:
    """text_generation/seq2seq용. TextIteratorStreamer + SSE."""
    async def handle(self, raw_input: dict) -> StreamingResponse: ...

class BatchHandler:
    """classification/detection용. BatchScheduler + JSON."""
    async def handle(self, raw_input: dict) -> dict: ...
```

### 변경 4: `cli/serve.py` 재작성

```python
def run_serve(run_id: str, port: int, host: str = "0.0.0.0") -> None:
    import mlflow, uvicorn

    model_dir = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model"))
    handler = _create_handler(model_dir)
    recipe = _load_recipe(model_dir)
    app = create_app(handler, recipe)
    uvicorn.run(app, host=host, port=port)
```

### 변경 5: `__main__.py`

serve 시그니처 변경 (`--backend` 제거, `--host` 추가). `export` 커맨드 없음.

```python
@app.command()
def serve(
    run_id: str = typer.Option(..., "--run-id"),
    port: int = typer.Option(8000, "--port"),
    host: str = typer.Option("0.0.0.0", "--host"),
): ...
```

### 변경 6: `cli/inference.py` — artifact 경로 변경

`_download_checkpoint(run_id)` → `_download_model(run_id)`. `artifact_path="checkpoint"` → `"model"`.

### 변경 7: `serving/model_loader.py` — 함수 정리

- `SettingsFactory.from_checkpoint()` → `from_artifact()`로 이름 변경 (model/과 checkpoint/ 모두 recipe.yaml이 있음)
- `load_serving_model(model_dir)`: safetensors만 (merge 완료 상태)
- `load_checkpoint_weights(ckpt_dir)`: adapter/safetensors/pt 3가지 분기 (resume용)

### 삭제 대상

- `serving/export.py` — training 저장에 흡수. 별도 모듈 불필요
- `cli/export.py` — CLI 커맨드 불필요
- `serving/vllm_server.py` — MDP가 vLLM을 관리하지 않음
- `serving/torchserve.py` — 동일

---

## 전체 CLI 흐름

```bash
# 1. 학습
mdp train -r recipe.yaml -c config.yaml
# → MLflow run abc123:
#     checkpoint/ (resume용)
#     model/      (MDP 서빙 + vLLM 배포 + HF Hub 모두 가능)

# 2. 배치 추론 + 평가
mdp inference --run-id abc123 --data ./test.csv --metrics Accuracy

# 3. MDP 실시간 서빙
mdp serve --run-id abc123 --port 8000

# 4. vLLM 프로덕션 배포 (MDP 밖, model/ artifact 직접 사용)
mlflow artifacts download --run-id abc123 --artifact-path model -d ./model
vllm serve ./model

# 5. HF Hub 업로드 (MDP 밖)
huggingface-cli upload myorg/my-model ./model
```

별도 export 단계 없음. model/ artifact가 모든 용도로 즉시 사용 가능.

---

## 배치 추론과 실시간 서빙의 일관성

| | inference | serve (streaming) | serve (batching) |
|---|---|---|---|
| **artifact** | `model/` | `model/` | `model/` |
| **모델 로딩** | `reconstruct_model` | `reconstruct_model` | `reconstruct_model` |
| **전처리** | recipe tokenizer/transform | recipe tokenizer | recipe tokenizer/transform |
| **모델 호출** | `model(batch)` | `model.generate(streamer=)` | `model(batch)` via scheduler |
| **후처리** | `_postprocess()` | 토큰 디코딩 | `_postprocess()` |

---

## 의존성

```toml
[project.optional-dependencies]
serve = ["fastapi>=0.100", "uvicorn>=0.20"]
```

---

## 테스트 계획

### 신규 테스트

#### `tests/e2e/test_server.py`

```python
def test_predict_text_classification():
    """text_classification → /predict에 text → JSON 응답."""

def test_predict_image_classification():
    """image_classification → /predict에 이미지 → JSON 응답."""

def test_streaming_text_generation():
    """text_generation → /predict가 SSE 스트리밍."""

def test_health_endpoint():
    """/health가 model, task를 반환."""

def test_dynamic_batching():
    """동시 요청 N개 → 배치 처리."""
```

### 기존 테스트 수정

| 파일 | 변경 |
|------|------|
| `test_mlflow_artifacts.py` | `model/` artifact에 serving_meta.json 포함 검증 추가 |
| `test_inference.py` | artifact 경로 변경 관련 (CLI 레벨, 단위 테스트는 유지) |

---

## 구현 순서

```
1. trainer.py — _prepare_serving_model (save_pretrained/save_file + recipe.yaml + tokenizer + serving_meta.json)
2. serving/model_loader.py — from_artifact 이름 변경, load_serving_model/load_checkpoint_weights 분리
3. serving/handlers.py — StreamingHandler + BatchHandler + BatchScheduler
4. serving/server.py — FastAPI + handler 기반 동적 엔드포인트
5. cli/serve.py — 재작성 (--backend 제거)
6. cli/inference.py — artifact 경로 model/로 변경
7. __main__.py — serve 시그니처 변경, export 커맨드 없음
8. 삭제: serving/export.py, serving/vllm_server.py, serving/torchserve.py
9. 테스트
```
