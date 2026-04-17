# Getting Started

설치부터 첫 학습/추론까지의 워크플로우를 안내한다.

## 설치

```bash
# 코어 (PyTorch, Pydantic, MLflow 등)
pip install -e .

# 태스크별 선택 설치
pip install -e ".[language]"              # LLM/NLP (transformers, peft, bitsandbytes)
pip install -e ".[vision]"                # Vision (torchvision, timm, albumentations)
pip install -e ".[serving]"               # 서빙 (accelerate, FastAPI, uvicorn)
pip install -e ".[language,serving]"      # 복수 선택
pip install -e ".[all]"                   # 전체
```

Python >= 3.10, PyTorch >= 2.1이 필요하다.

## 워크플로우 개요

```
mdp init → Recipe/Config 작성 → mdp estimate → mdp train → mdp inference → mdp export → mdp serve
```

각 단계는 독립적이므로 필요한 단계만 선택적으로 사용할 수 있다. 예를 들어, 학습 없이 오픈소스 모델을 바로 추론하려면 `mdp inference --pretrained hf://model-name`만 실행하면 된다.

## Step 1: 프로젝트 스캐폴딩

```bash
# 지원 태스크 확인
mdp list tasks

# 태스크에 호환되는 모델 확인
mdp list models --task text_generation

# 프로젝트 생성
mdp init my_project --task text_generation --model llama3-8b
```

`mdp init`은 다음 구조를 생성한다:

```
my_project/
├── recipes/example.yaml   # 실험 정의 (모델, 데이터, 학습 설정)
├── configs/local.yaml     # 인프라 설정 (GPU, MLflow, 체크포인트)
├── data/                  # (빈 디렉토리) 로컬 데이터 둘 곳
├── checkpoints/           # (빈 디렉토리) 체크포인트 저장 경로
└── .gitignore
```

이후 명령은 이 경로를 참조하면 된다:

```bash
mdp train -r my_project/recipes/example.yaml -c my_project/configs/local.yaml
```

생성된 Recipe에서 `???`로 표시된 필드만 채우면 된다:
- `data.dataset.source` — 데이터셋 이름 또는 경로
- `head.num_classes` — 분류 태스크 시 클래스 수
- `metadata.*` — 실험 메타정보

## Step 2: GPU 메모리 추정

학습 전에 필요한 GPU 메모리를 확인한다:

```bash
mdp estimate -r my_project/recipes/example.yaml
```

출력 예시:
```
Model Memory:     16.1 GB (bf16)
Gradient Memory:  32.2 GB (fp32)
Optimizer Memory: 64.4 GB (AdamW)
Activation Memory: ~8.0 GB
─────────────────────────────────
Total Estimated:  120.7 GB
Suggested GPUs:   4x (24GB each)
Suggested Strategy: fsdp
```

QLoRA를 사용하면 메모리를 대폭 절약할 수 있다:
```yaml
# recipe.yaml에 adapter 추가
adapter:
  _component_: QLoRA
  r: 64
  alpha: 128
  quantization:
    bits: 4
```

## Step 3: 학습

```bash
# 단일 GPU
mdp train -r my_project/recipes/example.yaml -c my_project/configs/local.yaml

# 멀티 GPU (자동 감지 — GPU가 2개 이상이면 torchrun 자동 실행)
mdp train -r my_project/recipes/example.yaml -c my_project/configs/local.yaml

# 런타임 override
mdp train -r my_project/recipes/example.yaml -c my_project/configs/local.yaml \
  --override training.epochs=5 \
  --override data.dataloader.batch_size=8

# wall-clock 상한 + graceful shutdown
timeout 2h mdp train -r my_project/recipes/example.yaml -c my_project/configs/local.yaml
# 2시간 후 SIGTERM → Trainer가 현재 step 완료 후 finally 실행 →
# MLflow run이 FINISHED로 마감되고 stopped_reason=signal_term tag가 남는다
```

학습 결과는 MLflow에 자동 기록된다. 체크포인트는 Config의 `storage.checkpoint_dir`에 저장된다.

## Step 4: 추론

```bash
# 학습된 모델로 추론
mdp inference --run-id <mlflow-run-id> --data test.jsonl \
  --metrics Accuracy F1Score \
  --output-dir ./results

# 오픈소스 모델 직접 추론 (학습 없이)
mdp inference --pretrained hf://meta-llama/Meta-Llama-3-8B \
  --data test.jsonl
```

## Step 5: 텍스트 생성

```bash
mdp generate --run-id <id> \
  --prompts prompts.jsonl \
  --output generated.jsonl \
  --max-new-tokens 256 \
  --temperature 0.7
```

## Step 6: 모델 내보내기 & 서빙

```bash
# adapter merge + 패키징
mdp export --run-id <id> --output ./model-production

# REST API 서빙
mdp serve --model-dir ./model-production --port 8000
```

서버 엔드포인트:
- `POST /predict` — 추론 요청
- `GET /health` — 상태 확인

## 다음 단계

- [Configuration Guide](configuration.md) — Recipe/Config YAML 상세 가이드
- [Training Guide](training.md) — 분산 학습, RL, 콜백
- [Inference & Serving](inference-and-serving.md) — 배치 추론, 서빙 상세
- [Extending MDP](extending.md) — 커스텀 모델, 콜백 작성
