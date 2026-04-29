# MDP — Modern Deeplearning Pipeline

YAML 설정 기반의 딥러닝 학습 / 추론 / 서빙 CLI 도구.

Recipe(실험 정의) + Config(인프라 설정) 두 파일만 작성하면 SFT, RL alignment(DPO/GRPO/PPO), 배치 추론, REST 서빙까지 하나의 CLI로 수행할 수 있다.

## Quick Start

```bash
# 설치 (코어)
pip install -e .

# 언어 모델 + 서빙이 필요하면
pip install -e ".[language,serving]"

# 비전도 필요하면
pip install -e ".[all]"
```

```bash
# 1. 프로젝트 스캐폴딩
mdp init my_project --task text_generation --model llama3-8b

# 2. Recipe의 ??? 필드 채우기 (data.dataset.source 등)

# 3. GPU 메모리 추정
mdp estimate -r recipe.yaml

# 4. 학습
mdp train -r recipe.yaml -c config.yaml

# 5. 추론
mdp inference --run-id <id> --data test.jsonl

# 6. 서빙
mdp serve --run-id <id> --port 8000
```

## Commands

| Command | 설명 |
|---------|------|
| `mdp init` | 프로젝트 스캐폴딩 + Recipe/Config 템플릿 생성 |
| `mdp train` | SFT 학습 (단일/멀티 GPU 자동 감지) |
| `mdp rl-train` | RL alignment 학습 (DPO, GRPO, PPO) |
| `mdp inference` | 배치 추론 + 평가 |
| `mdp generate` | autoregressive 텍스트 생성 |
| `mdp export` | adapter merge + 서빙용 패키징 |
| `mdp serve` | FastAPI REST 서버 |
| `mdp estimate` | GPU 메모리 추정 + 전략 추천 |
| `mdp list` | 모델/태스크/콜백/전략 카탈로그 조회 |

모든 명령은 `--format json`을 지원하며, `mdp <command> --help`로 상세 옵션을 확인할 수 있다.

## Supported Tasks

| Task | 예시 모델 |
|------|----------|
| `image_classification` | ViT, ResNet, EfficientNet, Swin |
| `object_detection` | YOLOv8, DETR |
| `semantic_segmentation` | SegFormer |
| `text_classification` | BERT, RoBERTa |
| `token_classification` | BERT (NER) |
| `text_generation` | Llama 3, Qwen 2.5, Mistral, Phi-3 |
| `seq2seq` | T5 |
| `image_generation` | (모델 내장) |
| `feature_extraction` | CLIP, DINOv2, SigLIP |

## Key Features

- **`_component_` 패턴**: 모든 컴포넌트(모델, 어댑터, 옵티마이저, 콜백, sampler 등)를 YAML에서 Python import path로 지정. 레지스트리 없이 커스텀 컴포넌트 즉시 사용
- **Length-bucketed sampling**: 가변 길이 LM 학습에서 `data.sampler`로 `LengthGroupedBatchSampler`(single-GPU) 또는 `DistributedLengthGroupedBatchSampler`(DDP/FSDP)를 선택하면 batch 내 padding 토큰을 거의 제거 (LLaMA-3 측정 53%→0.2%). 미지정 시 기존 동작 100% 보존
- **Recipe/Config 분리**: 실험 정의와 인프라 설정을 분리하여 동일 실험을 다른 환경에서 재현
- **분산 학습**: DDP, FSDP, DeepSpeed ZeRO-2/3 자동 감지 및 설정
- **RL Alignment**: DPO, GRPO, PPO를 멀티 모델(policy, reference, value, reward)로 지원
- **어댑터**: LoRA, QLoRA, Prefix Tuning 내장
- **MLflow 통합**: 학습 메트릭, 아티팩트 자동 추적. `TrainResult.checkpoints_saved`로 산출물 유무를 JSON에서 즉시 판정
- **Graceful shutdown**: `timeout` 명령이나 Ctrl+C로 중단되어도 MLflow run이 zombie로 남지 않고 `stopped_reason=signal_term|signal_int` tag와 함께 정상 마감
- **추론 콜백**: `BaseInferenceCallback`으로 hidden state, attention 등 내부 분석
- **드리프트 모니터링**: 학습 시 baseline 생성, 추론 시 분포 변화 감지

## Documentation

상세 문서는 [`docs/`](docs/) 디렉토리를 참조:

- **[Getting Started](docs/getting-started.md)** — 설치부터 첫 학습까지
- **[Configuration Guide](docs/configuration.md)** — Recipe/Config YAML 상세 가이드
- **[Training Guide](docs/training.md)** — SFT, RL, 분산 학습, 콜백
- **[Inference & Serving](docs/inference-and-serving.md)** — 배치 추론, 텍스트 생성, REST API
- **[Extending MDP](docs/extending.md)** — 커스텀 모델, 콜백, 컴포넌트 작성법
- **[Agent Integration](docs/agent-integration.md)** — AI Agent와의 연동 가이드

개발자 레퍼런스는 [AGENT.md](AGENT.md)를 참조.

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- CUDA GPU (학습 시 권장)

## License

Private
