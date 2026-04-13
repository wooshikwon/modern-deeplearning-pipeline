# Agent Integration

AI Agent(Claude, GPT 등)가 MDP CLI를 자율적으로 사용하는 방법을 안내한다.

## 개요

MDP는 AI Agent와의 연동을 일급 시민으로 지원한다:

- **`--format json`**: 모든 명령에서 구조화된 JSON 출력 (stdout=JSON, stderr=로그)
- **`mdp list`**: 프로그래매틱 카탈로그 탐색
- **`mdp init`**: 자동 스캐폴딩 + 템플릿 생성
- **에러 스키마**: 타입별 에러 분류로 자동 복구 지원

## Agent Discovery Flow

Agent가 MDP를 처음 사용할 때 권장하는 탐색 순서:

```bash
# 1. 지원 태스크 확인
mdp list tasks --format json

# 2. 태스크에 맞는 모델 탐색
mdp list models --task text_generation --format json

# 3. 프로젝트 스캐폴딩
mdp init my_project --task text_generation --model llama3-8b

# 4. Recipe의 ??? 필드 채우기 (data.dataset.source, head.num_classes 등)

# 5. GPU 메모리 추정 → 전략 결정
mdp estimate -r recipe.yaml --format json

# 6. 학습 실행
mdp train -r recipe.yaml -c config.yaml --format json

# 7. 결과 분석 및 추론
mdp inference --run-id <id> --data test.jsonl --format json
```

### mdp list 출력 예시

```bash
mdp list tasks --format json
```
```json
{
  "status": "success",
  "command": "list",
  "tasks": [
    {"name": "text_generation", "required_fields": ["text"]},
    {"name": "image_classification", "required_fields": ["image", "label"]},
    ...
  ]
}
```

```bash
mdp list models --task text_generation --format json
```
```json
{
  "status": "success",
  "command": "list",
  "models": [
    {
      "name": "llama3-8b",
      "family": "llama",
      "memory": {"fp16_gb": 16.1, "qlora_4bit_gb": 5.5},
      "pretrained_sources": ["hf://meta-llama/Meta-Llama-3-8B"]
    },
    ...
  ]
}
```

## JSON 출력 스키마

### 공통 래퍼

```json
{
  "status": "success | error",
  "command": "train | inference | estimate | ...",
  "timestamp": "ISO-8601",
  "error": null
}
```

### 에러 응답

```json
{
  "status": "error",
  "command": "train",
  "error": {
    "type": "ValidationError",
    "message": "Task 'text_generation' requires field 'text' but got ['image']",
    "details": {
      "validation_stage": "business",
      "errors": [...]
    }
  }
}
```

### 명령별 결과 필드

**Train/RL-Train**:
```json
{
  "run_id": "abc123",
  "checkpoint_dir": "./checkpoints/...",
  "metrics": {"val_loss": 0.12, "val_accuracy": 0.95},
  "total_epochs": 27,
  "total_steps": 12600,
  "stopped_reason": "early_stopped",
  "duration_seconds": 3600.5,
  "monitoring": {"baseline_saved": true}
}
```

**Inference**:
```json
{
  "run_id": "abc123",
  "output_path": "./output/predictions.parquet",
  "task": "text_generation",
  "evaluation_metrics": {"accuracy": 0.95, "f1": 0.93},
  "monitoring": {"drift_detected": false}
}
```

**Estimate**:
```json
{
  "model_mem_gb": 16.1,
  "gradient_mem_gb": 32.2,
  "optimizer_mem_gb": 64.4,
  "activation_mem_gb": 8.0,
  "total_mem_gb": 120.7,
  "suggested_gpus": 4,
  "suggested_strategy": "fsdp"
}
```

## 에러 복구 전략

Agent가 에러를 받았을 때의 복구 패턴:

| error.type | 원인 | 자동 복구 |
|-----------|------|----------|
| `ValidationError` | YAML 설정 오류 | Recipe/Config 수정 후 재시도 |
| `RuntimeError` (OOM) | GPU 메모리 부족 | batch_size 축소, precision 변경, 분산 전략 변경 |
| `RuntimeError` (NCCL) | 분산 통신 오류 | GPU 수 조정, 전략 변경 |

### OOM 복구 예시

```python
# Agent 로직 (의사 코드)
result = run("mdp train -r recipe.yaml -c config.yaml --format json")

if result["status"] == "error" and "CUDA out of memory" in result["error"]["message"]:
    # 1차: batch_size 절반
    result = run("mdp train ... --override data.dataloader.batch_size=2 --format json")

    if still_oom:
        # 2차: gradient_accumulation 2배 + precision 변경
        result = run("mdp train ... --override training.precision=bf16 "
                     "--override training.gradient_accumulation_steps=8 --format json")

    if still_oom:
        # 3차: QLoRA 적용
        # Recipe YAML 수정하여 adapter 섹션 추가
```

## 자율 학습 루프

Agent가 반복적으로 실험을 최적화하는 패턴:

```python
# 1. 초기 추정
estimate = run("mdp estimate -r recipe.yaml --format json")

# 2. Config 조정
config = adjust_config(estimate)

# 3. 학습
train_result = run(f"mdp train -r recipe.yaml -c config.yaml --format json")

# 4. 평가
eval_result = run(f"mdp inference --run-id {train_result['run_id']} "
                  f"--data val.jsonl --metrics Accuracy F1Score --format json")

# 5. 결과 분석 → 하이퍼파라미터 조정 → 재학습
if eval_result["evaluation_metrics"]["accuracy"] < target:
    # learning rate, epochs, batch_size 등 조정
    ...
```

## --override로 실험 변주

Agent가 YAML 파일을 직접 수정하지 않고 CLI override로 실험을 변주할 수 있다:

```bash
# 기본 학습
mdp train -r recipe.yaml -c config.yaml --format json

# learning rate sweep
mdp train -r recipe.yaml -c config.yaml --override optimizer.lr=1e-4 --format json
mdp train -r recipe.yaml -c config.yaml --override optimizer.lr=5e-5 --format json

# batch size 변경
mdp train -r recipe.yaml -c config.yaml --override data.dataloader.batch_size=16 --format json

# 짧은 테스트 실행
mdp train -r recipe.yaml -c config.yaml --override training.epochs=0.1 --format json
```

## AGENT.md

코드베이스 루트의 [`AGENT.md`](../AGENT.md)는 CLI 레퍼런스와 YAML 스키마를 포함하는 개발자 가이드다. Agent가 MDP를 사용할 때 이 파일을 참조하면 모든 옵션과 제약 조건을 확인할 수 있다.

## 역할 경계

| 영역 | 사용자/Agent 책임 | MDP 책임 |
|------|------------------|----------|
| 데이터 | 수집, 전처리, JSONL/CSV/HF 저장 | 로드, 토큰화, augmentation |
| 모델 | 커스텀 BaseModel 작성 (필요 시) | pretrained 로드, head 교체, adapter 적용 |
| 인프라 | GPU 프로비저닝 (SSH, 클라우드, K8s) | GPU 감지, torchrun 분산 |
| 학습 | Recipe/Config YAML 작성 | 검증, 학습 루프, AMP, MLflow |
| 서빙 | LB, autoscale, DNS | FastAPI (스트리밍 + 배칭) |
| 추론 | 결과 후처리, 비즈니스 로직 | 배치 추론, 생성, 드리프트 감지 |
