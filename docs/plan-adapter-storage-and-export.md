# Adapter 저장 전략 변경 + `mdp export` 명령어 추가

## 배경

현재 학습 완료 시 MLflow artifact에 **merge된 전체 모델**을 저장한다. LoRA adapter + base model을 merge_and_unload()로 합치고 전체 가중치(~16GB)를 기록하는 방식.

문제:
- 8B 모델 기준 artifact 16GB → 실험 N개 = 16GB × N
- merge하면 adapter 재사용 불가 (다른 base에 붙이기, multi-LoRA 서빙, 다음 phase의 입력)
- base model이 HF 캐시에 이미 있는데 전체를 중복 저장

변경 방향: **저장은 경량(adapter만), merge는 on-demand(export/serve 시점).**

---

## 변경 범위

### 1. MLflow artifact 저장 변경 (`trainer.py:_export_and_log_model`)

**현재**: checkpoint → reconstruct → merge_and_unload → 전체 모델 저장 → MLflow
**변경**: checkpoint → adapter/가중치 + recipe + tokenizer + serving_meta → MLflow (merge 없음)

```
현재 artifact:                    변경 후 artifact:
model/                            model/
  model.safetensors  (16GB)         adapter_model.safetensors  (50MB, LoRA)
  config.json                       adapter_config.json
  tokenizer.json                    config.json  (base model config)
  recipe.yaml                       tokenizer.json
  serving_meta.json                 recipe.yaml
                                    serving_meta.json
```

full finetuning (adapter 없음) 시에는 현재와 동일하게 전체 모델 저장.

**수정 파일**: `mdp/training/trainer.py` — `_export_and_log_model` 메서드
**핵심 변경**: `merge_and_unload()` 호출 제거. 대신 checkpoint의 파일을 그대로 복사.

### 2. `mdp export` CLI 명령어 추가

adapter artifact 또는 checkpoint에서 merge된 전체 모델을 만드는 명령어.

```bash
# MLflow run에서
mdp export --run-id <id> --output ./deploy/

# 특정 checkpoint에서
mdp export --checkpoint ./checkpoints/checkpoint-1000/ --output ./deploy/
```

두 입력 모두 같은 처리:
1. recipe.yaml 읽기 → base model pretrained 경로 확인
2. base model 로딩 (HF 캐시 또는 다운로드)
3. adapter 있으면 적용 → merge_and_unload()
4. 전체 모델 + tokenizer + recipe + serving_meta를 --output에 저장
5. full finetuning이면 merge 건너뛰고 가중치 그대로 복사

**새 파일**: `mdp/cli/export.py`
**수정 파일**: `mdp/__main__.py` — export 명령어 등록

### 3. `mdp serve` 변경 — `--model-dir` 옵션 추가

현재: `--run-id`만 지원. 내부에서 reconstruct_model로 모델 로딩.
변경: `--run-id` (on-demand merge) + `--model-dir` (이미 export된 디렉토리) 둘 다 지원.

```bash
# 개발: MLflow에서 직접 (adapter → on-demand merge)
mdp serve --run-id abc123

# 프로덕션: export된 디렉토리에서 (이미 merge됨)
mdp serve --model-dir ./deploy/
```

**수정 파일**: `mdp/cli/serve.py`, `mdp/__main__.py`
**핵심**: `--run-id`일 때 adapter 감지 → merge → 서빙. `--model-dir`일 때 그대로 로딩.

### 4. model_loader.py 수정

`reconstruct_model`이 현재 merge된 전체 모델을 기대한다. adapter-only artifact도 처리하도록 변경.

**수정 파일**: `mdp/serving/model_loader.py`
**핵심**: adapter_config.json 존재 시 → base model 로딩 + PeftModel.from_pretrained

### 5. `mdp inference` 변경

serve와 동일한 패턴. `--run-id`에서 adapter artifact를 받으면 on-demand merge.

**수정 파일**: `mdp/cli/inference.py`

---

## 수정 파일 요약

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `mdp/cli/export.py` | **신규** | `run_export()` — merge + 패키징 |
| `mdp/__main__.py` | 수정 | export 명령어 추가, serve에 --model-dir 옵션 |
| `mdp/training/trainer.py` | 수정 | `_export_and_log_model`에서 merge 제거, adapter 그대로 저장 |
| `mdp/cli/serve.py` | 수정 | `--model-dir` 옵션, adapter on-demand merge |
| `mdp/cli/inference.py` | 수정 | adapter on-demand merge |
| `mdp/serving/model_loader.py` | 수정 | adapter-only artifact 로딩 지원 |

---

## 구현 순서

### Step 1: model_loader.py — adapter-aware reconstruct

adapter_config.json이 있으면 base model + adapter 로딩하는 경로 추가. 이 변경이 나머지 모든 작업의 기반.

### Step 2: trainer.py — artifact 저장 변경

`_export_and_log_model`에서 merge 제거. LoRA면 adapter 파일만, full이면 전체 모델 저장.

### Step 3: export.py — 새 CLI 명령어

`--run-id` 또는 `--checkpoint` → model_loader로 모델 복원 → merge → 저장.

### Step 4: serve.py — --model-dir 추가

`--model-dir`이면 로컬 디렉토리에서 직접 로딩. `--run-id`면 기존 흐름 + adapter merge.

### Step 5: inference.py — adapter merge 경로

serve와 동일한 패턴 적용.

### Step 6: 테스트

- `test_export_e2e.py`: LoRA 학습 → export → merge된 모델 로딩 검증
- `test_serve_endpoint.py` 확장: --model-dir 경로 검증
- `test_mlflow_artifacts.py` 확장: adapter-only artifact 저장 검증

---

## 영향 범위

### 하위 호환성

- 기존 merge된 artifact를 가진 MLflow run은 **model_loader가 양쪽 모두 처리**하므로 호환.
- `mdp serve --run-id`는 기존과 동일하게 동작 (내부에서 adapter 감지 → merge).

### AGENT.md 업데이트

- Commands 테이블에 `mdp export` 추가
- `mdp serve`에 `--model-dir` 옵션 문서화
- CLI Output 섹션에 export JSON 응답 추가

### 테스트 영향

- `test_mlflow_artifacts.py`: artifact 내용 검증 변경 (model.safetensors → adapter_model.safetensors)
- `test_checkpoint_recipe.py`: 영향 없음 (checkpoint 저장은 변경 없음)
