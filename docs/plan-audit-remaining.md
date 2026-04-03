# 감사 잔여 항목 구현 계획

**기준**: `docs/audit-2026-04-03.md`의 P1/P2/P3 권장 액션 + /coding 위반
**선행 계획**: `docs/plan-audit-discussion-items.md` (논의 10건)
**이 문서**: 논의 항목을 제외한 나머지 코드 개선 항목

---

## 감사 정정 사항

코드베이스 재검증 결과 6건이 부정확. audit 문서에서 수정 필요.

| 항목 | 감사 기록 | 실제 | 조치 |
|------|----------|------|------|
| P2-3 | SequenceClassificationHead가 TASK_HEAD_COMPAT에 누락 | 해당 클래스가 코드베이스에 존재하지 않음. aliases.yaml에만 등록 | 감사에서 삭제 |
| /coding #13 | alias 해석 전 비교 (SequenceClassificationHead 예시) | 위와 동일. 예시 무효 | 감사에서 삭제 |
| P2-6, /coding #1 | _generation_kwargs가 train() 전에 사용 | train() 내에서 L500 설정 → L553 사용. 정상 흐름에서 문제 없음 | 감사에서 심각도 하향 → 방어적 초기화 선택 사항 |
| /coding #12 | serving/model_loader.py vs server.py tokenizer 중복 | model_loader.py에 tokenizer 로딩 없음 | 감사에서 삭제 |
| 이전 잔여: InferenceResult monitoring 미사용 | inference.py L207-237에서 조건부 계산/전달 완료 | 감사에서 삭제 |
| 이전 잔여: embedding centroid 미사용 | compute_baseline → compare_baselines로 완전 연결 | 감사에서 삭제 |

추가 발견:
- 감사 항목 6(RLTrainer validation 부재)도 부정확 — rl_trainer.py L547-554에서 이미 호출 중. plan-audit-discussion-items.md에 정정 반영 완료.

---

## 구현 항목 정리

### Phase A — 단순 수정 (병렬 가능)

#### A-1. [P1-1] RLTrainer EP 통합

**파일**: `training/rl_trainer.py`
**검증 결과**: CONFIRMED — L115에서 EP 생성, train()에서 미사용

SFT Trainer의 EP 패턴(trainer.py L301-313, L262-292)을 RLTrainer에 적용:

1. `train()` 시작부(L462-479)에 EP setup:
```python
if self.expert_parallel is not None:
    if self.strategy is not None and not dist.is_initialized():
        _dist.init_process_group(backend=...)
    self.policy = self.expert_parallel.setup(self.policy, self.device)
```

2. `_fire()` (L190-198)에 EP gather/scatter (checkpoint hook 전후)

**변경 범위**: ~20줄

#### A-2. [P1-2] EMA + MLflow export 순서

**파일**: `training/trainer.py`
**검증 결과**: CONFIRMED — L402(export, mlflow_ctx 내) → L404(on_train_end, mlflow_ctx 밖)

`on_train_end`를 `with mlflow_ctx:` 블록 안, `_log_mlflow_summary` 직전으로 이동:

```python
# finally 블록 내, strategy cleanup 후
self._fire("on_train_end", metrics=self.last_metrics)  # EMA 복원
training_duration = time.time() - start_time
if self._is_main_process:
    self._log_mlflow_summary(training_duration, stopped_reason)  # 모델 export
```

**변경 범위**: 1줄 이동

#### A-3. [P3-9] format 주석 수정

**파일**: `settings/schema.py:64`
**검증 결과**: CONFIRMED

```python
format: str = "auto"  # auto | csv | json | parquet | imagefolder
```

#### A-4. [P3-10] column_map dead field 제거

**파일**: `settings/schema.py:67`
**검증 결과**: CONFIRMED — 전체 codebase에서 미사용

`column_map` 필드 삭제. test fixture에 정의되어 있다면 함께 제거.

#### A-5. [P3-11] serving_meta.json 생성 제거

**파일**: `cli/export.py:95-101`, `training/trainer.py:761-767`, `training/rl_trainer.py:356-362`
**검증 결과**: CONFIRMED — 3곳에서 생성, 0곳에서 읽음

3곳 모두 삭제.

#### A-6. [P3-12] apply_lora 도달 불가능 기본값 수정

**파일**: `models/adapters/lora.py`
**검증 결과**: CONFIRMED — schema dropout=0.0이 항상 우선, apply_lora의 0.05는 도달 불가

기본값을 0.0으로 통일하거나 기본값 제거.

---

### Phase B — 중규모 수정 (병렬 가능)

#### B-1. [P2-5] Trainer 공통 코드 추출

**파일**: 신규 `training/_common.py`, `trainer.py`, `rl_trainer.py`
**검증 결과**: CONFIRMED — 3개 메서드 완전 동일 복제

추출 대상:
| 메서드 | trainer.py | rl_trainer.py | 비고 |
|--------|-----------|--------------|------|
| `_detect_device` | L109-115 | L130-136 | 완전 동일 |
| `_create_strategy` | L210-234 | L144-162 | 미세 차이 (none/auto 검사) |
| `_create_expert_parallel` | L236-251 | L164-179 | 완전 동일 |

`training/_common.py`에 모듈 레벨 함수로 추출. `STRATEGY_MAP`도 함께 이동.

#### B-2. [P2-7] streaming + multimodal 가드

**파일**: `data/loader.py:69-94`
**검증 결과**: CONFIRMED — L94 set_transform이 IterableDataset에서 미지원

multimodal 경로 시작부에 가드:
```python
if streaming:
    raise ValueError(
        "streaming=True와 multimodal(vision+language) 조합은 지원되지 않습니다."
    )
```

#### B-3. [P2-8] tokenizer 인스턴스 공유

**파일**: `data/dataloader.py`, `data/tokenizer.py`
**검증 결과**: CONFIRMED — tokenizer.py:105와 dataloader.py:45에서 이중 로드

`create_dataloaders`에서 tokenizer를 한 번만 생성 → `build_tokenizer`와 `_select_collator`에 전달.

#### B-4. [P3-13] chat_template 경고

**파일**: `data/tokenizer.py:117-129`
**검증 결과**: CONFIRMED

```python
if chat_template:
    if "messages" not in examples:
        logger.warning("chat_template이 설정되었으나 'messages' 컬럼 없음. 무시됩니다.")
    else:
        # 기존 chat_template 처리
```

#### B-5. [P3-15] baseline.py 호출 패턴 통일

**파일**: `monitoring/baseline.py:248`
**검증 결과**: CONFIRMED — L172 `model(batch)` vs L248 `model(**batch)`

L248을 `model(model_batch)`로 변경. MDP forward 계약은 positional dict.

---

### Phase C — /coding 위반 정리

#### C-1. 흐름 단절

| # | 위치 | 내용 | 수정 | 검증 |
|---|------|------|------|------|
| #2 | rl_trainer.py:624 | old_log_probs가 _forward_model 우회 | `_forward_model(self.policy, gen_input, role="policy")` 사용. 수정 후 static `_extract_logits`(L731-737)의 호출처를 확인하여 미사용이면 삭제 | CONFIRMED |
| #3 | rl_trainer.py:728 | zero_grad 전체 optimizer에 적용 | `for name in losses:` 범위로 제한 | CONFIRMED |
| #5 | tokenizer.py:129 | text/input 없으면 빈 리스트 | `if not texts: raise ValueError(...)` | CONFIRMED |
| #6 | inference.py:124 | private 함수 import | `_load_source`, `_rename_columns`를 public으로 승격 | CONFIRMED |
| #7 | rl_trainer.py:536 vs 718 | callback/optimizer step 타이밍 불일치 | 콜백 조건을 `(global_step + 1) % accum == 0`으로 통일 | CONFIRMED |

#### C-2. 기존 패턴 불일치

| # | 위치 | 내용 | 수정 | 검증 |
|---|------|------|------|------|
| #14 | factory.py:166-167 | BusinessValidator 정적 메서드 직접 호출 | `validate_partial(settings, checks=[...])` 공개 API 추가 | CONFIRMED |

#### C-3. 누락

| # | 위치 | 내용 | 수정 | 검증 |
|---|------|------|------|------|
| #17 | loader.py:24-28 | streaming에서 remove_columns 건너뜀 | IterableDataset에서도 remove_columns 호출 추가. set_format 스킵은 유지 | CONFIRMED (minor) |
| #19 | serve.py:82-85 | ServeResult에 run_id 미전달 | `build_result(..., run_id=run_id)` 추가 | CONFIRMED |
| #20 | deepspeed.py:96-114 | setup_models에서 dist 초기화 없음 | 현상 유지 + 주석. deepspeed.initialize가 내부 처리 | CONFIRMED (by design) |

#### C-4. 암묵적 의존 + 과잉 구현

| # | 위치 | 내용 | 수정 | 검증 |
|---|------|------|------|------|
| #22 | schema.py:171-188 | ComputeConfig 미검증 필드 14개 | 현상 유지. docstring에 "원격 실행 미구현" 표시 | CONFIRMED |
| #24 | adapters/__init__.py:15 | PEFT 전용 메서드 의존 | `hasattr` 가드 추가, 없으면 수동 계산 fallback | CONFIRMED |

#### C-5. 04-02 감사 잔여 (신규 계획에 누락되었던 항목)

| # | 위치 | 내용 | 수정 | 출처 |
|---|------|------|------|------|
| 04-02 #7 | business_validator.py | vision task + tokenizer 경고 미구현 | `_check_task_fields`에 VISION_TASKS 검사 추가 | audit-2026-04-02 권장 #10 |

---

### Phase D — 문서 업데이트

| 대상 | 내용 |
|------|------|
| audit-2026-04-03.md | 위 정정 사항 6건 반영 |
| 설계 문서 04 | SettingsFactory "3개 파이프라인" → "4개 메서드" |
| 설계 문서 05 | QLoRA + head "무시" 동작 명시 |
| 설계 문서 07 | FSDP frozen NO_SHARD 허용 |
| AGENT.md | DualEncoderHead forward_pair() 가이드 (plan-discussion-items #4) |

---

## 구현 순서 종합

```
Phase A (병렬):  A-1 ~ A-6         ← 단순, 독립적
Phase B (병렬):  B-1 ~ B-5         ← 중규모, 독립적
Phase C (순차):  C-1 → C-2 → C-3 → C-4  ← /coding 위반
Phase D:         문서 업데이트       ← 코드 변경 완료 후
```

plan-audit-discussion-items.md의 10건과 합치면 전체 구현 순서:
1. **P1 즉시 수정** (병렬): EP 통합(A-1), EMA 순서(A-2) — 기능 정확성 최우선
2. **단순 수정** (병렬): A-3~A-6 + Discussion #1+2(compat_validator), #3(Seq2SeqLMHead alias)
3. **중규모 — 설정/검증** (병렬): Discussion #5(val_split), #10(BatchScheduler), B-2(streaming 가드)
4. **중규모 — 학습 코어**: Discussion #7(콜백 critical) → B-1(공통 코드 추출) → Discussion #9(PPO accum)
5. **중규모 — 데이터/서빙**: B-3(tokenizer 공유), B-4(chat_template 경고), B-5(baseline 통일)
6. **RL 통합**: Discussion #6(DPO validation 분기) + Discussion #8(step-level resume)
7. **/coding 위반 정리**: Phase C 전체
8. **문서**: Phase D

## 검증 전략

- Phase A/B 완료 후: `pytest tests/ -x` 전체 regression
- Phase C 각 항목: 관련 테스트 개별 실행
- Phase D: 코드 변경 없음
