## 현재 상태
- phase: closed (커밋 대기)
- cycle: 22
- codebase: ~/projects/modern-deeplearning-pipeline
- refactor: [[personal/projects/modern-deeplearning-pipeline/docs/refactor-component-unification]]
- next: factory-removal 리팩토링 세션 (review-22 구조 2건 + Factory→ModelBuilder)

## 산출물
- spec: [[personal/projects/modern-deeplearning-pipeline/_draft]], [[personal/projects/modern-deeplearning-pipeline/docs/spec-architecture|spec-architecture]] ~ [[personal/projects/modern-deeplearning-pipeline/docs/spec-rl|spec-rl]]
- refactor: [[personal/projects/modern-deeplearning-pipeline/docs/refactor-component-unification]]
- code-S1: [[personal/projects/modern-deeplearning-pipeline/plans/code-2026-04-07|code-2026-04-07 (S1)]] (DataSpec + Dataset/Collator wrapper + create_dataloaders) — git: ff487f3
- code-S2+S3: [[personal/projects/modern-deeplearning-pipeline/plans/code-2026-04-07|code-2026-04-07 (S2+S3)]] (ModelSpec/AdapterSpec → dict + adapter routing) — git: 0be9e0a
- code-S4: [[personal/projects/modern-deeplearning-pipeline/plans/code-2026-04-08|code-2026-04-08 (S4)]] (TrainingSpec.strategy + AutoStrategy) — git: 0be9e0a
- code-S5: [[personal/projects/modern-deeplearning-pipeline/plans/code-2026-04-08-S5|code-2026-04-08-S5]] (코드 변경 없음 — S1~S4에서 완료)
- code-S6: [[personal/projects/modern-deeplearning-pipeline/plans/code-2026-04-08-S6|code-2026-04-08-S6]] (generate + --override, 194 passed)
- code-S7: [[personal/projects/modern-deeplearning-pipeline/plans/code-2026-04-08-S7|code-2026-04-08-S7]] (fixture 5개 수정 + gpt2-dpo-preference 신규 + test_recipe_fixtures.py 신규 + init.py 번외 수정, 258 passed) — git: uncommitted
- code-S8: [[personal/projects/modern-deeplearning-pipeline/plans/code-2026-04-08-S8|code-2026-04-08-S8]] (spec-schema/data/foundation/infra + AGENT.md + weighted-mtp/mdp-integration 갱신, 코드 변경 없음)
- verify-1: [[personal/projects/modern-deeplearning-pipeline/plans/verify-2026-04-08-c1|verify-2026-04-08-c1]] (258 passed, 8 skipped, 0 failed)
- review-1: [[personal/projects/modern-deeplearning-pipeline/plans/review-2026-04-08-c1|review-2026-04-08-c1]] (버그: 3, 구조: 3, 문서갱신: 1)
- review-19: [[personal/projects/modern-deeplearning-pipeline/plans/review-2026-04-08-c19|review-2026-04-08-c19]] (0건 — S1~S3 수렴)
- review-21 (재): [[personal/projects/modern-deeplearning-pipeline/plans/review-2026-04-08-c21|review-2026-04-08-c21]] (버그 3, 구조 2)
- fix-21: [[personal/projects/modern-deeplearning-pipeline/plans/fix-2026-04-08-c21|fix-2026-04-08-c21]] (5건 전수 수정, 252 passed)
- review-22: [[personal/projects/modern-deeplearning-pipeline/plans/review-2026-04-08-c22|review-2026-04-08-c22]] (버그 0, 구조 2 — 다음 세션으로 이관)

## 다음 세션: factory-removal + 잔여 구조 개선
- B: RLTrainer scheduler interval 미사용 (review-22 2-1)
- C: generate CLI num_beams/repetition_penalty 플래그 (review-22 2-2)
- A: Factory → models/builder.py 구조 변경
- 순서: B → C → A

## 이력
- 2026-03-31 ~ 2026-04-03: spec → code → audit cycle 1~8 완료 (spec 10개 문서 안정화)
- 2026-04-04 ~ 2026-04-06: verify/review/fix cycle 9~20 (Multi-GPU Serving 구현 포함)
- 2026-04-07: refactor 모드 진입 — refactor-component-unification.md (S1~S8)
- 2026-04-07: S1 완료 — DataSpec schema + HuggingFaceDataset/ImageClassificationDataset + Collator wrapper + create_dataloaders 리팩토링
- 2026-04-07: S2+S3 완료 — ModelSpec/AdapterSpec/RLModelSpec → dict, Factory dict 기반 리팩토링, apply_adapter 제거
- 2026-04-08: S4 완료 — TrainingSpec.strategy dict, STRATEGY_MAP → aliases.yaml, auto_strategy 함수 (198 passed, 7 env-only failed)
- 2026-04-08: S5 완료 — 코드 변경 없음 (validator 갱신은 S1~S4에서 완료)
- 2026-04-08: S6 완료 — mdp generate 커맨드 + --override 옵션 (4개 커맨드). 신규 3파일, 수정 7파일, 테스트 22건 (194 passed)
- 2026-04-08: S7 완료 — tests/fixtures 5개 수정 + gpt2-dpo-preference 신규 + test_recipe_fixtures.py 6개 parametrize suite 신규 + init.py 번외 수정 (258 passed, 8 skipped)
- 2026-04-08: S8 완료 → verify 진입 (cycle 1) — 전 Stage(S1~S8) 완료, spec 문서 6개 동기화
- 2026-04-08: verify-1 완료 (258 passed, 0 failed) → review 진입
- 2026-04-08: review-1 완료 (버그 3, 구조 3, 문서 1) → fix 진입
- 2026-04-08: cycle 19 수렴 (S1~S3 영역 0건) → cycle 20-21 오염(§3.5 미인식) → cycle 21 재실행 (5건) → fix-21 (5건 수정)
- 2026-04-08: review-22 완료 (버그 0, 구조 2) — 구조 2건은 기존 한계, 다음 리팩토링 세션으로 이관. 현 cycle closed
