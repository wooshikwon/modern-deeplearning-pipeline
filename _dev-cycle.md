## 현재 상태
- phase: code (대기)
- spec: dev-cycle/spec/spec-semantic-module-routing.md
- unit: U1 (다음 실행)
- codebase: ~/Desktop/wooshikwon/modern-deeplearning-pipeline

## 산출물
- **spec-semantic-module-routing** (작성 중): dev-cycle/spec/spec-semantic-module-routing.md
  - Factory proxy 검토 결과: **신규 Factory 불필요**. 독립 stateless 모듈 `mdp/models/family_routing.py`가 semantic → actual 번역 서비스. Factory는 소비자.
  - U1~U4: family_routing 신규 → adapter 소비자 갱신 → Factory head slot → catalog 전환
- spec: [[personal/archive/modern-deeplearning-pipeline/dev-cycle/spec/spec-pretrained-compat|spec-pretrained-compat]]
- design: [[personal/archive/modern-deeplearning-pipeline/docs/architecture|아키텍처]], [[personal/archive/modern-deeplearning-pipeline/docs/model|모델]], [[personal/archive/modern-deeplearning-pipeline/docs/data|데이터]], [[personal/archive/modern-deeplearning-pipeline/docs/training|학습]], [[personal/archive/modern-deeplearning-pipeline/docs/infra|인프라]]
- (cycle 9~16: archive/cycle-9..16/ — 버그 14→0 수렴, multi-GPU serving, import-linter 해소)
- spec-component-unification: archive/spec-component-unification/ (U2~U6 코드 리포트 4개, 5개 영역 _component_ 통일 + CLI 2건, spec 흡수 완료)
- (cycle 17~19: archive/cycle-17..19/ — U1-U3 review 수렴, fix-18 5건 전수)
- (cycle 20-21 오염 리포트: archive/contaminated-c20-c21/)
- (cycle 21~23: archive/cycle-21..23/ — 252 pass, 버그 3→1→0 수렴, doc-refine 완료)
- spec-inference-callbacks: archive/spec-inference-callbacks/ (U1~U3, --pretrained 3-way + --callbacks + BaseInferenceCallback, 287 pass, spec 흡수 완료)
- (cycle 1~3: archive/cycle-1..3/ — 버그 1→1→0 수렴, 283→287 pass, doc-refine: architecture/infra/training/usage-scenarios 갱신)
- spec-pretrained-compat: archive/spec-pretrained-compat/ (U1~U3, forward 어댑터 + config.architectures 직접 로드 + metadata 보존, 323 pass, spec 흡수 완료)
- (cycle 1~2: archive/cycle-1..2/ — 버그 1→0 수렴, --task 삭제 + config.architectures, doc-refine: model/architecture/training/infra/usage-scenarios 갱신)

## 이력
- 2026-04-04~07: cycle 9~16 — 버그 수렴 14→0, multi-GPU serving, import-linter 해소, spec 3건 동기화. done 선언
- 2026-04-07: cycle 17 spec 진입 — U1(DataSpec), U2(ModelSpec→dict), U3(adapter→_component_) 완료. review cycle 17~19 수렴
- 2026-04-08: U4(Strategy aliases.yaml 이관) + U5(변경 없음) + U6(generate CLI + --override) 완료
- 2026-04-08: **§3.5 오염 사건** — U4 code가 spec §3.5 번복 결정을 무시하고 recipe.training.strategy 구현. 이후 세션에서 Config 단일 경로로 복원했으나 미커밋. review-20/fix-20/review-21이 잘못된 전제로 작성됨
- 2026-04-08: §3.5 복원 + U7(fixtures/tests alias 정규화) + U8(AGENT.md 동기화) 커밋 (bd7307d). 오염 리포트 3건 archive. cycle 21 review 재시작
- 2026-04-08: review-21 재실행 완료 (252 pass / 버그 3, 구조 2) → fix 완료 (5건 전수: generate.py reconstruct_model 교체, GenerationSpec 7/7 필드, compat_validator "auto" 기본값 통일, RLTrainer scheduler 3키 pop + val int cast) → review 진입 (cycle 22)
- 2026-04-08: review-22 완료 (252 pass / 버그 1) → fix 완료 (1건: RLTrainer scheduler interval per-model 저장 + step/epoch 분기 5개 지점) → review 진입 (cycle 23)
- 2026-04-08: review-23 완료 (252 pass / 0건) — U4-U8 사이클 수렴
- 2026-04-08: doc-refine 완료 — spec 흡수(spec-training/spec-rl/spec-model/usage-scenarios 갱신) → spec-component-unification.md 삭제. phase=done. Unit 리포트 4건 + cycle 17-19 archive
- 2026-04-09: spec-inference-callbacks — U1~U3 완료(--pretrained 3-way, --callbacks 4커맨드, BaseInferenceCallback). cycle 1~3 수렴(버그 1→1→0, 283→287 pass). doc-refine 완료(architecture/infra/training/usage-scenarios 갱신, 콜백 아키텍처·시나리오 9-10 추가). phase=done
- 2026-04-09: spec-pretrained-compat 완료 → code 진입 (U1~U3, Group1: U1+U2 병렬)
- 2026-04-10: spec-pretrained-compat — U1~U3 완료(forward 어댑터, Auto 클래스→config.architectures 직접 로드, metadata 보존). cycle 1~2 수렴(버그 1→0, 323 pass, --task 삭제). doc-refine 완료(model/architecture/training/infra/usage-scenarios 갱신). phase=done
- 2026-04-13: spec-semantic-module-routing 초안 작성 — 드리프트 사건 3건(catalog 5개 target_modules 불일치, `"all_linear"` default 오류, Mixtral expert Parameter 전환)이 동일 구조적 원인(프레임워크 내부 이름 노출)에서 비롯. 추상화 층위 재정립. Factory proxy layer 검토 결과 **신규 Factory 필요 없음** — 독립 모듈 `family_routing.py`가 stateless 번역 서비스 제공. U1~U4 4단계. phase=code 진입 대기
