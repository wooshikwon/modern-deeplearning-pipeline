# `mdp rl-train` 탐색 계획

## 목표

MDP에 `mdp rl-train` 명령어를 추가하여 LLM/VLM alignment 학습을 지원한다. DPO, weighted-NTP, GRPO, PPO 등의 알고리즘을 YAML 설정으로 실행할 수 있게 한다. 기존 `mdp train` (SFT)과 완전히 독립된 경로이며, 학습 결과는 기존 `mdp export` / `mdp serve` / `mdp inference`로 그대로 서빙·추론한다.

## 탐색이 필요한 이유

RL 학습은 SFT와 근본적으로 다른 학습 루프를 가진다. 구현 전에 다음을 확인해야 한다:

1. **기존 코드에서 재사용 가능한 경계가 어디인지** — Trainer를 확장하는 건지, 완전히 새로 만드는 건지
2. **multi-model FSDP/DDP가 기술적으로 가능한지** — frozen 모델과 trainable 모델에 다른 sharding 정책 적용
3. **TRL 등 기존 라이브러리의 핵심 로직이 무엇인지** — 참고하되 의존하지 않기 위해 핵심을 이해
4. **알고리즘 간 공통 추상화가 무엇인지** — DPO, PPO, GRPO, weighted-NTP의 공통점과 차이점

## 탐색 항목

### 1. 기존 MDP 코드 경계 분석

**목적**: RLTrainer가 기존 Trainer에서 무엇을 가져오고 무엇을 새로 만들지 결정.

**탐색 방법**:
- `trainer.py` (801줄)를 블록별로 분해: `__init__` 설정, `train()` 루프, `_train_one_epoch()`, MLflow, resume, validation, callbacks
- 각 블록이 RL에서도 필요한지, 수정 없이 재사용 가능한지, 새로 구현해야 하는지 분류

**핵심 질문**:
- AMP, gradient accumulation, gradient clipping → RL에서도 동일하게 필요. 재사용 가능한가?
- `_create_optimizer`, `_create_scheduler` → 재사용 가능한가?
- `train()` 루프 구조 → RL은 에폭 기반이 아니라 step 기반. 어떻게 다른가?
- MLflow 로깅 → 재사용 가능. params/metrics가 다를 뿐.
- callbacks (EarlyStopping, ModelCheckpoint) → policy 모델에만 적용하면 재사용 가능한가?
- Strategy → frozen 모델에 다른 정책 적용이 필요. 현재 setup()이 1개 모델만 받는 구조.

### 2. Multi-model FSDP/DDP 기술 검증

**목적**: 같은 process group 안에서 trainable 모델과 frozen 모델에 다른 sharding 전략을 적용할 수 있는지 확인.

**탐색 방법**:
- PyTorch 문서에서 `FSDP(model_a)` + `FSDP(model_b, NO_SHARD)` 동시 사용 사례 확인
- TRL의 multi-model 처리 방식 조사 (소스 코드 또는 문서)
- CPU gloo에서 minimal 테스트: 2개 모델을 다른 FSDP 정책으로 래핑하고 forward/backward

**핵심 질문**:
- 한 process group에서 FSDP 모델 2개 공존 가능한가?
- frozen 모델에 `requires_grad=False` + `NO_SHARD`이면 메모리/통신 최적화가 되는가?
- reference 모델을 policy와 weight sharing하는 방식 (TRL의 `ref_model=None` 패턴)은 가능한가?

### 3. TRL 핵심 로직 분석

**목적**: TRL을 의존성으로 추가하지 않되, 핵심 알고리즘 구현을 참고.

**탐색 방법**:
- TRL GitHub 소스에서 DPOTrainer, PPOTrainer의 `training_step` 핵심 로직 추출
- log_prob 계산, KL divergence, advantage estimation의 구현 패턴
- generation 루프 (PPO/GRPO에서 policy가 텍스트를 생성하는 부분)

**핵심 질문**:
- DPO의 핵심이 몇 줄인가? (loss 함수 자체는 ~20줄일 것)
- PPO에서 generation + reward + advantage + policy update 루프가 어떤 단위로 구성되는가?
- 알고리즘 간 공유 가능한 유틸 함수는? (log_prob 계산, KL divergence, advantage normalization 등)

### 4. 알고리즘별 요구사항 매트릭스

**목적**: RLTrainer의 추상화 수준 결정. 너무 범용적이면 복잡하고, 너무 구체적이면 알고리즘 추가가 어렵다.

**탐색 방법**:
- 4개 알고리즘 (DPO, weighted-NTP, GRPO, PPO)의 학습 루프를 단계별로 분해
- 공통 단계와 알고리즘 고유 단계를 식별
- 공통 부분 → RLTrainer 기반 클래스, 고유 부분 → 알고리즘별 서브클래스 (또는 전략 패턴)

예상 매트릭스:

```
         | Generation | Ref Log-probs | Reward Model | Value Model | KL penalty |
---------|------------|---------------|-------------|-------------|------------|
DPO      |     X      |       O       |      X      |      X      |     O      |
w-NTP    |     X      |       X       |      X      |   O(critic) |     X      |
GRPO     |     O      |       O       |      O      |      X      |     O      |
PPO      |     O      |       O       |      O      |      O      |     O      |
```

### 5. Recipe 스키마 설계

**목적**: `rl-recipe.yaml`이 어떤 구조여야 하는지.

**탐색 방법**:
- 4개 알고리즘의 필수 설정을 나열
- 기존 Recipe 스키마와의 공통점/차이점
- `models` (복수) 필드 구조 결정: 역할(policy, reference, value, reward/critic)별 설정

**핵심 질문**:
- `models.policy`와 `models.reference`가 같은 pretrained를 가리킬 때 가중치를 공유할 수 있는가?
- algorithm별 config (dpo.beta, ppo.clip_range 등)를 어떤 구조로 넣을 것인가?
- 기존 `training` 섹션 (epochs, precision, grad_accum 등)은 재사용 가능한가?

## 탐색 순서

| 순서 | 항목 | 의존 | 산출물 |
|:---:|------|------|--------|
| 1 | 기존 코드 경계 분석 | 없음 | 재사용/신규 분류표 |
| 2 | TRL 핵심 로직 분석 | 없음 | 알고리즘별 핵심 코드 (~100줄 수준으로 정제) |
| 3 | 알고리즘 매트릭스 | #2 | 공통/고유 단계 분류 |
| 4 | Multi-model FSDP 검증 | #1 | 기술 가능 여부 + 구현 패턴 |
| 5 | Recipe 스키마 설계 | #1, #3 | rl-recipe.yaml 초안 |

탐색 결과를 `docs/plan-rl-train.md`에 구체적 구현 계획으로 정리한다.
