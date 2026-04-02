# GRPO/PPO 프로덕션 갭

DPO는 프로덕션 준비 수준이다. GRPO/PPO는 핵심 흐름이 동작하지만, 실전 대규모 학습에 필요한 5가지가 미구현이다.

---

## 1. K개 응답 생성 (Group Sampling)

**현재**: prompt 1개 → 응답 1개 생성.
**필요**: prompt 1개 → K개 응답 생성 (GRPO 기본 K=4~16).

GRPO의 핵심은 **같은 prompt에 대해 여러 응답을 비교**하여 group 내 상대적 우위로 advantage를 계산하는 것이다. 응답이 1개면 group이 형성되지 않으므로 advantage가 의미를 잃는다.

**구현 필요**:
- `_train_step_generation`에서 `input_ids`를 K번 repeat하여 생성
- 생성 결과를 (batch × K, seq) 형태로 확장
- reward scoring 후 (batch, K) 형태로 reshape하여 group normalization

**예상**: ~30줄.

## 2. Reward Model 인터페이스 표준화

**현재**: reward model의 `logits[:, last_token, 0]`을 scalar reward로 사용.
**필요**: reward model이 명시적으로 scalar reward를 반환하는 인터페이스.

실제 reward model은 backbone + **Linear(hidden, 1)** scalar head를 가진다. LM의 vocab logits 첫 값을 reward로 쓰는 건 우연히 동작할 뿐이다.

**해결 방향**: reward model이 `forward(input_ids) → {"reward": (batch,)}` dict를 반환하는 규약. `_compute_rewards`가 `reward_out.get("reward")` 또는 fallback으로 logits 사용.

**예상**: ~20줄.

## 3. PPO Value Bootstrapping

**현재**: `returns = advantages + values`로 value target 계산.
**필요**: `return_t = reward_t + gamma * V(t+1)` bootstrapped return.

현재 방식은 GAE advantage에 현재 value를 더해 return을 근사하는데, 이건 **수학적으로 올바르지만 수치적으로 불안정**할 수 있다. V(t)가 부정확할 때 return 추정이 크게 흔들린다.

정석: reward를 시간 역순으로 누적하여 discounted return을 직접 계산하고, value loss를 `MSE(V, return)`으로 잡는다.

**예상**: ~15줄 (compute_gae와 유사한 역순 루프).

## 4. 분산 RL Generation 검증

**현재**: 단일 GPU 테스트만 통과. 분산 환경에서 RL generation 미검증.
**잠재 문제**:
- FSDP 래핑된 policy에서 `model.generate()` 호출 시 all-gather 타이밍
- Multi-model backward에서 DDP gradient 동기화 순서
- Generation 중 DistributedSampler가 각 rank에 다른 prompt를 분배하는지

**검증 방법**: gloo CPU에서 2 프로세스 DDP + RLTrainer generation 테스트.

**예상**: ~60줄 테스트.

## 5. RL Resume

**현재**: 미구현. 학습 중단 시 처음부터 재시작.
**필요**: policy + value model의 checkpoint에서 이어서 학습.

SFT의 `_maybe_resume`와 동일한 패턴이지만, 복수 모델의 state_dict + 복수 optimizer 상태를 저장/복원해야 한다.

**예상**: ~50줄.

---

## 요약

| 갭 | 영향 | 난이도 | 의존 |
|---|---|:---:|---|
| K개 응답 생성 | GRPO 핵심 기능 누락 | 소 | 없음 |
| Reward model 인터페이스 | 실전 reward model 호환 불가 | 소 | 없음 |
| Value bootstrapping | PPO 수렴 불안정 가능 | 소 | 없음 |
| 분산 RL 검증 | 멀티 GPU에서 동작 미보장 | 중 | gloo CPU 테스트 |
| RL Resume | 긴 학습 중단 시 재시작 필요 | 중 | checkpoint 확장 |

총 ~175줄 + 60줄 테스트. DPO는 이 갭들과 무관하게 프로덕션 사용 가능.
