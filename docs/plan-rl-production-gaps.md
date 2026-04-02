# RL 프로덕션 갭 수정 계획

DPO는 프로덕션 준비 수준이다. GRPO와 PPO는 핵심 흐름이 동작하지만, 실전 대규모 학습에 필요한 5가지가 미구현이다. 이 문서는 각 갭의 현재 코드 상태를 정확히 진단하고, 수정 방향과 구체적인 변경 지점을 기술한다.

5개 갭은 독립적이지만, **갭 4(분산 RL)와 갭 5(RL Resume)는 갭 1~3의 수정이 완료된 상태를 전제**로 검증해야 한다. 따라서 구현 순서는 1 → 2 → 3 → 4 → 5를 권장한다.

---

## 1. K개 응답 생성 (Group Sampling)

### 왜 이것이 필요한가

GRPO의 핵심 아이디어는 **같은 prompt에 대해 여러 응답을 생성하고, group 내 상대적 우위로 advantage를 계산**하는 것이다. 응답이 1개면 group이 형성되지 않으므로, advantage는 단순히 "이 응답의 reward가 높은가 낮은가"가 되어 baseline 없는 REINFORCE와 같아진다. K=4~16개의 응답을 비교해야 reward의 noise를 줄이고 학습이 안정화된다.

### 현재 코드 상태

`rl_trainer.py:430-435`에서 generation이 일어난다:

```python
with torch.no_grad():
    generated_ids = self.policy.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        **self._generation_kwargs,
    )
```

prompt 1개당 `.generate()`를 한 번 호출하여 응답 1개만 생성한다. 이후 `_compute_rewards()`(480행)에서 reward를 추출하고, `GRPOLoss.__call__()`(rl.py:197-228)에서 이 reward를 `normalize_advantages()`에 넣는다. 그런데 `normalize_advantages()`(rl.py:42-65)는 **배치 전체의 reward 통계**로 정규화하지, group 내 통계로 정규화하지 않는다. 이것은 GRPO의 원래 의도와 다르다.

### 수정 방향

두 곳을 변경한다: (1) `_train_step_generation`에서 K개 응답을 생성하고, (2) `GRPOLoss`에서 group 단위 정규화를 수행한다.

### 변경 1-A: `rl_trainer.py` — `_train_step_generation`

`_generation_kwargs`에 `group_size`(K)를 추가한다. Recipe의 `generation.group_size` 필드에서 읽는다.

```python
# rl_trainer.py:430-435 → 수정
K = self._generation_kwargs.pop("group_size", 1)  # GRPO 기본: 4~16

with torch.no_grad():
    # prompt를 K번 repeat: (batch, seq) → (batch*K, seq)
    expanded_ids = prompt_ids.repeat_interleave(K, dim=0)
    expanded_mask = prompt_mask.repeat_interleave(K, dim=0) if prompt_mask is not None else None

    generated_ids = self.policy.generate(
        input_ids=expanded_ids,
        attention_mask=expanded_mask,
        **self._generation_kwargs,
    )
    gen_mask = (generated_ids != self.policy.config.pad_token_id).long()
```

generation 이후의 모든 텐서가 (batch\*K, ...) 형태가 된다. frozen model forward, reward 계산도 이 확장된 배치에서 수행한다. `gen_batch`에 `group_size=K`를 포함시켜 loss 클래스에 전달한다.

```python
gen_batch = {
    ...
    "rewards": rewards,        # (batch*K,) per-sequence scalar
    "group_size": K,           # loss가 group normalization에 사용
}
```

### 변경 1-B: `rl.py` — `GRPOLoss.__call__`

현재(rl.py:212-218) `normalize_advantages()`는 배치 전체 통계로 정규화한다. group_size가 1보다 크면 **group 내 정규화**로 전환한다.

```python
# rl.py:212-218 → 수정
raw_rewards = batch["rewards"]  # (batch*K,)
K = batch.get("group_size", 1)

if K > 1:
    # group 내 정규화: (batch*K,) → (batch, K) → normalize → (batch*K,)
    grouped = raw_rewards.view(-1, K)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
    advantages = ((grouped - mean) / std).view(-1)
else:
    advantages = normalize_advantages(raw_rewards.unsqueeze(-1), torch.ones_like(raw_rewards.unsqueeze(-1))).squeeze(-1)

# per-token broadcast (기존 로직 유지)
token_advantages = advantages.unsqueeze(-1).expand_as(new_log_probs) * mask
```

### 스키마 변경

`settings/schema.py`의 `GenerationSpec`에 `group_size: int = 1` 필드를 추가한다. GRPO Recipe에서 `generation.group_size: 8` 등으로 지정한다.

### 예상 규모

`rl_trainer.py` ~15줄, `rl.py` ~15줄, `schema.py` 1줄. 총 ~30줄.

---

## 2. Reward Model 인터페이스 표준화

### 왜 이것이 필요한가

현재 `_compute_rewards()`(rl_trainer.py:480-520)는 reward model의 출력에서 scalar reward를 추출하는데, **LM의 vocab logits 첫 번째 값**을 reward로 사용한다. 구체적으로 3D logits의 경우 `logits[batch_idx, last_token_idx, 0]`(499-504행)을 취한다. 이것이 "우연히 동작하는" 이유는 테스트에서 TinyLM(Embedding → Linear)의 출력이 곧 scalar처럼 사용되기 때문이다.

실제 reward model은 backbone + `Linear(hidden_dim, 1)` scalar head를 가진다. 이 모델의 출력은 `(batch, seq, 1)` 형태이고, 마지막 토큰의 값이 sequence-level reward다. 현재 코드는 이 경우에도 동작하지만(`logits[:, last_token, 0]`), reward model이 `(batch, 1)` 또는 `(batch,)` 형태의 명시적 reward를 반환하면 처리할 수 없다.

더 문제적인 것은 **fallback 로직**(511-518행)이다. reward model이 없으면 reference model의 log probability 합을 reward로 사용하는데, reference model은 KL penalty 계산용이지 reward 신호가 아니다. 이 fallback은 의미론적으로 잘못된 값을 조용히 반환한다.

### 현재 코드 상태

`rl_trainer.py:480-520` — `_compute_rewards()`:

```python
if "reward" in frozen_out:
    reward_logits = frozen_out["reward"].get("logits")
    if reward_logits is not None:
        if reward_logits.dim() == 3:       # (batch, seq, vocab)
            # ... logits[:, last_token, 0]
        elif reward_logits.dim() == 2:     # (batch, seq)
            scalar_rewards = reward_logits[:, -1]
        else:
            scalar_rewards = reward_logits.squeeze()

# fallback: reference log probs → reward (의미론적 오류)
if "reference" in frozen_out and "logits" in frozen_out["reference"]:
    ...
    return (ref_lp * mask).sum(dim=-1)

return torch.zeros(...)  # 최후 fallback: 0 reward
```

세 가지 문제가 있다:
1. reward model 출력 규약이 없어서 dim별 분기가 fragile하다
2. fallback이 잘못된 reward 신호를 조용히 반환한다
3. `_forward_model()`(557-591행)이 reward model 출력에서 `_extract_logits()`를 호출하여 `.logits` 속성이나 `dict["logits"]`를 찾는데, reward model이 `{"reward": tensor}` dict를 반환하면 logits 키가 없어 None이 된다

### 수정 방향

reward model의 출력 규약을 정한다: `forward(input_ids, ...) → dict`이며, `"reward"` 키가 있으면 그것이 `(batch,)` scalar reward다. `"reward"` 키가 없으면 기존 logits 기반 추출을 fallback으로 사용한다.

### 변경 2-A: `rl_trainer.py` — `_forward_model` 확장

reward model의 출력에서 `"reward"` 키를 우선 처리한다.

```python
# rl_trainer.py:557-591의 _forward_model 내부
# reward model 출력 처리 추가
if "input_ids" in batch:
    out = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
    logits = _extract_logits(out)
    result["logits"] = logits
    # reward model 전용: "reward" 키가 있으면 보존
    if isinstance(out, dict) and "reward" in out:
        result["reward"] = out["reward"]
    elif hasattr(out, "reward"):
        result["reward"] = out.reward
```

### 변경 2-B: `rl_trainer.py` — `_compute_rewards` 리팩토링

명시적 `"reward"` 키를 우선 사용하고, fallback 순서를 명확히 한다. reference model fallback은 제거한다.

```python
def _compute_rewards(self, frozen_out, generated_ids, gen_mask) -> torch.Tensor:
    if "reward" not in frozen_out:
        return torch.zeros(generated_ids.shape[0], device=generated_ids.device)

    reward_out = frozen_out["reward"]

    # 1순위: 명시적 scalar reward
    if "reward" in reward_out:
        r = reward_out["reward"]
        return r.view(-1) if r.dim() > 1 else r

    # 2순위: logits에서 마지막 유효 토큰의 첫 번째 값 추출
    logits = reward_out.get("logits")
    if logits is None:
        return torch.zeros(generated_ids.shape[0], device=generated_ids.device)

    if logits.dim() == 3:
        last_idx = gen_mask.sum(dim=-1).long() - 1
        return logits[torch.arange(logits.shape[0], device=logits.device), last_idx, 0]
    elif logits.dim() == 2:
        last_idx = gen_mask.sum(dim=-1).long() - 1
        return logits[torch.arange(logits.shape[0], device=logits.device), last_idx]
    else:
        return logits.squeeze()
```

reference model fallback을 제거한 이유: reference model의 log probability는 KL penalty 계산에 사용하는 것이 올바르고, reward 신호로 사용하면 학습 방향이 왜곡된다. reward model이 없는 상황에서는 0 reward를 반환하여 "reward 없이는 GRPO/PPO를 실행할 수 없다"는 사실을 명시적으로 드러내는 것이 안전하다.

### 예상 규모

`rl_trainer.py` ~25줄 변경. 신규 코드 없음.

---

## 3. PPO Value Bootstrapping

### 왜 이것이 필요한가

PPO는 value function V(s)로 advantage를 추정한다. 현재 `PPOLoss`(rl.py:234-318)에서 `compute_gae()`(rl.py:123-152)를 호출하여 GAE advantage를 계산하고, `returns = advantages + values`(313행)로 value target을 만든다. 이 수식 자체는 수학적으로 올바르다 — GAE advantage가 `A_t = sum_{l=0}^{T-t-1} (gamma*lam)^l * delta_t+l`이고, `return_t = A_t + V_t`이므로.

그러나 두 가지 실전 문제가 있다:

**첫째, value model 출력 처리에 잠재적 버그가 있다.** `PPOLoss.__call__`의 291행에서:

```python
if "value" in trainable_out and "logits" in trainable_out["value"]:
    values_raw = trainable_out["value"]["logits"][:, :-1, 0]
```

이 조건은 `trainable_out`에 `"value"` 키가 있고, 그 하위 dict에 `"logits"` 키가 있을 때만 value를 사용한다. 문제는 `_forward_model()`이 reward model이나 value model의 출력을 모두 `"logits"` 키로 저장하는데, value model의 출력 구조가 `(batch, seq, 1)`인지 `(batch, seq, vocab_size)`인지에 따라 `[:, :-1, 0]`의 의미가 완전히 달라진다는 것이다. value head가 `Linear(hidden, 1)`이면 올바르지만, LM head가 그대로 있으면 vocab의 첫 번째 토큰 logit을 value로 오인한다.

**둘째, `compute_gae`의 terminal value가 항상 0이다.** 147행에서:

```python
next_value = values[:, t + 1] if t + 1 < seq_len else torch.zeros_like(last_gae)
```

sequence의 마지막 토큰에서 `V(T+1) = 0`을 가정한다. 텍스트 생성에서 EOS 토큰이 나오면 이것이 맞지만, `max_new_tokens`에 의해 잘린 경우 V(T+1) != 0이다. 이 경우 return 추정이 과소평가된다.

### 수정 방향

value model 출력을 명시적으로 처리하고, truncation 시 bootstrap value를 보존한다.

### 변경 3-A: `rl_trainer.py` — value model 출력 처리

`_forward_model()`에서 value model 출력을 별도 키로 저장한다.

```python
# _forward_model 또는 _train_step_generation에서
# value model forward 후
if "value" in self.trainable:
    value_out = self._forward_model(self.trainable["value"],
                                     {"input_ids": generated_ids, "attention_mask": gen_mask})
    # value model이 "value" 키를 반환하면 우선 사용
    if "value" in value_out:
        trainable_out["value"] = {"values": value_out["value"]}
    else:
        # logits[:, :, 0]을 value로 해석 (scalar head 가정)
        trainable_out["value"] = {"values": value_out["logits"][:, :, 0]}
```

### 변경 3-B: `rl.py` — `PPOLoss` value 처리 안정화

value 추출을 `"logits"` 대신 `"values"` 키로 변경하고, shape 검증을 추가한다.

```python
# PPOLoss.__call__ 291-295행 → 수정
if "value" in trainable_out and "values" in trainable_out["value"]:
    values_raw = trainable_out["value"]["values"]  # (batch, seq) 또는 (batch, seq, 1)
    if values_raw.dim() == 3:
        values_raw = values_raw.squeeze(-1)  # (batch, seq)
    # causal shift: logits[:, :-1]에 대응
    values_shifted = values_raw[:, :-1]
    # new_log_probs shape에 맞추기
    if values_shifted.shape[1] < new_log_probs.shape[1]:
        values_padded = F.pad(values_shifted, (0, new_log_probs.shape[1] - values_shifted.shape[1]))
    else:
        values_padded = values_shifted[:, :new_log_probs.shape[1]]
else:
    values_padded = torch.zeros_like(new_log_probs)
```

### 변경 3-C: `rl.py` — `compute_gae` truncation bootstrap

`max_new_tokens`에 의해 잘린 시퀀스에서 terminal value를 bootstrap한다. 기존 구현은 유지하되, `done` mask를 선택적으로 받아 EOS가 아닌 truncation을 구분한다.

```python
def compute_gae(
    values: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
    last_values: torch.Tensor | None = None,  # 추가: truncation bootstrap
) -> torch.Tensor:
    seq_len = values.shape[1]
    advantages = torch.zeros_like(values)
    # truncation 시 마지막 value를 bootstrap으로 사용
    last_gae = torch.zeros(values.shape[0], device=values.device)

    for t in reversed(range(seq_len)):
        if t + 1 < seq_len:
            next_value = values[:, t + 1]
        elif last_values is not None:
            next_value = last_values  # bootstrap from value model
        else:
            next_value = torch.zeros_like(last_gae)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae

    return advantages * mask.float()
```

`last_values`는 `PPOLoss`에서 전달한다. EOS로 끝난 시퀀스는 `last_values=0`, truncation은 `last_values=V(last_token)`을 넣는다. 기존 호출부(`last_values=None`)는 동작이 변하지 않으므로 하위 호환된다.

### 예상 규모

`rl_trainer.py` ~10줄, `rl.py` ~20줄. 총 ~30줄.

---

## 4. 분산 RL Generation 검증

### 왜 이것이 필요한가

현재 RL 학습은 단일 GPU에서만 테스트되었다. 분산 환경에서는 세 가지 잠재 문제가 있다:

1. **FSDP 래핑된 policy에서 `.generate()` 호출**: FSDP는 forward pass 전에 all-gather로 파라미터를 모은다. `.generate()`는 내부적으로 여러 번의 forward를 실행하는데, 각 forward마다 all-gather가 발생하여 모든 rank가 동기화된 상태여야 한다. 한 rank만 `.generate()`를 호출하면 다른 rank가 대기하며 deadlock이 발생한다.

2. **DistributedSampler가 각 rank에 다른 prompt를 분배하는지**: SFT Trainer는 `set_epoch()`로 이를 보장하지만(plan-distributed-training.md 이슈 1), RLTrainer의 `train()` 메서드(304-406행)에서 sampler에 대한 `set_epoch()` 호출이 **존재하는지 확인이 필요**하다.

3. **Multi-model backward에서 DDP gradient 동기화**: DDP는 backward 시 자동으로 gradient를 동기화한다. 하지만 RL에서는 policy만 backward를 수행하고 reference/reward는 frozen이다. frozen model이 DDP로 래핑되지 않았는지 확인해야 한다.

### 현재 코드 상태

`rl_trainer.py:307-328`에서 strategy 적용:

```python
if self.strategy is not None:
    if hasattr(self.strategy, "setup_models"):
        wrapped = self.strategy.setup_models(all_models, self.device, trainable_names)
        ...
    else:
        for name, model in self.trainable.items():
            self.trainable[name] = self.strategy.setup(model, self.device)
        ...
        for name, model in self.frozen.items():
            self.frozen[name] = model.to(self.device)  # DDP 래핑 없이 device 이동만
```

frozen model은 DDP 래핑 없이 device로 이동만 한다 — 이것은 올바르다. frozen model에 backward가 흐르지 않으므로 gradient 동기화가 불필요하다.

`setup_models`는 FSDPStrategy(`fsdp.py:127`)와 DeepSpeedStrategy(`deepspeed.py:78`)가 구현하고 있으므로, 이 두 전략에서는 trainable/frozen 구분이 `setup_models` 내부에서 처리된다. **DDPStrategy만 `setup_models`가 없어** else 분기로 빠지며, 이때 frozen model은 DDP 래핑 없이 device 이동만 한다 — 이것은 올바르다.

RLTrainer의 `train()` 메서드(304-406행)는 SFT Trainer와 달리 **에폭 루프가 아니라 step 기반 while 루프**를 사용한다:

```python
while self.global_step < max_steps:
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(self.train_loader)
        batch = next(data_iter)
    ...
```

`StopIteration`이 발생하면 iterator를 재생성하여 다음 에폭으로 넘어간다. 이 시점에 `set_epoch()` 호출이 없으므로, 분산 환경에서 매 에폭 동일한 셔플 순서가 반복된다.

### 수정 방향

코드 수정보다 **검증 테스트**가 우선이다. 실제 multi-GPU 환경이 없어도 gloo CPU backend로 2 프로세스를 띄워 기본 동작을 검증할 수 있다.

### 변경 4-A: `rl_trainer.py` — `set_epoch` 추가

RLTrainer는 step 기반 while 루프를 사용하므로, `StopIteration` 발생 시(에폭 경계) epoch counter를 증가시키고 `set_epoch()`을 호출한다.

```python
# rl_trainer.py:354 while 루프 내부, StopIteration 핸들러
epoch_counter = 0
while self.global_step < max_steps:
    try:
        batch = next(data_iter)
    except StopIteration:
        epoch_counter += 1
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch_counter)
        data_iter = iter(self.train_loader)
        batch = next(data_iter)
    ...
```

### 변경 4-B: `tests/e2e/test_distributed_rl.py` — 분산 RL 테스트

gloo CPU backend로 2 프로세스 DDP + RLTrainer를 테스트한다.

```python
# 테스트 시나리오
# 1. DPO 분산: 2 rank에서 preference 데이터 학습, loss가 finite
# 2. GRPO 분산: 2 rank에서 generation, 각 rank가 다른 prompt를 받는지 확인
# 3. set_epoch: 에폭 간 데이터 순서가 달라지는지 확인
```

각 테스트는 `torch.multiprocessing.spawn()`으로 2 프로세스를 생성하고, `init_process_group(backend="gloo")`로 분산 환경을 구성한다. `DDPStrategy`를 사용한다.

### 예상 규모

`rl_trainer.py` ~8줄 수정, `test_distributed_rl.py` ~80줄 신규 테스트.

> **참고**: frozen model forward(445-449행)는 이미 `torch.no_grad()` 컨텍스트 안에 있으므로 추가 수정이 불필요하다.

---

## 5. RL Resume

### 왜 이것이 필요한가

GRPO/PPO 학습은 대규모 모델에서 수 시간~수 일이 걸린다. 학습 중단 시 처음부터 재시작하면 GPU 시간이 낭비된다. SFT Trainer에는 `_maybe_resume()`(trainer.py:541-617)이 구현되어 있지만, RLTrainer에는 resume 로직이 전혀 없다.

SFT resume이 복원하는 상태:
- 모델 가중치 (safetensors 또는 adapter)
- optimizer state dict
- scheduler state dict
- GradScaler state dict (fp16 시)
- trainer state (epoch, global_step)

RL resume은 이 모든 것을 **모델별로** 복원해야 한다. policy는 가중치 + optimizer + scheduler, reference/reward는 가중치만, value는 가중치 + optimizer가 필요하다.

**주의**: RLTrainer의 `self.schedulers` dict는 scheduler 객체를 직접 저장한다(rl_trainer.py:104). SFT Trainer와 달리 `(scheduler, interval)` 튜플이 아니다. 또한 RLTrainer는 `self._current_epoch` 속성이 없고 `self.global_step`만 추적한다. resume 코드는 이 차이를 반영해야 한다.

### 현재 코드 상태

RLTrainer에 resume 관련 코드가 전혀 없다:
- `__init__`에서 `global_step = 0`(121행)으로 초기화
- checkpoint 저장은 callback이 담당하지만, RLTrainer용 checkpoint 구조가 정의되지 않음
- `_export_policy_artifact()`(250-300행)는 MLflow artifact 저장 전용이고 resume과 무관

SFT의 `_maybe_resume()`가 참조 구현이다. RLTrainer는 이 패턴을 확장하여 **모델별 하위 디렉토리**를 가진 체크포인트 구조를 사용한다.

### 수정 방향

### 변경 5-A: 체크포인트 디렉토리 구조

```
checkpoint-{global_step}/
  policy/
    model.safetensors (또는 adapter_model.safetensors)
    optimizer.pt
    scheduler.pt
  value/                    # PPO만
    model.safetensors
    optimizer.pt
  reference/
    model.safetensors       # frozen이라 optimizer 없음
  reward/
    model.safetensors       # frozen
  trainer_state.json        # {"epoch": N, "global_step": M}
  scaler.pt                 # fp16 시
```

### 변경 5-B: `rl_trainer.py` — `_save_checkpoint` 메서드

ModelCheckpoint 콜백이 호출할 저장 메서드를 RLTrainer에 추가한다.

```python
def _save_checkpoint(self, ckpt_dir: Path) -> None:
    """모든 모델의 상태를 저장한다."""
    for name, model in {**self.trainable, **self.frozen}.items():
        model_dir = ckpt_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        # 모델 가중치
        unwrapped = getattr(model, "module", model)
        if hasattr(unwrapped, "save_pretrained"):
            unwrapped.save_pretrained(model_dir)
        else:
            torch.save(unwrapped.state_dict(), model_dir / "model.pt")

        # optimizer (trainable만)
        if name in self.optimizers:
            torch.save(self.optimizers[name].state_dict(), model_dir / "optimizer.pt")
        # scheduler — RLTrainer는 scheduler를 직접 저장 (튜플이 아님)
        if name in self.schedulers:
            torch.save(self.schedulers[name].state_dict(), model_dir / "scheduler.pt")

    # 공유 상태 — RLTrainer는 epoch를 추적하지 않으므로 global_step만 저장
    import json
    (ckpt_dir / "trainer_state.json").write_text(json.dumps({
        "global_step": self.global_step,
    }))
    if hasattr(self, "scaler") and self.scaler.is_enabled():
        torch.save(self.scaler.state_dict(), ckpt_dir / "scaler.pt")
```

### 변경 5-C: `rl_trainer.py` — `_maybe_resume` 메서드

SFT의 `_maybe_resume` 패턴을 RL 멀티모델 구조에 맞게 확장한다.

```python
def _maybe_resume(self) -> None:
    """체크포인트에서 모든 모델 상태를 복원한다."""
    resume_cfg = self.settings.config.job.resume
    if resume_cfg == "disabled":
        return

    ckpt_dir = self._resolve_checkpoint_dir(resume_cfg)
    if ckpt_dir is None:
        return

    for name, model in {**self.trainable, **self.frozen}.items():
        model_dir = ckpt_dir / name
        if not model_dir.exists():
            logger.warning(f"Resume: {name} 체크포인트 없음, 건너뜀")
            continue

        unwrapped = getattr(model, "module", model)
        # 가중치 로드 (adapter 우선)
        adapter_path = model_dir / "adapter_model.safetensors"
        if adapter_path.exists() and hasattr(unwrapped, "load_adapter"):
            unwrapped.load_adapter(model_dir, adapter_name="default")
        elif (model_dir / "model.safetensors").exists():
            from safetensors.torch import load_file
            unwrapped.load_state_dict(load_file(model_dir / "model.safetensors"), strict=False)
        elif (model_dir / "model.pt").exists():
            unwrapped.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))

        # optimizer/scheduler (trainable만)
        if name in self.optimizers and (model_dir / "optimizer.pt").exists():
            self.optimizers[name].load_state_dict(
                torch.load(model_dir / "optimizer.pt", map_location="cpu"))
        # scheduler — RLTrainer는 scheduler를 직접 저장 (튜플이 아님)
        if name in self.schedulers and (model_dir / "scheduler.pt").exists():
            self.schedulers[name].load_state_dict(
                torch.load(model_dir / "scheduler.pt", map_location="cpu"))

    # 공유 상태 — global_step만 복원 (RLTrainer는 epoch를 추적하지 않음)
    state_path = ckpt_dir / "trainer_state.json"
    if state_path.exists():
        import json
        state = json.loads(state_path.read_text())
        self.global_step = state.get("global_step", 0)

    scaler_path = ckpt_dir / "scaler.pt"
    if scaler_path.exists() and hasattr(self, "scaler") and self.scaler.is_enabled():
        self.scaler.load_state_dict(torch.load(scaler_path, map_location="cpu"))

    logger.info(f"Resumed from {ckpt_dir} (epoch={self._current_epoch}, step={self.global_step})")
```

### 변경 5-D: `train()` 메서드에 resume 호출 삽입

`rl_trainer.py:304-406`의 `train()` 메서드, strategy setup 이후 에폭 루프 시작 전에:

```python
# strategy setup 이후, 에폭 루프 이전
self._maybe_resume()
```

### 변경 5-E: ModelCheckpoint 콜백과의 연동

현재 `checkpoint.py`의 `save_checkpoint()`는 SFT Trainer용으로, 단일 model/optimizer/scheduler를 받는다. RLTrainer는 자체 `_save_checkpoint`를 가지므로, 콜백이 Trainer의 저장 메서드를 호출하도록 변경한다.

현재 `_fire()` 메서드(rl_trainer.py:162-169, trainer.py:243-260)는 callback에 trainer 인스턴스를 전달하지 않는다. 두 가지 접근이 가능하다:

**A. `_fire`에 `trainer=self` 추가** (권장):
RLTrainer와 SFT Trainer 모두의 `_fire` 메서드에서 `trainer=self`를 kwargs에 포함시킨다. 기존 콜백은 `**kwargs`를 받으므로 하위 호환된다.

```python
# rl_trainer.py:162-169, trainer.py:243-260
def _fire(self, event: str, **kwargs):
    for cb in self.callbacks:
        fn = getattr(cb, event, None)
        if fn:
            fn(**kwargs, trainer=self)
```

ModelCheckpoint의 save 로직에서:
```python
# checkpoint.py — save 분기
trainer = kwargs.get("trainer")
if trainer is not None and hasattr(trainer, "_save_checkpoint"):
    trainer._save_checkpoint(ckpt_dir)
else:
    # 기존 SFT 저장 로직
    ...
```

**B. ModelCheckpoint에 save callable 주입** (대안):
RLTrainer가 ModelCheckpoint 생성 시 `save_fn=self._save_checkpoint`를 주입한다. callback이 trainer를 알 필요 없이 저장 함수만 호출한다. 다만 이 경우 `_create_callbacks`에서 ModelCheckpoint를 특별 처리해야 하므로 A가 더 단순하다.

### 예상 규모

`rl_trainer.py` ~80줄 (save + resume + resolve), `checkpoint.py` ~5줄, `_fire` 수정 ~3줄. 총 ~90줄.

---

## 교차 관심사

5개 갭 수정 시 공통으로 영향받는 영역을 정리한다.

### `_train_step_generation` 메서드의 흐름 변경

갭 1(K개 응답), 갭 2(reward 인터페이스), 갭 3(value 처리)이 모두 이 메서드를 수정한다. 수정 후 흐름:

```
prompt (batch, seq)
  → repeat_interleave K   # 갭 1
  → policy.generate()
  → gen_mask 생성
  → frozen forward (reward, reference)
  → _compute_rewards()    # 갭 2: 표준화된 인터페이스
  → value forward          # 갭 3: 별도 value 처리
  → gen_batch 조립 (group_size 포함)
  → algorithm(trainable_out, frozen_out, gen_batch)
```

### 스키마 변경 요약

| 변경 | 파일 | 필드 |
|------|------|------|
| 갭 1 | `schema.py` GenerationSpec | `group_size: int = 1` |
| 갭 5 | 없음 (기존 `config.job.resume` 재사용) | — |

### 테스트 추가 요약

| 갭 | 테스트 파일 | 시나리오 |
|---|---|---|
| 1 | `test_rl_generation.py` | GRPO K=4 group sampling, advantage가 group 내 정규화됨 |
| 2 | `test_rl_generation.py` | reward model이 `{"reward": (batch,)}` 반환 시 정상 추출 |
| 3 | `test_rl_generation.py` | PPO value model 출력이 `"values"` 키로 전달됨 |
| 4 | `test_distributed_rl.py` (신규) | gloo CPU 2-process DPO/GRPO 분산 학습 |
| 5 | `test_rl_integration.py` | checkpoint 저장 → resume → global_step 복원 |

### AGENT.md 업데이트

갭 1의 `generation.group_size` 필드를 AGENT.md의 Recipe Schema에 추가해야 한다. AGENT.md는 에이전트의 단일 참조 문서이므로, 스키마 변경 시 반드시 동기화한다.

---

## 구현 순서

```
독립 (병렬 가능):
  1. K개 응답 생성 — rl_trainer.py + rl.py + schema.py
  2. Reward model 인터페이스 — rl_trainer.py
  3. Value bootstrapping — rl_trainer.py + rl.py

순차 (1~3 완료 후):
  4. 분산 RL 검증 — rl_trainer.py + test_distributed_rl.py
  5. RL Resume — rl_trainer.py + checkpoint.py + test

총 예상: ~180줄 코드 + ~80줄 테스트
```
