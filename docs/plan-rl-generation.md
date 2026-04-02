# Phase 3-4: GRPO + PPO — Generation 루프 구현 계획

## 현재 상태

Phase 1-2 완료. `mdp rl-train`이 동작하며:
- DPO: 내장 `DPOLoss` (`_component_: DPO`)
- weighted-NTP: 커스텀 알고리즘 주입 (`_component_: my_project.WeightedNTPLoss`)
- 데이터 → forward → loss → backward 경로 검증 완료
- PreferenceCollator (chosen/rejected) + causal (input_ids) 양쪽 데이터 형태 지원

**미구현**: policy가 학습 중 텍스트를 **생성**하고, 그 생성 결과에 대해 보상을 받아 업데이트하는 "online RL" 경로.

---

## Phase 1-2 vs Phase 3-4의 차이

```
Phase 1-2 (DPO, weighted-NTP):
  DataLoader → batch → forward → loss → backward
  (데이터가 이미 완성됨. policy가 생성하지 않음)

Phase 3-4 (GRPO, PPO):
  DataLoader → prompts → policy.generate() → responses
  → reward scoring → advantage → policy update
  (policy가 매 step 직접 텍스트를 생성)
```

### 이것이 변경하는 것

| 항목 | Phase 1-2 | Phase 3-4 |
|------|-----------|-----------|
| train() 루프 | batch → forward → loss | **prompt → generate → score → update** |
| forward 횟수 | 1회 per step | 2회 (generate용 no_grad + update용 grad) |
| 배치 크기 | 고정 | prompt × K responses로 **확대** |
| mini-epoch | 없음 | 같은 생성 결과로 **여러 번 update** |
| 데이터 | chosen/rejected 또는 input_ids | **prompt만** (응답은 policy가 생성) |
| 저장해야 할 것 | 없음 | 생성 시점의 old_log_probs |

---

## 수정 파일 및 변경 내용

### 1. `training/rl_trainer.py` — train() 루프 분기 + generation 메서드

**변경**: train() 안에 `needs_generation` 분기 1개 추가. 기존 DPO/weighted-NTP 경로는 그대로.

```python
def train(self):
    ...
    while self.global_step < max_steps:
        batch = next(train_iter)
        batch = self._move_to_device(batch)

        if hasattr(self.algorithm, 'needs_generation') and self.algorithm.needs_generation:
            self._train_step_with_generation(batch, device_type)
        else:
            self._train_step_offline(batch, device_type)  # 기존 로직 추출
```

**신규 메서드**:

`_train_step_offline(batch)` — 기존 DPO/weighted-NTP 로직을 메서드로 추출. 코드 이동만, 변경 없음.

`_train_step_with_generation(batch)`:
```python
def _train_step_with_generation(self, batch, device_type):
    prompt_ids = batch["input_ids"]  # prompt만
    
    # 1. Generation (no_grad)
    with torch.no_grad():
        generated_ids = self.policy.generate(
            input_ids=prompt_ids,
            attention_mask=batch.get("attention_mask"),
            **self._generation_kwargs,
        )
    
    # 2. Old log_probs 저장 (update 전 policy 상태)
    with torch.no_grad():
        old_logits = self.policy(generated_ids).logits
        old_log_probs = compute_log_probs(old_logits, generated_ids)
    
    # 3. Frozen forward (reference log_probs, reward 등)
    with torch.no_grad():
        frozen_out = {}
        for name, model in self.frozen.items():
            frozen_out[name] = self._forward_model(model, {"input_ids": generated_ids})
    
    # 4. Mini-epoch update
    gen_batch = {
        "input_ids": generated_ids,
        "prompt_length": prompt_ids.shape[1],
        "old_log_probs": old_log_probs,
    }
    
    ppo_epochs = getattr(self.algorithm, 'ppo_epochs', 1)
    for _ in range(ppo_epochs):
        with autocast(device_type, ...):
            trainable_out = {}
            for name, model in self.trainable.items():
                trainable_out[name] = self._forward_model(model, {"input_ids": generated_ids})
            
            losses = self.algorithm(trainable_out, frozen_out, gen_batch)
        
        self._backward_and_step(losses)
```

`_backward_and_step(losses)` — 기존 backward + optimizer step 로직 추출.

**예상 줄 수 변화**: rl_trainer.py ~326줄 → ~480줄 (+154줄)

### 2. `training/losses/rl.py` — GRPOLoss, PPOLoss 클래스 추가

기존 `DPOLoss` 옆에 동일 패턴으로 추가.

**GRPOLoss** (~40줄):

```python
class GRPOLoss:
    """Group Relative Policy Optimization.
    
    Generation 필요. K개 응답의 group reward 평균 대비 advantage로 policy gradient.
    Value model 불필요 (PPO 대비 메모리 절약).
    """
    needs_generation = True
    
    def __init__(self, clip_range=0.2, kl_coeff=0.1, num_generations=4, ppo_epochs=1):
        self.clip_range = clip_range
        self.kl_coeff = kl_coeff
        self.num_generations = num_generations
        self.ppo_epochs = ppo_epochs
    
    def __call__(self, trainable_out, frozen_out, batch):
        # new log_probs
        new_log_probs = compute_log_probs(trainable_out["policy"]["logits"], batch["input_ids"])
        old_log_probs = batch["old_log_probs"]
        
        # importance ratio
        ratio = (new_log_probs - old_log_probs).exp()
        
        # group advantage (batch에서 계산)
        rewards = batch.get("rewards", torch.zeros_like(new_log_probs[:, 0]))
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # clipped surrogate
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = ratio.clamp(1 - self.clip_range, 1 + self.clip_range) * advantages.unsqueeze(-1)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL penalty
        ref_log_probs = compute_log_probs(frozen_out["reference"]["logits"], batch["input_ids"])
        kl = (new_log_probs - ref_log_probs).mean()
        
        return {"policy": policy_loss + self.kl_coeff * kl}
```

**PPOLoss** (~60줄):

GRPOLoss와 유사하되 value model loss 추가:

```python
class PPOLoss:
    needs_generation = True
    
    def __init__(self, clip_range=0.2, kl_coeff=0.1, value_coeff=0.5, ppo_epochs=4):
        ...
    
    def __call__(self, trainable_out, frozen_out, batch):
        # policy loss (GRPO와 동일한 clipped surrogate)
        ...
        
        # value loss (value model이 trainable일 때)
        if "value" in trainable_out:
            values = trainable_out["value"]["logits"][:, :, 0]  # scalar value
            value_targets = batch.get("value_targets")
            value_loss = F.mse_loss(values, value_targets)
            return {"policy": policy_loss, "value": self.value_coeff * value_loss}
        
        return {"policy": policy_loss}
```

### 3. `mdp/aliases.yaml` — GRPO, PPO alias 추가

```yaml
algorithm:
  DPO: mdp.training.losses.rl.DPOLoss
  GRPO: mdp.training.losses.rl.GRPOLoss
  PPO: mdp.training.losses.rl.PPOLoss
```

### 4. `settings/schema.py` — 변경 없음

`algorithm`은 이미 `dict[str, Any]` (_component_ 패턴). GRPO/PPO의 config (clip_range, kl_coeff 등)는 algorithm dict 안에 들어간다:

```yaml
algorithm:
  _component_: GRPO
  clip_range: 0.2
  kl_coeff: 0.1
  num_generations: 4
```

Recipe의 `generation` 섹션도 이미 `GenerationSpec`으로 존재. RLTrainer가 이를 읽어 `policy.generate()` kwargs로 전달.

### 5. 기타 파일 — 변경 없음

| 파일 | 이유 |
|------|------|
| `factory/factory.py` | `create_models()`가 이미 복수 모델 지원 |
| `cli/rl_train.py` | algorithm을 모르므로 변경 불필요 |
| `__main__.py` | 이미 rl-train 등록됨 |
| `strategies/*.py` | `setup_models()`가 이미 trainable/frozen 분리 |
| `callbacks/*.py` | policy만 전달하면 기존과 동일 |
| `mdp export` / `serve` | policy 모델 1개 → 기존 그대로 |

---

## 분산 학습 호환성

### DDP/FSDP

Phase 1-2와 동일. policy → FULL_SHARD, frozen → NO_SHARD. generation은 no_grad이므로 FSDP all-gather만 발생하고 backward은 update 시에만.

### DeepSpeed (PPO 특수)

DeepSpeed는 `deepspeed.initialize(model, optimizer)`로 모델과 optimizer를 결합한다. PPO에서 policy + value 2개 모델이 trainable이면:

```python
policy_engine, policy_opt, _, _ = deepspeed.initialize(model=policy, config=ds_config_policy)
value_engine, value_opt, _, _ = deepspeed.initialize(model=value, config=ds_config_value)
```

**별도 DeepSpeed engine 2개**를 생성해야 한다. 현재 `DeepSpeedStrategy.setup()`은 1개 모델만 받으므로, `setup_models()`에서 trainable 모델마다 별도 engine을 생성하도록 확장한다.

frozen 모델 (reference)은 DeepSpeed engine으로 감싸지 않는다 — forward만 하므로 `model.to(device)`로 충분.

```python
class DeepSpeedStrategy(BaseStrategy):
    def setup_models(self, models, device, trainable_names=None):
        import deepspeed
        
        wrapped = {}
        for name, model in models.items():
            if name in trainable_names:
                engine, *_ = deepspeed.initialize(model=model, config=self.ds_config)
                wrapped[name] = engine
            else:
                wrapped[name] = model.to(device)
        return wrapped
```

### generation 중 분산 동기화

`model.generate()`는 각 rank에서 독립 실행된다. FSDP 래핑된 policy에서 generate()를 호출하면 FSDP가 내부적으로 all-gather를 수행하여 전체 파라미터를 복원한 뒤 생성한다. **별도 처리 불필요** — FSDP가 자동으로 관리.

DDP에서도 generate()는 각 rank가 독립 실행하고, 서로 다른 prompt를 생성한다 (DistributedSampler가 데이터를 분배). 동기화는 backward에서만 필요.

---

## LoRA + Generation 호환

LoRA adapter가 적용된 policy에서 `model.generate()`를 호출하면 PEFT가 내부적으로 adapter를 적용한 상태에서 생성한다. **별도 처리 불필요.**

LoRA + FSDP + generate 조합도 동작한다 — PEFT의 FSDP 지원이 이를 커버.

---

## Checkpoint + Export

generation이 추가되어도 최종 결과물은 **policy 모델 1개**다. 기존 `ModelCheckpoint`가 policy만 저장하고, `mdp export`가 policy만 merge한다.

GRPO/PPO에서 value model이 함께 학습되는 경우, value model의 체크포인트는 **별도 저장이 필요할 수 있다** (PPO 학습 resume 시). 이건 `ModelCheckpoint`에 `models_to_save` 파라미터를 추가하거나, RLTrainer가 별도 저장 로직을 가지는 방식으로 해결. **Phase 4 구현 시 결정.**

---

## Recipe YAML 예시

### GRPO

```yaml
algorithm:
  _component_: GRPO
  clip_range: 0.2
  kl_coeff: 0.1
  num_generations: 4
  ppo_epochs: 1

models:
  policy:
    class_path: transformers.AutoModelForCausalLM
    pretrained: hf://meta-llama/Meta-Llama-3-8B
    adapter: {method: lora, r: 64}
    optimizer:
      _component_: AdamW
      lr: 1e-6
  reference:
    pretrained: hf://meta-llama/Meta-Llama-3-8B

generation:
  max_new_tokens: 256
  temperature: 0.7
  do_sample: true

data:
  source: ./prompts.jsonl
  fields:
    text: prompt             # prompt만 — 응답은 policy가 생성
  tokenizer:
    pretrained: meta-llama/Meta-Llama-3-8B
    max_length: 512          # prompt 최대 길이

training:
  max_steps: 500
  precision: bf16
  gradient_accumulation_steps: 4
```

### PPO

```yaml
algorithm:
  _component_: PPO
  clip_range: 0.2
  kl_coeff: 0.1
  value_coeff: 0.5
  ppo_epochs: 4

models:
  policy:
    class_path: transformers.AutoModelForCausalLM
    pretrained: hf://meta-llama/Meta-Llama-3-8B
    adapter: {method: lora, r: 64}
    optimizer:
      _component_: AdamW
      lr: 5e-7
  value:
    class_path: my_models.ValueModel
    pretrained: hf://meta-llama/Meta-Llama-3-8B
    adapter: {method: lora, r: 32}
    freeze: false
    optimizer:
      _component_: AdamW
      lr: 1e-4
  reference:
    pretrained: hf://meta-llama/Meta-Llama-3-8B
  reward:
    class_path: my_models.RewardModel
    pretrained: ./reward-model-checkpoint/

generation:
  max_new_tokens: 512
  temperature: 0.8
  do_sample: true
```

---

## 구현 순서

### Phase 3: GRPO (generation 인프라 구축)

| Step | 작업 | 파일 |
|:---:|------|------|
| 3-1 | train()에서 offline/generation 분기 | `rl_trainer.py` |
| 3-2 | `_train_step_with_generation()` 구현 | `rl_trainer.py` |
| 3-3 | `GRPOLoss` 클래스 | `losses/rl.py` |
| 3-4 | aliases.yaml GRPO 추가 | `aliases.yaml` |
| 3-5 | 테스트 (TinyLM generate + GRPO loss) | `test_rl_grpo.py` |

### Phase 4: PPO (value model + multi-loss)

| Step | 작업 | 파일 |
|:---:|------|------|
| 4-1 | `PPOLoss` 클래스 (policy + value loss) | `losses/rl.py` |
| 4-2 | DeepSpeed `setup_models` 구현 | `strategies/deepspeed.py` |
| 4-3 | aliases.yaml PPO 추가 | `aliases.yaml` |
| 4-4 | 테스트 (TinyLM + TinyValue PPO) | `test_rl_ppo.py` |

---

## 예상 코드량

| 파일 | 변경 | 줄 |
|------|------|:---:|
| `training/rl_trainer.py` | +_train_step_with_generation, +_backward_and_step, train() 분기 | +154 |
| `training/losses/rl.py` | +GRPOLoss, +PPOLoss | +100 |
| `strategies/deepspeed.py` | +setup_models | +20 |
| `aliases.yaml` | +GRPO, +PPO | +2 |
| `test_rl_grpo.py` | 신규 | +60 |
| `test_rl_ppo.py` | 신규 | +80 |
| **합계** | | **~416** |
