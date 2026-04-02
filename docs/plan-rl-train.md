# `mdp rl-train` 구현 계획

## 개요

MDP에 `mdp rl-train` 명령어를 추가하여 LLM/VLM preference alignment 학습을 지원한다.

### 지원 알고리즘 (점진적)

| 단계 | 알고리즘 | 모델 수 | Generation | 작업량 |
|:---:|------|:---:|:---:|:---:|
| 1 | DPO | 2 (policy + ref) | X | 2일 |
| 2 | weighted-NTP | 2 (policy + critic) | X | 1일 |
| 3 | GRPO | 2 (policy + ref) + gen | O | 1.5일 |
| 4 | PPO | 3~4 + gen | O | 2일 |

### 설계 원칙

- 기존 `mdp train` (SFT)은 **한 줄도 바꾸지 않는다**
- 학습 결과는 policy 모델 1개 → 기존 `mdp export` / `mdp serve`로 서빙
- RL 전용 코드는 `mdp/training/rl/` 하위에 격리
- 기존 인프라 (Strategy, MLflow, Checkpoint, ComponentResolver) 최대 재사용

---

## 기존 코드 재사용 분석

탐색 결과, 기존 Trainer의 29개 메서드/블록 중:

| 분류 | 개수 | 대표 |
|------|:---:|------|
| **그대로 재사용** | 9 | _detect_device, AMP setup, gradient clipping, callbacks, MLflow, _move_to_device |
| **소폭 수정** | 13 | _create_optimizer (복수 모델), _create_scheduler (복수), Strategy (복수 모델 sharding), resume (복수 모델 state) |
| **새로 구현** | 7 | train() 루프, _train_one_epoch, _compute_loss, validation, generation |

**접근**: Trainer를 상속하지 않고, **공통 유틸을 뽑아서 RLTrainer가 직접 사용**한다. 상속은 메서드 오버라이드 체인이 복잡해지고 SFT 코드에 영향을 줄 위험이 있다.

---

## 아키텍처

### 디렉토리 구조

```
mdp/
  training/
    trainer.py            # 기존 SFT Trainer (변경 없음)
    rl/
      __init__.py
      rl_trainer.py       # RLTrainer 기반 클래스
      algorithms/
        __init__.py
        dpo.py            # DPO loss + training step
        weighted_ntp.py   # Critic + GAE + AWR + weighted CE
        grpo.py           # Group reward + policy gradient
        ppo.py            # Full PPO (generation + critic + clipping)
      rl_utils.py         # compute_log_probs, masked_mean, compute_kl
  settings/
    rl_schema.py          # RLRecipe Pydantic 모델
  cli/
    rl_train.py           # mdp rl-train CLI
```

### RLRecipe YAML 구조

```yaml
algorithm: dpo          # dpo | weighted_ntp | grpo | ppo

models:
  policy:
    class_path: transformers.AutoModelForCausalLM
    pretrained: hf://meta-llama/Meta-Llama-3-8B
    adapter: {method: lora, r: 64, alpha: 128}
  reference:
    pretrained: hf://meta-llama/Meta-Llama-3-8B  # frozen
    # adapter 없음 → 원본 가중치 그대로

# 알고리즘별 config
dpo:
  beta: 0.1             # DPO 온도

# weighted_ntp:
#   gae_lambda: 0.95
#   awr_beta: 1.0
#   weight_clip: [0.1, 3.0]

data:
  source: ./preference_pairs.jsonl
  fields:
    prompt: prompt
    chosen: chosen
    rejected: rejected
  tokenizer:
    pretrained: meta-llama/Meta-Llama-3-8B
    max_length: 2048

training:
  max_steps: 1000       # RL은 step 기반
  precision: bf16
  gradient_accumulation_steps: 4
  gradient_clip_max_norm: 1.0

optimizer:
  _component_: AdamW
  lr: 5e-7              # RL은 SFT보다 훨씬 낮은 LR

metadata:
  author: wesley
  description: "LLaMA-3 8B DPO alignment"
```

### RLTrainer 핵심 구조

```python
class RLTrainer:
    def __init__(self, settings: RLSettings, models: dict[str, nn.Module], train_loader, ...):
        self.policy = models["policy"]
        self.frozen_models = {k: v for k, v in models.items() if k != "policy"}
        # frozen 모델은 eval + no_grad
        for m in self.frozen_models.values():
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
        
        self.algorithm = create_algorithm(settings)  # DPO, PPO, etc.
        # optimizer는 policy 파라미터만
        self.optimizer = create_optimizer(self.policy.parameters(), ...)
    
    def train(self):
        for step in range(max_steps):
            batch = next(train_iter)
            
            # 1. (선택) Generation — PPO/GRPO만
            if self.algorithm.needs_generation:
                batch = self.algorithm.generate(self.policy, batch)
            
            # 2. Frozen 모델 forward (no_grad)
            with torch.no_grad():
                frozen_outputs = {
                    name: model(batch) for name, model in self.frozen_models.items()
                }
            
            # 3. Policy forward + loss
            policy_outputs = self.policy(batch)
            loss = self.algorithm.compute_loss(policy_outputs, frozen_outputs, batch)
            
            # 4. Backward + optimizer step (기존 Trainer와 동일한 AMP/grad_accum/clip 패턴)
            self.scaler.scale(loss).backward()
            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # 5. MLflow 로깅, callbacks
```

### Algorithm 인터페이스

```python
class BaseAlgorithm:
    needs_generation: bool = False
    
    def compute_loss(self, policy_outputs, frozen_outputs, batch) -> torch.Tensor:
        raise NotImplementedError
    
    def generate(self, policy, batch) -> dict:
        raise NotImplementedError  # PPO/GRPO만 구현
```

```python
class DPOAlgorithm(BaseAlgorithm):
    needs_generation = False
    
    def compute_loss(self, policy_outputs, frozen_outputs, batch):
        # policy_outputs: chosen/rejected forward 결과
        # frozen_outputs["reference"]: ref model forward 결과
        # → DPO loss (~15줄)

class WeightedNTPAlgorithm(BaseAlgorithm):
    needs_generation = False
    
    def compute_loss(self, policy_outputs, frozen_outputs, batch):
        # frozen_outputs["critic"]: 토큰별 value
        # → GAE → AWR weights → weighted CE (~30줄)
```

---

## Multi-model Strategy 처리

### 현재 Strategy 인터페이스

```python
class BaseStrategy:
    def setup(self, model, device, optimizer=None) -> nn.Module  # 1개 모델
```

### 확장

Strategy에 `setup_models` 메서드를 추가하되, 기존 `setup`은 유지 (SFT 호환):

```python
class BaseStrategy:
    def setup(self, model, device, optimizer=None) -> nn.Module:       # 기존
    def setup_models(self, models: dict, device, ...) -> dict:         # RL용

class FSDPStrategy(BaseStrategy):
    def setup_models(self, models, device, ...):
        wrapped = {}
        for name, model in models.items():
            if any(p.requires_grad for p in model.parameters()):
                # trainable → FULL_SHARD
                wrapped[name] = FSDP(model, sharding_strategy=FULL_SHARD, ...)
            else:
                # frozen → NO_SHARD (forward only)
                wrapped[name] = FSDP(model, sharding_strategy=NO_SHARD, ...)
        return wrapped
```

---

## 공유 유틸리티 (`rl_utils.py`)

모든 알고리즘이 공유하는 3개 함수:

| 함수 | 입력 | 출력 | 사용 |
|------|------|------|------|
| `compute_log_probs(logits, labels)` | (batch, seq, vocab), (batch, seq) | (batch, seq) per-token log_prob | DPO, PPO, GRPO |
| `compute_kl(policy_logits, ref_logits)` | 두 모델의 logits | scalar KL divergence | DPO, PPO, GRPO |
| `masked_mean(tensor, mask)` | tensor + bool mask | scalar | 전부 |

---

## 구현 순서 (점진적)

### Phase 1: DPO (가장 단순, 인프라 검증)

DPO는 generation이 없고, 모델 2개(policy + ref)뿐이며, loss가 ~15줄이다. **multi-model 인프라를 검증하기에 최적**.

| Step | 작업 | 파일 |
|:---:|------|------|
| 1-1 | RLRecipe Pydantic 스키마 | `settings/rl_schema.py` |
| 1-2 | rl_utils (log_probs, kl, masked_mean) | `training/rl/rl_utils.py` |
| 1-3 | BaseAlgorithm + DPOAlgorithm | `training/rl/algorithms/dpo.py` |
| 1-4 | RLTrainer (step-based 루프, multi-model) | `training/rl/rl_trainer.py` |
| 1-5 | Strategy.setup_models | `training/strategies/{base,ddp,fsdp}.py` |
| 1-6 | CLI `mdp rl-train` | `cli/rl_train.py` + `__main__.py` |
| 1-7 | 테스트 (gloo CPU, TinyModel DPO) | `tests/e2e/test_rl_dpo.py` |

### Phase 2: weighted-NTP (Critic + GAE + AWR)

DPO 인프라 위에 critic forward + GAE advantage + AWR weighting을 추가.

| Step | 작업 |
|:---:|------|
| 2-1 | WeightedNTPAlgorithm (GAE + AWR + weighted CE) |
| 2-2 | 테스트 |

### Phase 3: GRPO (Generation 추가)

Generation 루프가 처음 등장. policy.generate() → reward scoring → group advantage.

### Phase 4: PPO (Value model + Clipped surrogate)

가장 복잡. Value model 학습 + clipped surrogate objective.

---

## 기존 코드와의 접점

| 기존 컴포넌트 | RL에서의 역할 | 수정 필요 |
|-------------|------------|:---:|
| ComponentResolver | optimizer/scheduler 생성 | 없음 |
| PreferenceCollator | DPO 데이터 처리 | 없음 |
| ModelCheckpoint | policy 체크포인트 저장 | 없음 (policy만 전달) |
| EarlyStopping | reward 기반 조기 종료 | 없음 (monitor metric만 다름) |
| MLflow | params/metrics/artifact | 없음 |
| `mdp export` | policy 모델 내보내기 | 없음 |
| `mdp serve` | policy 모델 서빙 | 없음 |
| Strategy (DDP/FSDP) | multi-model sharding | **setup_models 추가** |

**기존 SFT 코드 수정**: Strategy에 `setup_models` 메서드 추가 (기존 `setup`은 유지).
**그 외 기존 코드 변경 0건.**

---

## 예상 코드량

| 컴포넌트 | 줄 수 |
|----------|:---:|
| rl_schema.py | ~120 |
| rl_utils.py | ~60 |
| rl_trainer.py | ~350 |
| algorithms/dpo.py | ~80 |
| algorithms/weighted_ntp.py | ~120 |
| algorithms/grpo.py | ~100 |
| algorithms/ppo.py | ~250 |
| cli/rl_train.py | ~80 |
| Strategy setup_models | ~40 |
| 테스트 | ~200 |
| **합계** | **~1,400** |

현재 MDP가 ~7,700줄. RL 추가 후 ~9,100줄 (18% 증가). 기존 코드 변경은 Strategy의 40줄뿐.
