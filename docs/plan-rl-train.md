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

- 기존 `mdp train` (SFT)의 `trainer.py`는 **변경하지 않는다**
- RL 전용 코드는 `training/` 하위에 **같은 레벨**로 배치 (하위 디렉토리 중첩 없음)
- 학습 결과는 policy 모델 1개 → 기존 `mdp export` / `mdp serve`로 서빙
- 기존 인프라 (Strategy, MLflow, Checkpoint, ComponentResolver, PreferenceCollator) 최대 재사용
- loss 함수는 별도 폴더 없이 `losses/rl.py` 한 파일에 (알고리즘당 15~30줄)
- 표준 loss (FocalLoss 등)는 `_component_` 패턴으로 외부에서 자유롭게 주입 가능 (기존과 동일)

---

## 디렉토리 구조

```
mdp/
  training/
    trainer.py              # SFT Trainer (변경 없음)
    rl_trainer.py           # RL Trainer (SFT와 병렬, 같은 레벨)
    losses/                 # loss 함수 모음 (strategies/, callbacks/와 동일 패턴)
      __init__.py
      rl.py                 # DPO, weighted-NTP, GRPO, PPO loss + 공유 유틸
      dl.py                 # (향후) FocalLoss, QuantileLoss 등 내장 loss
    strategies/             # DDP, FSDP (기존 + setup_models 메서드 추가)
    callbacks/              # EarlyStopping, Checkpoint (기존, 변경 없음)
  settings/
    schema.py               # Recipe에 models/algorithm 필드 추가
  factory/
    factory.py              # create_models() 메서드 추가
  cli/
    rl_train.py             # mdp rl-train CLI (신규)
```

**신규 파일 3개**: `rl_trainer.py`, `losses/rl.py`, `cli/rl_train.py`
**확장 파일 3개**: `schema.py`, `factory.py`, `strategies/{base,ddp,fsdp}.py`
**변경 없는 파일**: `trainer.py`, callbacks, 기존 CLI

---

## 개발자 시점의 흐름

```
"RL DPO 학습이 어떻게 돌아가지?"

cli/rl_train.py          → 진입점. Settings 로딩, Factory.create_models(), RLTrainer 호출
settings/schema.py       → Recipe.models (복수), Recipe.algorithm, Recipe.dpo.beta
factory/factory.py       → create_models(): 역할별 모델 로딩 + frozen 설정
training/rl_trainer.py   → RL 학습 루프 (frozen forward → policy forward → loss → backward)
training/losses/rl.py    → dpo_loss() (~15줄)
```

SFT 흐름과 비교:
```
cli/train.py             → 진입점
settings/schema.py       → Recipe.model (단수)
factory/factory.py       → create_model()
training/trainer.py      → SFT 학습 루프
```

같은 패턴. RL이 파일 1개 더 있을 뿐 (losses/rl.py).

---

## Recipe YAML

```yaml
algorithm: dpo          # dpo | weighted_ntp | grpo | ppo. 없으면 SFT (mdp train)

models:
  policy:
    class_path: transformers.AutoModelForCausalLM
    pretrained: hf://meta-llama/Meta-Llama-3-8B
    torch_dtype: bfloat16
    adapter: {method: lora, r: 64, alpha: 128}
    optimizer:                        # 모델별 optimizer
      _component_: AdamW
      lr: 5e-7
      weight_decay: 0.01
    scheduler:
      _component_: CosineAnnealingLR
      T_max: 2000
      warmup_ratio: 0.1
  reference:
    pretrained: hf://meta-llama/Meta-Llama-3-8B
    torch_dtype: bfloat16
    # optimizer 없음 + freeze: true (기본) → frozen

dpo:
  beta: 0.1

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

metadata:
  author: wesley
  description: "LLaMA-3 8B DPO alignment"
```

기존 Recipe 스키마에 `models`, `algorithm`, 알고리즘별 config를 **optional 필드로 추가**. `models`와 `algorithm`이 None이면 기존 SFT 그대로.

**SFT vs RL의 optimizer 위치**:
- SFT: top-level `optimizer` (모델 1개니까)
- RL: `models.*.optimizer` (모델별 독립 설정)
- frozen 모델 규칙: `optimizer` 없으면 frozen. `freeze: false`인데 optimizer 없으면 검증 에러.

---

## RLTrainer 핵심 구조

```python
class RLTrainer:
    """RL alignment 학습 루프. SFT Trainer와 독립."""

    def __init__(self, settings, models: dict[str, nn.Module], train_loader, ...):
        self.trainable = {}   # optimizer가 있는 모델
        self.frozen = {}      # optimizer가 없는 모델 (frozen)
        self.optimizers = {}  # 모델별 optimizer
        self.schedulers = {}  # 모델별 scheduler

        for name, model_config in settings.recipe.models.items():
            model = models[name]
            if model_config.optimizer is not None:
                self.trainable[name] = model
                self.optimizers[name] = create_optimizer(model.parameters(), model_config.optimizer)
                if model_config.scheduler:
                    self.schedulers[name] = create_scheduler(self.optimizers[name], model_config.scheduler)
            else:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.frozen[name] = model

        self.policy = self.trainable["policy"]  # policy는 항상 trainable
        self.algorithm = settings.recipe.algorithm
        self.algo_config = getattr(settings.recipe, self.algorithm, {})

    def train(self):
        for step in range(max_steps):
            batch = next(train_iter)

            # frozen 모델 forward (no_grad)
            with torch.no_grad():
                frozen_out = {name: model(batch) for name, model in self.frozen.items()}

            # trainable 모델 forward
            trainable_out = {name: model(batch) for name, model in self.trainable.items()}

            # loss (losses/rl.py에서 가져옴)
            losses = compute_rl_loss(self.algorithm, trainable_out, frozen_out, batch, self.algo_config)
            # losses: {"policy": tensor, "value": tensor} 또는 {"policy": tensor}

            # backward — 모델별 독립 (AMP + grad_accum + clip)
            for name, loss in losses.items():
                self.scaler.scale(loss).backward()
                if (step + 1) % self.grad_accum == 0:
                    self.scaler.unscale_(self.optimizers[name])
                    clip_grad_norm_(self.trainable[name].parameters(), self.grad_clip)
                    self.scaler.step(self.optimizers[name])
                    if name in self.schedulers:
                        self.schedulers[name].step()
            if (step + 1) % self.grad_accum == 0:
                self.scaler.update()
                for opt in self.optimizers.values():
                    opt.zero_grad()
```

**optimizer가 있는 모델만 trainable, 없는 모델은 frozen** — Recipe의 models.*.optimizer 유무로 자동 결정.

RLTrainer가 SFT Trainer에서 "복사"하는 패턴: AMP setup, gradient accumulation, gradient clipping, MLflow 로깅, callbacks, checkpoint. **상속이 아닌 동일 패턴의 독립 구현**. 이유: SFT train()과 RL train()의 루프 구조가 다르므로(epoch vs step, 단일 모델 vs multi-model), 상속하면 오버라이드 체인이 복잡해진다.

---

## losses/rl.py 구조

```python
# 공유 유틸 (파일 상단)
def compute_log_probs(logits, labels): ...      # ~10줄
def masked_mean(tensor, mask): ...               # ~5줄
def compute_kl(policy_logits, ref_logits): ...   # ~8줄

# 알고리즘별 loss
def dpo_loss(policy_out, ref_out, batch, beta): ...             # ~15줄
def weighted_ntp_loss(policy_out, critic_out, batch, config): ... # ~30줄
def grpo_loss(policy_out, ref_out, batch, config): ...           # ~20줄
def ppo_loss(policy_out, ref_out, value_out, batch, config): ... # ~30줄

# 라우터
def compute_rl_loss(algorithm, policy_out, frozen_out, batch, config):
    if algorithm == "dpo":
        return dpo_loss(policy_out, frozen_out["reference"], batch, config.get("beta", 0.1))
    elif algorithm == "weighted_ntp":
        return weighted_ntp_loss(policy_out, frozen_out["critic"], batch, config)
    ...
```

반환값은 `dict[str, Tensor]` — 모델별 loss. DPO는 `{"policy": loss}`, PPO는 `{"policy": policy_loss, "value": value_loss}`. RLTrainer가 모델별로 독립 backward.

전체 ~150줄. 한 파일에서 알고리즘 비교/추가가 쉽다.

---

## Multi-model Strategy

기존 `setup(model, device)` 유지 + `setup_models(models, device)` 추가:

```python
class BaseStrategy:
    def setup(self, model, device, ...): ...           # SFT용 (기존)
    def setup_models(self, models, device, ...): ...   # RL용 (신규)

class FSDPStrategy(BaseStrategy):
    def setup_models(self, models, device, ...):
        wrapped = {}
        for name, model in models.items():
            if any(p.requires_grad for p in model.parameters()):
                wrapped[name] = FSDP(model, sharding_strategy=FULL_SHARD, ...)
            else:
                wrapped[name] = FSDP(model, sharding_strategy=NO_SHARD, ...)
        return wrapped
```

---

## 기존 코드와의 접점

| 기존 컴포넌트 | RL에서의 역할 | 수정 |
|-------------|------------|:---:|
| ComponentResolver | optimizer/scheduler 생성 | 없음 |
| PreferenceCollator | DPO 데이터 처리 | 없음 |
| ModelCheckpoint | policy 체크포인트 저장 | 없음 |
| EarlyStopping | reward 기반 조기 종료 | 없음 |
| MLflow | params/metrics/artifact | 없음 |
| `mdp export` | policy 모델 내보내기 | 없음 |
| `mdp serve` / `mdp inference` | policy 모델 서빙/추론 | 없음 |
| Strategy (DDP/FSDP) | multi-model sharding | **setup_models 추가** |
| Recipe 스키마 | models/algorithm 필드 | **optional 필드 추가** |
| Factory | 복수 모델 생성 | **create_models 추가** |

**기존 SFT 동작에 영향을 주는 변경 0건.** Schema에 optional 필드를 추가하므로 기존 Recipe YAML은 그대로 동작.

---

## 구현 순서

### Phase 1: DPO (인프라 검증)

| Step | 작업 | 파일 |
|:---:|------|------|
| 1-1 | Recipe에 models/algorithm 필드 추가 | `settings/schema.py` |
| 1-2 | Factory.create_models() | `factory/factory.py` |
| 1-3 | Strategy.setup_models() | `training/strategies/{base,ddp,fsdp}.py` |
| 1-4 | losses/rl.py (compute_log_probs, dpo_loss) | `training/losses/rl.py` |
| 1-5 | RLTrainer (step-based 루프) | `training/rl_trainer.py` |
| 1-6 | CLI `mdp rl-train` | `cli/rl_train.py` + `__main__.py` |
| 1-7 | 테스트 (gloo CPU, TinyModel DPO) | `tests/e2e/test_rl_dpo.py` |

### Phase 2: weighted-NTP

DPO 인프라 위에 critic forward + GAE + AWR + weighted CE 추가. `losses/rl.py`에 `weighted_ntp_loss` 함수 추가.

### Phase 3: GRPO

Generation 루프 추가. `rl_trainer.py`에 generation step. `losses/rl.py`에 `grpo_loss`.

### Phase 4: PPO

Value model 학습 + clipped surrogate. 가장 복잡하지만 Phase 1~3의 인프라를 전부 활용.

---

## 예상 코드량

| 파일 | 줄 수 | 유형 |
|------|:---:|:---:|
| `training/rl_trainer.py` | ~350 | 신규 |
| `training/losses/__init__.py` | ~5 | 신규 |
| `training/losses/rl.py` | ~150 | 신규 |
| `cli/rl_train.py` | ~80 | 신규 |
| `settings/schema.py` | +50 | 확장 |
| `factory/factory.py` | +40 | 확장 |
| `training/strategies/*.py` | +40 | 확장 |
| `__main__.py` | +10 | 확장 |
| 테스트 | ~200 | 신규 |
| **합계** | **~920** | |

현재 MDP ~7,700줄. RL 추가 후 ~8,600줄 (12% 증가).
