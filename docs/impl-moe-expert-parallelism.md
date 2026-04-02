# MoE Expert Parallelism — 구현 명세

`plan-moe-expert-parallelism.md`의 설계를 코드베이스 현재 상태와 대조하여, 파일별 구체적 변경사항을 기술한다. plan의 설계 의도를 그대로 따르되, 코드 분석 과정에서 발견된 불일치와 보완 사항을 반영한다.

---

## 1. 왜 이 변경이 필요한가

MDP는 현재 3가지 분산 전략(DDP, FSDP, DeepSpeed)을 지원한다. 이 전략들은 모든 파라미터를 균등하게 취급한다 — "이 레이어가 attention인지 expert인지"는 고려하지 않는다.

MoE(Mixture-of-Experts) 모델에서는 이 균등 취급이 심각한 비효율을 만든다. Mixtral 8x7B를 예로 들면, 각 디코더 레이어에 8개의 expert FFN이 있고 입력 토큰의 top-2만 활성화된다. 현재 FSDP의 `size_based_auto_wrap_policy`는 각 expert(58.7M params)를 개별 FSDP unit으로 래핑하므로, 토큰이 특정 expert로 라우팅될 때마다 4개 GPU에서 all-gather로 전체 파라미터를 복원해야 한다. Expert Parallelism은 각 GPU가 expert를 통째로 보유하고, 토큰만 해당 GPU로 전송(all-to-all)하는 방식으로 이 통신량을 약 56배 줄인다.

plan은 이 문제를 3단계로 해결한다:
- **Phase 1**: 기존 FSDP에 MoE-aware wrapping을 추가하여 "일단 돌아가게" 만든다
- **Phase 2**: 전용 MoEStrategy를 신규 구현하여 expert parallelism을 실현한다
- **Phase 3**: DeepSpeed의 검증된 MoE 지원을 통합한다

이 문서는 각 Phase의 변경사항을 파일 단위로 구체화한다. 다음 섹션에서는 먼저 현재 코드가 어떤 구조인지를 확인하고, 그 위에 어떤 변경을 가하는지를 기술한다.

---

## 2. 현재 코드 구조 — 변경 대상 파일들

이 섹션은 변경 대상 파일들의 현재 상태를 정리한다. 코드를 직접 읽고 확인한 결과이며, plan 문서의 참조 라인 번호와 실제 코드가 다른 부분은 실제 코드를 기준으로 기술한다.

### 2.1 전략 생성 흐름

YAML의 `config.compute.distributed` 딕셔너리가 전략 인스턴스가 되기까지의 경로:

```
config.yaml
  compute:
    distributed:
      strategy: fsdp
      sharding_strategy: FULL_SHARD
      mixed_precision: true
          │
          ▼
Settings.config.compute.distributed  (dict[str, Any] | None)
          │
          ▼  Trainer._create_strategy() [trainer.py:209-233]
          │
          │  1. strategy 키로 STRATEGY_MAP에서 class_path 조회
          │  2. strategy 키를 제외한 나머지를 kwargs로 추출
          │  3. resolver.resolve({"_component_": class_path, **kwargs})
          │
          ▼
FSDPStrategy(sharding_strategy="FULL_SHARD", mixed_precision=True, ...)
```

**핵심 포인트**: `distributed` 필드가 `dict[str, Any]`이므로, 새로운 키(`auto_wrap_cls`, `moe` 등)를 추가해도 스키마 변경 없이 전략의 `__init__` kwargs로 전달된다. 이것이 plan의 Phase 1이 스키마를 건드리지 않고 동작할 수 있는 이유다.

### 2.2 RL Trainer의 전략 생성 — 발견된 문제

RL Trainer(`rl_trainer.py:137-151`)의 `_create_strategy()`는 SFT Trainer와 **다른 패턴**을 사용한다:

```python
# rl_trainer.py:151 — kwargs를 전달하지 않음
return self.resolver.resolve({"_component_": class_path})

# trainer.py:231-233 — kwargs를 전달함
strategy_kwargs = {k: v for k, v in dist_config.items() if k != "strategy"}
return self.resolver.resolve({"_component_": class_path, **strategy_kwargs})
```

RL Trainer는 `strategy` 이름만으로 전략을 생성하고, `sharding_strategy`, `mixed_precision` 같은 나머지 설정을 **무시**한다. 현재 코드에서 이 문제가 표면화되지 않은 이유는 RL 학습이 아직 프로토타입 단계(GRPO/PPO)이고 분산 테스트가 제한적이기 때문이다. 그러나 MoE 전략은 `ep_size`, `decoder_layer_cls` 같은 필수 파라미터를 kwargs로 받아야 하므로, 이 문제를 Phase 1에서 선행 수정해야 한다.

### 2.3 FSDP 전략 현재 구현

`training/strategies/fsdp.py`의 현재 구조:

```python
class FSDPStrategy(BaseStrategy):
    def __init__(
        self,
        sharding_strategy: str = "FULL_SHARD",
        mixed_precision: bool = True,
        backend: str = "nccl",
        cpu_offload: bool = False,
        precision: str = "bf16",
        min_num_params: int = 1_000_000,    # ← auto_wrap 기준
    ) -> None:
```

`setup()` 내 auto_wrap_policy 생성 (line 75-77):

```python
fsdp_kwargs: dict[str, Any] = {
    "sharding_strategy": sharding,
    "auto_wrap_policy": functools.partial(
        size_based_auto_wrap_policy, min_num_params=self.min_num_params,
    ),
}
```

6개 파라미터, `setup()` 메서드 44줄(line 53-96). `size_based_auto_wrap_policy`만 사용하며, transformer layer class 기반 wrapping은 지원하지 않는다.

### 2.4 DeepSpeed 전략 현재 구현

`training/strategies/deepspeed.py`:

```python
_DEFAULT_DS_CONFIG: dict[str, Any] = {
    "zero_optimization": {"stage": 2},
    "bf16": {"enabled": True},
    "gradient_clipping": 1.0,
}

class DeepSpeedStrategy(BaseStrategy):
    def __init__(
        self,
        ds_config: dict[str, Any] | None = None,
        batch_size: int = 32,
    ) -> None:
```

2개 파라미터, `ds_config`가 None이면 기본값(ZeRO-2, bf16) 사용. MoE 관련 설정은 없다.

### 2.5 Validation 현재 구조

`settings/validation/compat_validator.py` (104줄)는 3개 규칙을 검증한다:
1. `_check_gpu_distributed` — GPU 수 vs 전략 정합성
2. `_check_serving_backend` — vLLM 호환성
3. `_check_fsdp_qlora` — FSDP + QLoRA 비호환

각 규칙은 `@staticmethod`이고 `(settings, result)` 시그니처를 따른다. 새 MoE 검증 규칙도 이 패턴을 따르면 된다.

### 2.6 Model Catalog 현재 구조

`models/catalog/text_generation/`에 11개 YAML이 있다. 공통 필드:

```yaml
name: llama3-8b
family: llama
class_path: transformers.AutoModelForCausalLM
head_builtin: true
pretrained_sources: [...]
supported_tasks: [...]
default_head: { task, _component_, hidden_dim, vocab_size }
adapter_defaults:
  lora: { target_modules, r, alpha, task_type }
memory: { params_m, fp16_gb, qlora_4bit_gb }
```

plan이 제안하는 `moe:` 섹션은 기존 구조의 자연스러운 확장이다. `architecture: moe` 필드는 제거했다 — MoE 여부는 모델 자체의 속성이므로 `model.config.num_local_experts`에서 자동 감지하며, catalog에 별도 필드를 둘 필요가 없다.

### 2.7 aliases.yaml 현재 구조

```yaml
strategy:
  DDPStrategy: mdp.training.strategies.ddp.DDPStrategy
  FSDPStrategy: mdp.training.strategies.fsdp.FSDPStrategy
  DeepSpeedStrategy: mdp.training.strategies.deepspeed.DeepSpeedStrategy
```

### 2.8 STRATEGY_MAP (trainer.py:30-35)

```python
STRATEGY_MAP: dict[str, str] = {
    "ddp": "mdp.training.strategies.ddp.DDPStrategy",
    "fsdp": "mdp.training.strategies.fsdp.FSDPStrategy",
    "deepspeed": "mdp.training.strategies.deepspeed.DeepSpeedStrategy",
    "deepspeed_zero3": "mdp.training.strategies.deepspeed.DeepSpeedStrategy",
}
```

`deepspeed`와 `deepspeed_zero3`가 같은 클래스를 가리키는 이유: 둘 다 `DeepSpeedStrategy`이지만 config의 `ds_config.zero_optimization.stage`로 stage를 구분한다. MoE 전략도 같은 패턴으로 등록할 수 있다.

이제 현재 코드 구조를 확인했으므로, 각 Phase의 구체적 변경사항을 기술한다.

---

## 3. Phase 1: MoE-Aware FSDP Wrapping

Phase 1의 목표는 MoE 모델이 기존 FSDP로 **학습 가능**하게 만드는 것이다. 최적의 성능이 아니라 "동작하는 상태"가 목표다. 핵심은 두 가지: (1) transformer layer class 기반의 wrapping 정책으로 expert를 올바르게 래핑하고, (2) MoE 모델에서 HYBRID_SHARD를 자동 적용하여 inter-node 통신을 줄이는 것이다.

### 3.1 선행 수정: RL Trainer kwargs 전달 (`training/rl_trainer.py`)

Phase 1의 어떤 변경도 RL Trainer에서 동작하려면, 먼저 kwargs 전달 문제를 수정해야 한다.

**변경 위치**: `rl_trainer.py` `_create_strategy()` 메서드 (line 137-151)

**현재**:
```python
def _create_strategy(self, settings: Settings) -> Any:
    from mdp.training.trainer import STRATEGY_MAP
    dist_config = settings.config.compute.distributed
    if dist_config is None:
        return None
    strategy_name = dist_config.get("strategy", "auto") if isinstance(dist_config, dict) else "auto"
    if strategy_name in ("none", "auto"):
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
            return None
        strategy_name = "ddp"
    class_path = STRATEGY_MAP.get(strategy_name)
    if class_path is None:
        raise ValueError(f"알 수 없는 분산 전략: {strategy_name}")
    return self.resolver.resolve({"_component_": class_path})
```

**변경 후**:
```python
def _create_strategy(self, settings: Settings) -> Any:
    from mdp.training.trainer import STRATEGY_MAP
    dist_config = settings.config.compute.distributed
    if dist_config is None:
        return None
    strategy_name = dist_config.get("strategy", "auto") if isinstance(dist_config, dict) else "auto"
    if strategy_name in ("none", "auto"):
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
            return None
        strategy_name = "ddp"
    class_path = STRATEGY_MAP.get(strategy_name)
    if class_path is None:
        raise ValueError(f"알 수 없는 분산 전략: {strategy_name}")
    strategy_kwargs = {
        k: v for k, v in (dist_config if isinstance(dist_config, dict) else {}).items()
        if k != "strategy"
    }
    return self.resolver.resolve({"_component_": class_path, **strategy_kwargs})
```

**변경 이유**: SFT Trainer(trainer.py:227-233)와 동일한 패턴으로 통일한다. 이 수정이 없으면 RL 학습에서 어떤 전략 파라미터도 전달되지 않는다 — `sharding_strategy`, `precision` 같은 기존 파라미터조차 무시되므로, 이 문제는 MoE와 무관하게 수정이 필요한 기존 결함이다.

**영향 범위**: RL Trainer를 사용하는 기존 코드에서 `distributed` dict에 `strategy` 외의 키가 있었다면 이전에는 무시되던 것이 이제 전략 생성자에 전달된다. 기존 RL 테스트가 `strategy: ddp` 만으로 구성되어 있다면 동작에 변화가 없다.

### 3.2 FSDP 전략 확장 (`training/strategies/fsdp.py`)

**변경 1: `__init__`에 2개 파라미터 추가** (line 32-46)

현재:
```python
def __init__(
    self,
    sharding_strategy: str = "FULL_SHARD",
    mixed_precision: bool = True,
    backend: str = "nccl",
    cpu_offload: bool = False,
    precision: str = "bf16",
    min_num_params: int = 1_000_000,
) -> None:
    self.sharding_strategy_name = sharding_strategy
    self.mixed_precision = mixed_precision
    self.backend = backend
    self.cpu_offload = cpu_offload
    self.precision = precision
    self.min_num_params = min_num_params
    self._local_rank: int | None = None
```

변경 후:
```python
def __init__(
    self,
    sharding_strategy: str = "FULL_SHARD",
    mixed_precision: bool = True,
    backend: str = "nccl",
    cpu_offload: bool = False,
    precision: str = "bf16",
    min_num_params: int = 1_000_000,
    auto_wrap_cls: str | None = None,
    moe: dict | None = None,
) -> None:
    self.sharding_strategy_name = sharding_strategy
    self.mixed_precision = mixed_precision
    self.backend = backend
    self.cpu_offload = cpu_offload
    self.precision = precision
    self.min_num_params = min_num_params
    self.auto_wrap_cls = auto_wrap_cls
    self.moe_config = moe or {}
    self._local_rank: int | None = None
```

`auto_wrap_cls`와 `moe`의 기본값이 각각 `None`과 `None`이므로, 기존 YAML config는 아무 변경 없이 동작한다.

**변경 2: `setup()` 내 auto_wrap_policy 분기** (line 73-78)

현재:
```python
fsdp_kwargs: dict[str, Any] = {
    "sharding_strategy": sharding,
    "auto_wrap_policy": functools.partial(
        size_based_auto_wrap_policy, min_num_params=self.min_num_params,
    ),
}
```

변경 후:
```python
if self.auto_wrap_cls is not None:
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    layer_cls = self._resolve_layer_class(self.auto_wrap_cls)
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_cls},
    )
else:
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=self.min_num_params,
    )

# MoE 모델에서 FULL_SHARD → HYBRID_SHARD 자동 전환
actual_sharding = sharding
if self.moe_config.get("enabled", False):
    from torch.distributed.fsdp import ShardingStrategy as SS
    if actual_sharding == SS.FULL_SHARD:
        logger.info(
            "MoE 모델: FULL_SHARD → HYBRID_SHARD 자동 전환 "
            "(노드 내 sharding으로 expert 통신 최소화)"
        )
        actual_sharding = SS.HYBRID_SHARD

fsdp_kwargs: dict[str, Any] = {
    "sharding_strategy": actual_sharding,
    "auto_wrap_policy": auto_wrap_policy,
}
```

`auto_wrap_cls`가 지정되면 `transformer_auto_wrap_policy`를 사용한다. 이 정책은 지정된 클래스의 인스턴스를 개별 FSDP unit으로 래핑한다. MoE 모델에서 이것이 중요한 이유: `MixtralDecoderLayer` 단위로 래핑하면 그 안의 attention과 expert가 하나의 FSDP unit에 포함되어, expert별 불필요한 all-gather를 피할 수 있다.

HYBRID_SHARD 자동 전환의 이유: FULL_SHARD는 전체 GPU에 걸쳐 파라미터를 분할하므로 노드 간 통신이 발생한다. HYBRID_SHARD는 노드 내에서만 sharding하고 노드 간에는 replicate한다. MoE의 expert가 이미 sparse하게 활성화되므로, 노드 내 NVLink의 빠른 통신으로 충분하다.

**변경 3: `_resolve_layer_class` 헬퍼 추가** (클래스 끝에 추가)

```python
@staticmethod
def _resolve_layer_class(cls_name: str) -> type:
    """클래스 이름 또는 전체 경로에서 클래스를 resolve한다."""
    import importlib

    if "." in cls_name:
        module_path, _, class_name = cls_name.rpartition(".")
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    # 단축명: transformers 패키지에서 탐색
    try:
        import transformers
        cls = getattr(transformers, cls_name, None)
        if cls is not None:
            return cls
    except ImportError:
        pass
    raise ValueError(
        f"transformer layer class '{cls_name}'를 찾을 수 없습니다. "
        "전체 경로를 사용하세요 (예: transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer)"
    )
```

이 헬퍼는 `ComponentResolver`의 패턴과 유사하게 동작한다: 점(`.`)이 있으면 전체 경로로 import하고, 없으면 `transformers` 패키지에서 탐색한다. plan의 설계를 그대로 따르되, `transformers` import 실패를 안전하게 처리한다.

**변경 4: `setup()` 상단에 logger import** (line 1 영역)

파일 최상단 import에 `logging` 추가:

```python
import logging
# ... 기존 imports ...
logger = logging.getLogger(__name__)
```

현재 `fsdp.py`에는 logger가 없다. HYBRID_SHARD 자동 전환 로그를 위해 추가한다.

### 3.3 Model Catalog 추가

**신규 파일**: `models/catalog/text_generation/mixtral-8x7b.yaml`

plan의 설계를 그대로 사용한다. 기존 catalog의 공통 필드(`name`, `family`, `class_path`, `head_builtin`, `pretrained_sources`, `supported_tasks`, `default_head`, `adapter_defaults`, `memory`)를 모두 포함하고, MoE 전용 필드를 `moe:` 섹션에 추가한다:

```yaml
name: mixtral-8x7b
family: mixtral
class_path: transformers.AutoModelForCausalLM
head_builtin: true
pretrained_sources:
  - "hf://mistralai/Mixtral-8x7B-v0.1"
  - "hf://mistralai/Mixtral-8x7B-Instruct-v0.1"

supported_tasks:
  - text_generation

default_head:
  task: text_generation
  _component_: CausalLMHead
  hidden_dim: 4096
  vocab_size: 32000

moe:
  num_experts: 8
  top_k: 2
  decoder_layer_cls: MixtralDecoderLayer
  expert_module_pattern: "block_sparse_moe.experts"
  router_module_pattern: "block_sparse_moe.gate"

adapter_defaults:
  lora:
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
    r: 16
    alpha: 32
    task_type: "CAUSAL_LM"
  lora_with_experts:
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"]
    r: 8
    alpha: 16
    task_type: "CAUSAL_LM"

memory:
  params_m: 46700
  active_params_m: 12900
  fp16_gb: 93.4
  qlora_4bit_gb: 26.0
```

`moe:` 섹션은 catalog의 새 필드다. CatalogValidator가 이 필드를 적극적으로 검증하지 않으므로(catalog는 warnings-only), 바로 추가해도 기존 코드를 깨뜨리지 않는다. MoE 자동 감지는 catalog가 아니라 로딩된 모델의 `config.num_local_experts`에서 수행한다 — catalog의 `moe:` 섹션은 사용자에게 모델의 MoE 구성 정보를 제공하는 참조 데이터다.

`adapter_defaults.lora_with_experts`는 기존 패턴의 확장이다 — 현재 catalog에서 `adapter_defaults` 키는 method name(`lora`, `qlora`)으로 구분된다. `lora_with_experts`는 사용자가 Recipe에서 명시적으로 선택하는 대안 preset이다.


### 3.4 Phase 1 테스트

Phase 1의 핵심 검증 포인트:

```
tests/e2e/
  test_fsdp_moe_wrap.py  (신규)
    ├── test_fsdp_default_size_based_policy
    │   └── auto_wrap_cls 미지정 시 기존 size_based 동작 확인
    ├── test_fsdp_transformer_auto_wrap
    │   └── auto_wrap_cls 지정 시 transformer_auto_wrap_policy 적용 확인
    ├── test_fsdp_hybrid_shard_auto_switch
    │   └── moe.enabled=True + FULL_SHARD → HYBRID_SHARD 자동 전환
    ├── test_fsdp_hybrid_shard_no_switch_when_already_set
    │   └── 이미 HYBRID_SHARD면 전환하지 않음
    └── test_resolve_layer_class_full_path / test_resolve_layer_class_short_name
        └── _resolve_layer_class 단위 테스트
```

### 3.5 Phase 1 요약

| 파일 | 변경 유형 | 줄 수 (추정) |
|------|----------|------------|
| `training/rl_trainer.py` | 수정 (kwargs 전달) | +4 |
| `training/strategies/fsdp.py` | 확장 (auto_wrap + HYBRID_SHARD) | +45 |
| `models/catalog/text_generation/mixtral-8x7b.yaml` | 신규 | +40 |
| `tests/e2e/test_fsdp_moe_wrap.py` | 신규 | +60 |

---

## 4. Phase 2: Expert Parallelism Strategy

Phase 1이 "FSDP로 MoE를 돌릴 수 있게" 만들었다면, Phase 2는 **전용 전략**을 구현하여 expert parallelism의 성능 이점을 실현한다. 이것이 전체 구현의 핵심이자 최고 난이도 부분이다.

### 4.1 왜 별도 전략이 필요한가

Phase 1의 FSDP + HYBRID_SHARD는 expert를 올바르게 래핑하지만, 여전히 FSDP의 all-gather/reduce-scatter 패턴을 따른다. Expert Parallelism은 근본적으로 다른 통신 패턴(token all-to-all dispatch)을 사용하므로 FSDP의 확장이 아니라 별도 전략이 필요하다.

두 접근의 통신 패턴 차이:

```
FSDP (Phase 1):
  매 expert forward마다:
    all-gather(expert 파라미터) → forward → reduce-scatter(gradient)
  통신 대상: 파라미터 (수십~수백 MB per expert)

Expert Parallelism (Phase 2):
  매 MoE layer마다:
    all-to-all(토큰 hidden states) → local forward → all-to-all(결과)
  통신 대상: 활성 토큰의 hidden state (수 MB)
```

### 4.2 신규 전략 파일 (`training/strategies/moe.py`)

**예상 규모**: ~300줄

MoEStrategy는 BaseStrategy를 상속하고, 4개 추상 메서드(`setup`, `save_checkpoint`, `load_checkpoint`, `cleanup`)를 구현한다. 추가로 `setup_models()`를 오버라이드하여 RL Trainer의 다중 모델 시나리오를 지원한다.

**클래스 구조**:

```python
class MoEStrategy(BaseStrategy):
    """Expert Parallelism + Data Parallelism 전략.

    Expert를 GPU에 분배하고, 공유 레이어는 FSDP로 sharding한다.
    MoE 레이어의 forward에 all-to-all token dispatch를 삽입한다.
    """

    def __init__(
        self,
        ep_size: int,                        # Expert Parallel degree
        precision: str = "bf16",
        backend: str = "nccl",
        decoder_layer_cls: str | None = None,
        expert_module_pattern: str = "experts",
    ) -> None: ...

    # ── BaseStrategy 구현 ──
    def setup(self, model, device, optimizer=None) -> nn.Module: ...  # dist.is_initialized() guard 포함
    def save_checkpoint(self, model, path) -> None: ...
    def load_checkpoint(self, model, path) -> nn.Module: ...
    def cleanup(self) -> None: ...

    # ── RL 지원 ──
    # RL Trainer는 setup_models()가 있으면 우선 호출한다 (rl_trainer.py:310-318).
    # MoEStrategy에서는 trainable 모델에만 EP hook을 설치하고,
    # frozen 모델(reference policy 등)은 NO_SHARD로 처리한다.
    def setup_models(self, models, device, trainable_names=None) -> dict: ...

    # ── 내부 메서드 ──
    def _create_process_groups(self, rank, world_size) -> None: ...
    def _detect_num_experts(self, model, config) -> int: ...
    def _install_ep_hooks(self, model) -> None: ...
    def _make_ep_forward(self, moe_module, original_forward) -> callable: ...
    def _wrap_shared_with_fsdp(self, model, device) -> nn.Module: ...
    def _distribute_experts(self, model, device) -> None: ...
    def _is_moe_layer(self, name, module) -> bool: ...
```

**setup() 흐름**:

```
setup(model, device, optimizer)
  │
  ├── 1. Process group 초기화 (FSDPStrategy와 동일한 `if not dist.is_initialized()` guard 적용)
  │     init_process_group → EP groups + DP groups 생성
  │     EP group: 같은 EP group 내 GPU들이 expert token을 교환
  │     DP group: 같은 위치의 GPU들이 gradient를 동기화
  │
  ├── 2. Expert 분배 계산
  │     num_experts 감지 → experts_per_gpu 계산 → GPU별 할당 범위
  │
  ├── 3. MoE forward hook 삽입
  │     _install_ep_hooks: expert module의 forward를 EP-aware로 래핑
  │     원래: router → expert forward → combine
  │     변경: router → all-to-all dispatch → local expert forward → all-to-all return
  │
  ├── 4. 공유 레이어 FSDP 적용
  │     attention 등 공유 레이어만 FSDP로 sharding
  │     expert는 FSDP 밖에서 관리
  │
  └── 5. Expert를 담당 GPU로 이동
        각 GPU는 자기 expert만 GPU 메모리에 보유
```

**all-to-all dispatch의 backward 처리**:

이 부분이 Phase 2의 최대 난관이다. `torch.autograd.Function`으로 커스텀 backward를 구현해야 한다:

```python
class AllToAllDispatch(torch.autograd.Function):
    """all-to-all token dispatch의 forward/backward."""

    @staticmethod
    def forward(ctx, input_tensor, send_counts, recv_counts, ep_group):
        # input_tensor를 send_counts에 따라 분할 → all_to_all로 교환
        ctx.save_for_backward(send_counts, recv_counts)
        ctx.ep_group = ep_group
        output = _all_to_all_tokens(input_tensor, send_counts, recv_counts, ep_group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # backward에서는 send/recv가 반전된다
        send_counts, recv_counts = ctx.saved_tensors
        grad_input = _all_to_all_tokens(grad_output, recv_counts, send_counts, ctx.ep_group)
        return grad_input, None, None, None
```

forward에서 GPU A → GPU B로 토큰을 보냈다면, backward에서는 GPU B → GPU A로 gradient를 보내야 한다. 이것이 `send_counts`와 `recv_counts`를 반전시키는 이유다.

`_all_to_all_tokens`는 `dist.all_to_all_single`을 래핑하는 헬퍼다. 각 GPU가 보내야 할 토큰을 `send_counts`에 따라 연속 블록으로 정렬한 뒤, `all_to_all_single`로 교환하고, `recv_counts`에 따라 수신된 토큰을 다시 분리한다. 핵심은 텐서 reshaping과 padding 처리 — EP group 내 GPU 수가 불균등할 수 있으므로 zero-padding 후 교환, 수신 후 unpadding한다.

**checkpoint 전략**:

Expert가 GPU별로 분산되어 있으므로, checkpoint 저장/로드도 분산 방식이어야 한다:

```python
def save_checkpoint(self, model, path):
    rank = dist.get_rank()
    # 1. 공유 레이어: rank-0만 저장 (FSDP full state dict)
    # 2. Expert: 각 GPU가 자기 expert만 저장
    #    path/expert-rank{rank}.safetensors
    # 3. rank-0이 메타데이터 저장 (어느 rank가 어느 expert를 갖고 있는지)

def load_checkpoint(self, model, path):
    # 1. 공유 레이어: 모든 rank가 rank-0의 state dict 로드
    # 2. Expert: 각 GPU가 자기 rank의 expert 파일 로드
    # 3. 메타데이터로 expert 할당 검증
```

### 4.3 전략 등록

**`trainer.py` 변경** (line 30-35):

```python
STRATEGY_MAP: dict[str, str] = {
    "ddp": "mdp.training.strategies.ddp.DDPStrategy",
    "fsdp": "mdp.training.strategies.fsdp.FSDPStrategy",
    "deepspeed": "mdp.training.strategies.deepspeed.DeepSpeedStrategy",
    "deepspeed_zero3": "mdp.training.strategies.deepspeed.DeepSpeedStrategy",
    "moe": "mdp.training.strategies.moe.MoEStrategy",
}
```

1줄 추가.

**`aliases.yaml` 변경**:

```yaml
strategy:
  DDPStrategy: mdp.training.strategies.ddp.DDPStrategy
  FSDPStrategy: mdp.training.strategies.fsdp.FSDPStrategy
  DeepSpeedStrategy: mdp.training.strategies.deepspeed.DeepSpeedStrategy
  MoEStrategy: mdp.training.strategies.moe.MoEStrategy
```

1줄 추가.

**`training/strategies/__init__.py` 변경**:

```python
from mdp.training.strategies.moe import MoEStrategy

__all__ = [
    "BaseStrategy",
    "DDPStrategy",
    "DeepSpeedStrategy",
    "FSDPStrategy",
    "MoEStrategy",
]
```

2줄 추가.

### 4.4 Factory MoE 감지 (`factory/factory.py`)

Factory에 MoE 감지를 추가하는 이유: 사용자가 Config에 `strategy: moe`를 지정했는데 모델이 MoE가 아니면 의미 없는 설정이다. 반대로, MoE 모델을 로딩했는데 전략이 일반 FSDP이면 Phase 1의 차선책은 동작하지만 경고를 줄 수 있다.

**변경 위치**: `_build_model()` 메서드 (현재 line 47-79)

pretrained 로딩 직후에 MoE 감지 로직을 삽입한다:

```python
def _build_model(self) -> nn.Module:
    # ... 기존 QLoRA 분기 ...
    model = self._load_pretrained(model_spec)

    # MoE 감지 (Phase 2)
    if self._is_moe_model(model):
        moe_info = self._extract_moe_info(model)
        self._cache["moe_info"] = moe_info
        logger.info(
            "MoE 모델 감지: %d experts, top-%d",
            moe_info.get("num_experts", "?"),
            moe_info.get("top_k", "?"),
        )

    # ... 기존 head, adapter 적용 ...
```

```python
@staticmethod
def _is_moe_model(model: nn.Module) -> bool:
    config = getattr(model, "config", None)
    if config is None:
        return False
    return hasattr(config, "num_local_experts") or hasattr(config, "num_experts")

@staticmethod
def _extract_moe_info(model: nn.Module) -> dict:
    config = model.config
    return {
        "num_experts": getattr(config, "num_local_experts", None) or getattr(config, "num_experts", None),
        "top_k": getattr(config, "num_experts_per_tok", None),
    }
```

이 감지는 HuggingFace 모델의 `config` 객체에 의존한다. Mixtral은 `num_local_experts`를 사용하고, 다른 MoE 모델은 `num_experts`를 사용할 수 있으므로 두 필드를 모두 확인한다.

### 4.5 Validation 확장 (`settings/validation/compat_validator.py`)

MoE 전략의 호환성 검증 규칙을 추가한다:

```python
class CompatValidator:
    def validate(self, settings: Settings) -> ValidationResult:
        result = ValidationResult()
        self._check_gpu_distributed(settings, result)
        self._check_serving_backend(settings, result)
        self._check_fsdp_qlora(settings, result)
        self._check_moe_distributed(settings, result)  # 추가
        return result

    @staticmethod
    def _check_moe_distributed(settings: Settings, result: ValidationResult) -> None:
        """4. MoE 전략 호환성 검증."""
        distributed = settings.config.compute.distributed
        if distributed is None:
            return
        strategy = distributed.get("strategy", "")
        if strategy != "moe":
            return

        # ep_size 필수
        ep_size = distributed.get("ep_size")
        if ep_size is None:
            result.errors.append(
                "MoE 전략에는 ep_size(Expert Parallel degree)가 필수입니다."
            )
            return

        # GPU 수가 ep_size의 배수인지 확인
        gpu_count = _resolve_gpu_count(settings.config.compute.gpus)
        if gpu_count is not None and gpu_count % ep_size != 0:
            result.errors.append(
                f"GPU 수({gpu_count})가 ep_size({ep_size})의 배수가 아닙니다. "
                f"world_size = ep_size × dp_size 관계가 성립해야 합니다."
            )
```

이 규칙은 기존 3개 규칙과 동일한 `@staticmethod` + `(settings, result)` 패턴을 따른다.

### 4.6 Phase 2 테스트

```
tests/e2e/
  test_moe_strategy.py  (신규)
    ├── test_moe_process_group_creation
    │   └── gloo CPU 2프로세스로 EP/DP group 생성 검증
    ├── test_moe_expert_distribution
    │   └── Expert가 GPU별로 올바르게 분배되는지
    ├── test_moe_all_to_all_dispatch_forward
    │   └── 토큰이 올바른 GPU로 dispatch되는지
    ├── test_moe_all_to_all_dispatch_backward
    │   └── gradient가 올바르게 역전파되는지
    ├── test_moe_checkpoint_save_load
    │   └── expert별 분산 저장/로드
    └── test_moe_validation_ep_size_required
        └── CompatValidator MoE 규칙 검증
```

분산 테스트는 `torch.multiprocessing.spawn` + gloo CPU backend로 실행한다. 실제 GPU 없이 통신 로직의 정확성을 검증할 수 있다.

### 4.7 Phase 2 요약

| 파일 | 변경 유형 | 줄 수 (추정) |
|------|----------|------------|
| `training/strategies/moe.py` | 신규 | ~300 |
| `training/strategies/__init__.py` | 수정 (export 추가) | +2 |
| `training/trainer.py` | 수정 (STRATEGY_MAP) | +1 |
| `factory/factory.py` | 수정 (MoE 감지) | +20 |
| `settings/validation/compat_validator.py` | 수정 (MoE 규칙) | +20 |
| `aliases.yaml` | 수정 (MoEStrategy 추가) | +1 |
| `tests/e2e/test_moe_strategy.py` | 신규 | ~120 |

---

## 5. Phase 3: DeepSpeed MoE 통합

Phase 2가 MDP 자체 구현이라면, Phase 3는 DeepSpeed의 검증된 MoE 인프라를 활용한다. DeepSpeed는 `deepspeed.moe.layer.MoE`로 expert parallelism, load balancing loss, token dropping 등을 내장 제공한다. Phase 2의 자체 구현 대비 검증된 인프라 위에서 안정성이 높다.

### 5.1 DeepSpeed 전략 확장 (`training/strategies/deepspeed.py`)

**변경**: `__init__`에 `moe` 파라미터 추가

현재:
```python
def __init__(
    self,
    ds_config: dict[str, Any] | None = None,
    batch_size: int = 32,
) -> None:
    self.ds_config = dict(ds_config) if ds_config is not None else dict(_DEFAULT_DS_CONFIG)
    self.batch_size = batch_size
```

변경 후:
```python
def __init__(
    self,
    ds_config: dict[str, Any] | None = None,
    batch_size: int = 32,
    moe: dict | None = None,
) -> None:
    self.ds_config = dict(ds_config) if ds_config is not None else dict(_DEFAULT_DS_CONFIG)
    self.batch_size = batch_size

    if moe:
        self.ds_config["moe"] = {
            "enabled": True,
            "ep_size": moe.get("expert_parallel_size", 1),
            "moe_param_group": True,
        }
        if "num_experts" in moe:
            self.ds_config["moe"]["num_experts"] = moe["num_experts"]
```

`moe` dict가 전달되면 `ds_config`에 MoE 섹션을 주입한다. `moe_param_group: True`는 expert 파라미터를 별도 param group으로 분리하여 ZeRO가 적절하게 처리하게 한다. `ds_config`에 이미 `moe` 키가 있으면(사용자가 직접 지정) 이 코드가 덮어쓴다. 이 동작은 의도적이다 — `moe` 파라미터와 `ds_config.moe` 양쪽에 MoE 설정이 존재하면 어느 쪽이 우선인지 모호해지므로, `moe` 파라미터가 명시되면 그것이 항상 우선한다. 덮어쓰기 시 warning을 로깅하여 사용자에게 알린다.

### 5.2 Phase 3 요약

| 파일 | 변경 유형 | 줄 수 (추정) |
|------|----------|------------|
| `training/strategies/deepspeed.py` | 수정 (moe 파라미터) | +10 |
| `tests/e2e/test_deepspeed_moe.py` | 신규 | ~40 |

---

## 6. 변경하지 않는 파일과 그 이유

| 파일 | 변경 불필요 이유 |
|------|----------------|
| `settings/schema.py` | `distributed`가 `dict[str, Any]`이므로 새 키 추가에 스키마 변경 불필요. plan이 제안한 `MoEDistributedConfig` 타입 모델은 Phase 2에서 검증 필요성이 구체화되면 추가한다 — 현재는 `compat_validator.py`에서 `ep_size` 필수 검증을 수행하므로 충분하다 |
| `models/pretrained.py` | HuggingFace `AutoModelForCausalLM`이 Mixtral 등 MoE 모델을 이미 지원 |
| `models/base.py` | BaseModel 인터페이스는 MoE와 무관 (MoE는 분산 전략의 관심사) |
| `training/callbacks/` | 콜백 시스템은 전략에 의존하지 않음 |
| `data/` | 데이터 파이프라인은 MoE와 무관 |
| `serving/` | 서빙은 단일 GPU에서 실행되므로 EP 불필요 |
| `cli/` | CLI는 전략 선택을 Config에 위임하므로 변경 불필요 |
| `settings/validation/business_validator.py` | MoE는 비즈니스 로직(task-head 호환 등)과 무관 |
| `settings/validation/catalog_validator.py` | catalog의 새 필드는 검증하지 않아도 동작 |

---

## 7. 구현 순서 및 의존관계

```
Phase 0 (선행):
  rl_trainer.py kwargs 수정                     의존: 없음

Phase 1 (MoE-Aware FSDP):
  1. fsdp.py — auto_wrap_cls + HYBRID_SHARD     의존: 없음
  2. catalog YAML — mixtral                       의존: 없음
  3. Phase 1 테스트                               의존: 1

Phase 2 (Expert Parallelism):
  4. moe.py — MoEStrategy 전체 구현              의존: 없음 (독립)
  5. trainer.py — STRATEGY_MAP 등록              의존: 4
  6. aliases.yaml + __init__.py — export         의존: 4
  7. factory.py — MoE 감지                       의존: 없음
  8. compat_validator.py — MoE 검증 규칙         의존: 없음
  9. Phase 2 테스트                               의존: 4, 5, 7, 8

Phase 3 (DeepSpeed MoE):
  10. deepspeed.py — MoE config 확장             의존: 없음
  11. Phase 3 테스트                              의존: 10
```

**병렬 가능한 작업**:
- Phase 0과 Phase 1(1, 2)은 독립적으로 병렬 진행 가능
- Phase 2의 4(moe.py)와 7(factory.py), 8(compat_validator.py)은 병렬 진행 가능
- Phase 1과 Phase 2는 코드 의존이 없으므로 병렬 진행 가능 (Phase 1이 더 작으므로 먼저 완성 권장)
- Phase 3는 Phase 2와 독립

---

## 8. 예상 난이도와 리스크

| Phase | 핵심 난관 | 리스크 |
|-------|---------|--------|
| Phase 0 | 없음 (3줄 수정) | 기존 RL 테스트 영향 확인 필요 |
| Phase 1 | `_resolve_layer_class`의 다양한 모델 지원 | HuggingFace 모델의 layer class 이름이 모델마다 다름 |
| Phase 2 | `AllToAllDispatch`의 backward 구현 | gradient 역전파 정확성, 메모리 릭 방지 |
| Phase 2 | expert별 분산 checkpoint | 복원 시 expert 재할당이 다를 수 있음 |
| Phase 3 | DeepSpeed MoE API 호환성 | DeepSpeed 버전에 따라 API 차이 |

Phase 2의 `AllToAllDispatch`가 전체 구현의 ~40%를 차지하며, `torch.autograd.Function`의 forward/backward에서 모든 텐서의 device, dtype, shape이 정확히 일치해야 한다. 디버깅이 어려운 영역이므로, gloo CPU backend로 소규모 테스트를 충분히 거친 후 GPU 테스트로 넘어가야 한다.
