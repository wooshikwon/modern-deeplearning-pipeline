# MoE Expert Parallelism 구현 설계서

## 배경

2025 프론티어 오픈소스 모델이 MoE(Mixture-of-Experts) 아키텍처로 수렴하고 있다:

| 모델 | 총 파라미터 | 활성 파라미터 | Expert 수 | Top-K |
|------|-----------|------------|----------|-------|
| Mixtral 8x7B | 46.7B | 12.9B | 8 | 2 |
| Mixtral 8x22B | 141B | 39B | 8 | 2 |
| DeepSeek-V3 | 671B | 37B | 256 | 8 |
| Qwen2-MoE 57B | 57.4B | 14.3B | 64 | 8(shared) + 4(routed) |
| DBRX | 132B | 36B | 16 | 4 |
| LLaMA 4 Maverick | 400B | ~17B | 128 | — |

MDP의 현재 FSDP 전략은 모든 파라미터를 GPU 전체에 균등 sharding한다. MoE에서 이것이 왜 문제인지, 그리고 어떻게 해결해야 하는지를 다룬다.

---

## 현재 구조 분석

### FSDP의 MoE 비효율성

현재 FSDP 전략 (`training/strategies/fsdp.py`):

```python
# fsdp.py:53-96 — setup()
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=self.min_num_params,  # 기본 1M
)
model = FSDP(model, sharding_strategy=..., auto_wrap_policy=...)
```

**size_based_auto_wrap_policy의 동작**: 파라미터 수가 `min_num_params` 이상인 모듈을 개별 FSDP unit으로 래핑한다. 모듈의 의미(attention인지 expert인지)는 고려하지 않는다.

**MoE에서 발생하는 문제:**

Mixtral 8x7B 기준, 한 디코더 레이어의 구조:

```
MixtralDecoderLayer
├── self_attn (공유 — 모든 토큰이 사용)
│   ├── q_proj: [4096, 4096]    ← 16.8M params
│   ├── k_proj: [4096, 1024]
│   ├── v_proj: [4096, 1024]
│   └── o_proj: [4096, 4096]
└── block_sparse_moe
    ├── gate: [4096, 8]          ← router (32K params, 무시 가능)
    └── experts: ModuleList[8]
        ├── expert[0]: MixtralBLockSparseTop2MLP
        │   ├── w1: [4096, 14336]  ← 58.7M params
        │   ├── w2: [14336, 4096]
        │   └── w3: [4096, 14336]
        ├── expert[1]: ...
        └── expert[7]: ...
```

**size_based (현재)**: 각 expert가 58.7M > 1M이므로 개별 FSDP unit으로 래핑된다. Expert 0의 `w1` 파라미터가 4개 GPU에 분할되므로, 토큰이 Expert 0으로 라우팅될 때마다 **all-gather**로 전체를 복원해야 한다.

```
현재 (FSDP FULL_SHARD):

GPU 0: expert[0].w1[:, :3584], expert[1].w1[:, :3584], ..., expert[7].w1[:, :3584]
GPU 1: expert[0].w1[:, 3584:7168], ...
GPU 2: expert[0].w1[:, 7168:10752], ...
GPU 3: expert[0].w1[:, 10752:14336], ...

토큰 → Expert 0 라우팅 → 4 GPU all-gather → forward → scatter
토큰 → Expert 3 라우팅 → 4 GPU all-gather → forward → scatter
... 매 토큰, 매 레이어에서 반복
```

**Expert Parallelism (목표)**: 각 GPU가 2개 expert를 통째로 보유. 토큰이 라우팅되면 해당 GPU로 **point-to-point 전송**.

```
목표 (Expert Parallelism, EP=4):

GPU 0: expert[0] 전체, expert[1] 전체
GPU 1: expert[2] 전체, expert[3] 전체
GPU 2: expert[4] 전체, expert[5] 전체
GPU 3: expert[6] 전체, expert[7] 전체

토큰 → Expert 0 라우팅 → GPU 0으로 전송 → forward → 결과 반환
                        (all-to-all, 1:1 통신)
```

**통신량 비교 (Mixtral 8x7B, 4 GPU, batch=12, seq=2048):**

| 방식 | 연산 | 통신량/layer | 비고 |
|------|------|------------|------|
| FSDP FULL_SHARD | expert all-gather + scatter | ~1.4GB | 모든 expert 파라미터 복원 |
| Expert Parallelism | token all-to-all | ~25MB | 활성 토큰의 hidden state만 |

약 **56배 통신량 차이**. 레이어 32개 × 양방향이므로 실제 체감은 더 크다.

### 영향받는 파일 목록

| 파일 | 역할 | 변경 유형 |
|------|------|----------|
| `settings/schema.py` | config 스키마 | 확장 |
| `training/strategies/fsdp.py` | FSDP 전략 | 확장 |
| `training/strategies/moe.py` | MoE 전략 | **신규** |
| `training/strategies/base.py` | 전략 인터페이스 | 확장 (선택) |
| `training/trainer.py` | SFT 트레이너 | 수정 |
| `training/rl_trainer.py` | RL 트레이너 | 수정 |
| `models/pretrained.py` | 모델 로딩 | 확장 |
| `models/catalog/text_generation/` | 모델 카탈로그 | 추가 |
| `factory/factory.py` | 팩토리 | 확장 |
| `aliases.yaml` | 컴포넌트 별칭 | 추가 |

---

## MoE 분산 학습 기본 개념

### 병렬화 차원

```
       Data Parallel (DP)          Expert Parallel (EP)
       ─────────────────          ──────────────────────
       같은 모델, 다른 데이터       다른 expert, 같은 데이터
       gradient all-reduce         token all-to-all dispatch

       ┌─────┐  ┌─────┐          ┌─────┐  ┌─────┐
       │GPU 0│  │GPU 1│          │GPU 0│  │GPU 1│
       │모델A│  │모델A│          │E0,E1│  │E2,E3│
       │배치1│  │배치2│          │배치1│  │배치1│
       └─────┘  └─────┘          └─────┘  └─────┘
```

### EP + DP 조합 (실전 구성)

8 GPU, Mixtral 8x7B (8 experts):

```
EP_size=4, DP_size=2

       DP Group 0                  DP Group 1
    ┌──────────────┐            ┌──────────────┐
    │ EP Group 0   │            │ EP Group 1   │
    │              │            │              │
    │ GPU0: E0,E1  │            │ GPU4: E0,E1  │
    │ GPU1: E2,E3  │            │ GPU5: E2,E3  │
    │ GPU2: E4,E5  │            │ GPU6: E4,E5  │
    │ GPU3: E6,E7  │            │ GPU7: E6,E7  │
    └──────────────┘            └──────────────┘

Forward:
1. 공유 레이어 (attention): DP 방식 — 각 GPU가 자기 배치 처리
2. MoE 레이어: EP 방식
   a. Router가 각 토큰의 expert 결정
   b. all-to-all: 토큰을 담당 GPU로 전송
   c. 각 GPU가 자기 expert로 forward
   d. all-to-all: 결과를 원래 GPU로 반환

Backward:
1. MoE 레이어: all-to-all gradient 역전파
2. 공유 레이어: DP all-reduce gradient 동기화
```

### Process Group 구성

```python
# world_size=8, ep_size=4, dp_size=2
# GPU [0,1,2,3]이 EP Group 0, [4,5,6,7]이 EP Group 1
# GPU [0,4]가 DP Group 0, [1,5]가 DP Group 1, ...

ep_groups = [[0,1,2,3], [4,5,6,7]]     # expert 교환용
dp_groups = [[0,4], [1,5], [2,6], [3,7]]  # gradient 동기화용
```

---

## 단계적 구현 계획

3단계로 나누어 각 단계가 독립적으로 가치를 제공하도록 한다.

### Phase 1: MoE-Aware FSDP Wrapping

**목표**: MoE 모델을 기존 FSDP로 돌리되, wrapping 정책을 최적화하여 기본적인 학습이 가능하게 한다.

**핵심 변경**: size_based → transformer_auto_wrap + MoE-aware hybrid sharding.

#### 1-1. 스키마 확장 (`settings/schema.py`)

**위치**: `ComputeConfig.distributed` (dict[str, Any])

현재 distributed config는 untyped dict다. MoE 관련 필드를 추가한다:

```python
# 현재: config.compute.distributed에 자유 형태로 전달
# distributed:
#   strategy: fsdp
#   sharding_strategy: FULL_SHARD
#   mixed_precision: true

# 확장:
# distributed:
#   strategy: fsdp
#   sharding_strategy: HYBRID_SHARD      # MoE 권장
#   mixed_precision: true
#   auto_wrap_cls: MixtralDecoderLayer   # 새 필드: transformer layer class
#   moe:                                  # 새 섹션
#     enabled: true
#     expert_parallel_size: 4
```

`distributed`가 untyped dict이므로 스키마 변경 없이 새 필드를 넣을 수 있다. 다만 validation을 위해 `MoEDistributedConfig` Pydantic 모델을 추가하는 것을 권장한다:

```python
# schema.py에 추가
class MoEDistributedConfig(BaseModel):
    """MoE 분산 학습 설정."""
    enabled: bool = False
    expert_parallel_size: int = 1
    auto_wrap_cls: str | None = None  # transformer layer class name
```

**영향**: 기존 config와 완전 호환. `moe` 섹션이 없으면 현재 동작 유지.

#### 1-2. FSDP 전략에 transformer-aware wrapping 추가 (`training/strategies/fsdp.py`)

**위치**: `FSDPStrategy.__init__` (line 32) + `setup` (line 53)

현재:
```python
def __init__(self, ..., min_num_params: int = 1_000_000):
    self.min_num_params = min_num_params
```

변경:
```python
def __init__(
    self,
    sharding_strategy: str = "FULL_SHARD",
    mixed_precision: bool = True,
    backend: str = "nccl",
    cpu_offload: bool = False,
    precision: str = "bf16",
    min_num_params: int = 1_000_000,
    # ── 새 필드 ──
    auto_wrap_cls: str | None = None,    # "MixtralDecoderLayer" 등
    moe: dict | None = None,             # MoE 설정 dict
) -> None:
    # ... 기존 ...
    self.auto_wrap_cls = auto_wrap_cls
    self.moe_config = moe or {}
```

**setup() 내 auto_wrap_policy 생성 로직 변경**:

현재 (line 76 부근):
```python
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=self.min_num_params,
)
```

변경:
```python
if self.auto_wrap_cls is not None:
    # transformer layer class 기반 wrapping
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import importlib

    # "MixtralDecoderLayer" → transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer
    layer_cls = self._resolve_layer_class(self.auto_wrap_cls)
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_cls},
    )
else:
    # 기존 size_based 유지
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=self.min_num_params,
    )
```

**`_resolve_layer_class` 헬퍼**:

```python
@staticmethod
def _resolve_layer_class(cls_name: str) -> type:
    """클래스 이름 또는 전체 경로에서 클래스를 resolve한다.

    - 전체 경로: "transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer"
    - 단축명: "MixtralDecoderLayer" → transformers 패키지에서 자동 탐색
    """
    if "." in cls_name:
        module_path, _, class_name = cls_name.rpartition(".")
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    # 단축명: transformers에서 탐색
    import transformers
    cls = getattr(transformers, cls_name, None)
    if cls is None:
        raise ValueError(
            f"transformer layer class '{cls_name}'를 찾을 수 없습니다. "
            "전체 경로를 사용하세요."
        )
    return cls
```

**MoE 감지 시 sharding 자동 조정**:

```python
def setup(self, model, device, optimizer=None):
    # ... 기존 process group 초기화 ...

    # MoE 모델에서 FULL_SHARD 사용 시 HYBRID_SHARD로 자동 전환
    actual_strategy = self._sharding_strategy
    if self.moe_config.get("enabled", False):
        if actual_strategy == ShardingStrategy.FULL_SHARD:
            logger.info(
                "MoE 모델 감지: FULL_SHARD → HYBRID_SHARD 자동 전환 "
                "(노드 내 sharding, 노드 간 replication으로 expert 통신 최소화)"
            )
            actual_strategy = ShardingStrategy.HYBRID_SHARD

    model = FSDP(
        model,
        sharding_strategy=actual_strategy,
        auto_wrap_policy=auto_wrap_policy,
        # ...
    )
```

**이점**: MoE 모델에서 HYBRID_SHARD를 쓰면, 노드 내에서만 파라미터를 sharding하고 노드 간에는 replicate한다. Expert all-gather가 노드 내 NVLink으로 제한되어 inter-node bandwidth 병목을 피한다.

#### 1-3. 모델 카탈로그 추가

**신규 파일**: `models/catalog/text_generation/mixtral-8x7b.yaml`

```yaml
name: mixtral-8x7b
family: mixtral
class_path: transformers.AutoModelForCausalLM
head_builtin: true
architecture: moe

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
    # 공유 attention만 (expert FFN 제외 — 메모리 절약 + expert 다양성 보존)
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
    r: 16
    alpha: 32
    task_type: "CAUSAL_LM"
  lora_with_experts:
    # expert FFN 포함 (전체 fine-tuning에 근접한 효과)
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"]
    r: 8  # expert 수 × rank이므로 작은 rank으로도 충분
    alpha: 16
    task_type: "CAUSAL_LM"

memory:
  params_m: 46700    # 총 파라미터
  active_params_m: 12900  # 활성 파라미터 (top-2 of 8)
  fp16_gb: 93.4
  qlora_4bit_gb: 26.0
```

**신규 파일**: `models/catalog/text_generation/deepseek-v3.yaml`

```yaml
name: deepseek-v3
family: deepseek
class_path: transformers.AutoModelForCausalLM
head_builtin: true
architecture: moe

pretrained_sources:
  - "hf://deepseek-ai/DeepSeek-V3"

supported_tasks:
  - text_generation

default_head:
  task: text_generation
  _component_: CausalLMHead
  hidden_dim: 7168
  vocab_size: 129280

moe:
  num_experts: 256
  top_k: 8
  decoder_layer_cls: DeepseekV3DecoderLayer
  expert_module_pattern: "mlp.experts"
  router_module_pattern: "mlp.gate"
  # MLA (Multi-head Latent Attention) 사용
  uses_mla: true

adapter_defaults:
  lora:
    # MLA 구조: kv_a_proj_with_mqa, kv_b_proj이 K/V 역할
    target_modules: ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
    r: 16
    alpha: 32
    task_type: "CAUSAL_LM"

memory:
  params_m: 671000
  active_params_m: 37000
  fp16_gb: 1342.0  # 멀티노드 필수
  qlora_4bit_gb: 168.0
```

#### 1-4. Config 사용 예시

```yaml
# config-mixtral.yaml
environment:
  name: local
compute:
  target: local
  gpus: 4
  distributed:
    strategy: fsdp
    sharding_strategy: HYBRID_SHARD
    mixed_precision: true
    precision: bf16
    auto_wrap_cls: MixtralDecoderLayer     # transformer-aware wrapping
    moe:
      enabled: true
```

**Phase 1 결과**: Mixtral 등 MoE 모델이 FSDP로 학습 가능해진다. 최적은 아니지만 동작은 한다. size_based의 비효율적 wrapping을 해결하고, HYBRID_SHARD로 inter-node 통신을 줄인다.

---

### Phase 2: Expert Parallelism Strategy

**목표**: 전용 MoE 전략을 구현하여 expert를 GPU에 분배하고, token all-to-all dispatch로 통신을 최소화한다.

#### 2-1. 신규 전략 파일 (`training/strategies/moe.py`)

**핵심 설계**:

```python
"""MoE Expert Parallelism + Data Parallelism 전략.

Expert를 GPU에 분배하고, 공유 레이어는 FSDP로 sharding한다.
MoE 레이어의 forward에 all-to-all token dispatch를 삽입한다.

Process Group 구성 (world_size=8, ep_size=4):
  EP groups: [[0,1,2,3], [4,5,6,7]]  — expert 간 token 교환
  DP groups: [[0,4], [1,5], [2,6], [3,7]]  — gradient 동기화
"""

class MoEStrategy(BaseStrategy):
    def __init__(
        self,
        ep_size: int,                        # Expert Parallel degree
        precision: str = "bf16",
        backend: str = "nccl",
        decoder_layer_cls: str | None = None,
        expert_module_pattern: str = "experts",
    ):
        self.ep_size = ep_size
        self.precision = precision
        self.backend = backend
        self.decoder_layer_cls = decoder_layer_cls
        self.expert_module_pattern = expert_module_pattern

        self._ep_group = None   # Expert parallel process group
        self._dp_group = None   # Data parallel process group
```

**setup() 핵심 로직**:

```python
def setup(self, model, device, optimizer=None):
    # 1. Process group 초기화
    dist.init_process_group(backend=self.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dp_size = world_size // self.ep_size

    # 2. EP / DP process group 생성
    #    EP group: 같은 EP group 내 GPU들이 expert token을 교환
    #    DP group: 같은 위치의 GPU들이 gradient를 동기화
    for ep_start in range(0, world_size, self.ep_size):
        ep_ranks = list(range(ep_start, ep_start + self.ep_size))
        group = dist.new_group(ep_ranks)
        if rank in ep_ranks:
            self._ep_group = group
            self._ep_rank = ep_ranks.index(rank)

    for dp_offset in range(self.ep_size):
        dp_ranks = list(range(dp_offset, world_size, self.ep_size))
        group = dist.new_group(dp_ranks)
        if rank in dp_ranks:
            self._dp_group = group

    # 3. Expert 분배: 각 GPU에 할당된 expert 인덱스 계산
    #    예: 8 experts, EP=4 → GPU 0: [0,1], GPU 1: [2,3], ...
    model_config = model.config if hasattr(model, "config") else None
    num_experts = self._detect_num_experts(model, model_config)
    experts_per_gpu = num_experts // self.ep_size
    my_expert_start = self._ep_rank * experts_per_gpu
    my_expert_end = my_expert_start + experts_per_gpu
    self._expert_range = (my_expert_start, my_expert_end)

    # 4. MoE forward hook 삽입
    #    expert module의 forward를 all-to-all dispatch로 래핑
    self._install_ep_hooks(model)

    # 5. 공유 레이어에 FSDP 적용 (expert 제외)
    model = self._wrap_shared_with_fsdp(model, device)

    # 6. Expert 파라미터를 해당 GPU로 이동
    #    각 GPU는 자기 expert만 GPU 메모리에 보유
    self._distribute_experts(model, device)

    return model.to(device)
```

**all-to-all token dispatch 핵심**:

```python
def _install_ep_hooks(self, model):
    """MoE 레이어에 all-to-all dispatch hook을 삽입한다.

    원래 MoE forward:
      1. router(hidden) → expert_indices  (어떤 expert에 보낼지)
      2. for each expert: expert(tokens_for_this_expert)
      3. combine results

    EP forward:
      1. router(hidden) → expert_indices
      2. all-to-all: 토큰을 담당 GPU로 전송
      3. 각 GPU가 자기 expert만 forward
      4. all-to-all: 결과를 원래 GPU로 반환
    """
    for name, module in model.named_modules():
        if self._is_moe_layer(name, module):
            original_forward = module.forward
            module.forward = self._make_ep_forward(module, original_forward)
```

**`_make_ep_forward` 구현 (핵심 알고리즘)**:

```python
def _make_ep_forward(self, moe_module, original_forward):
    """MoE layer의 forward를 EP-aware로 래핑한다."""
    ep_group = self._ep_group
    ep_size = self.ep_size
    ep_rank = self._ep_rank

    def ep_forward(hidden_states, **kwargs):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 1단계: 라우터로 expert 할당 결정
        router_logits = moe_module.gate(hidden_states.view(-1, hidden_dim))
        routing_weights, selected_experts = torch.topk(
            router_logits.softmax(dim=-1), k=moe_module.top_k, dim=-1
        )

        # 2단계: 토큰을 담당 GPU로 분류
        #   각 토큰이 어느 EP rank의 expert에 해당하는지 계산
        experts_per_gpu = moe_module.num_experts // ep_size
        target_ep_rank = selected_experts // experts_per_gpu  # [tokens, top_k]

        # 3단계: all-to-all dispatch
        #   각 GPU가 보낼 토큰 수를 교환 → 실제 토큰 교환
        send_counts = torch.zeros(ep_size, dtype=torch.long, device=hidden_states.device)
        for r in range(ep_size):
            send_counts[r] = (target_ep_rank == r).sum()

        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=ep_group)

        # 토큰 재정렬: 목적지 GPU별로 정렬
        dispatched = _dispatch_tokens(hidden_states, target_ep_rank, ep_size)

        # all-to-all: 실제 hidden_states 교환
        received = _all_to_all_tokens(dispatched, send_counts, recv_counts, ep_group)

        # 4단계: 로컬 expert forward
        #   이 GPU가 담당하는 expert만 실행
        local_expert_start = ep_rank * experts_per_gpu
        local_output = _compute_local_experts(
            received, moe_module.experts,
            local_expert_start, experts_per_gpu,
        )

        # 5단계: all-to-all 역방향 — 결과를 원래 GPU로 반환
        result = _all_to_all_tokens(local_output, recv_counts, send_counts, ep_group)

        # 6단계: 원래 순서로 복원 + routing weight 적용
        output = _undispatch_tokens(result, target_ep_rank, routing_weights)

        return output.view(batch_size, seq_len, hidden_dim)

    return ep_forward
```

#### 2-2. 전략 등록 (`aliases.yaml`, `trainer.py`)

`aliases.yaml`:
```yaml
strategy:
  # ... 기존 ...
  MoEStrategy: mdp.training.strategies.moe.MoEStrategy
```

`trainer.py` STRATEGY_MAP:
```python
STRATEGY_MAP: dict[str, str] = {
    "ddp": "...",
    "fsdp": "...",
    "deepspeed": "...",
    "deepspeed_zero3": "...",
    "moe": "mdp.training.strategies.moe.MoEStrategy",  # 추가
}
```

#### 2-3. Config 사용 예시

```yaml
# config-mixtral-ep.yaml
compute:
  gpus: 8
  distributed:
    strategy: moe
    ep_size: 4                    # 4 GPU per expert group
    precision: bf16
    decoder_layer_cls: MixtralDecoderLayer
    expert_module_pattern: "block_sparse_moe.experts"
```

**Phase 2 결과**: 전용 EP 전략으로 MoE 모델의 통신 효율이 극적으로 개선된다. Mixtral 기준 Phase 1 대비 ~5-10배 throughput 향상 기대.

---

### Phase 3: DeepSpeed MoE 통합

**목표**: DeepSpeed의 검증된 MoE 지원을 활용하여, MDP에서 `deepspeed.moe`를 사용할 수 있게 한다.

#### 3-1. DeepSpeed MoE 장점

DeepSpeed는 MoE를 위한 검증된 인프라를 제공한다:

- `deepspeed.moe.layer.MoE`: drop-in replacement MoE layer
- Expert parallelism + ZeRO-3 hybrid 자동 구성
- Top-K routing, capacity factor, load balancing loss 내장
- Token dropping / no-dropping 모드
- Expert 수 > GPU 수 시 자동 expert slicing

#### 3-2. DeepSpeed 전략 확장 (`training/strategies/deepspeed.py`)

현재 `_DEFAULT_DS_CONFIG`에 MoE 설정을 추가:

```python
def __init__(
    self,
    ds_config: dict | None = None,
    batch_size: int = 32,
    # ── 새 필드 ──
    moe: dict | None = None,
) -> None:
    self.ds_config = dict(ds_config or _DEFAULT_DS_CONFIG)
    self._batch_size = batch_size

    if moe:
        self.ds_config["moe"] = {
            "enabled": True,
            "ep_size": moe.get("expert_parallel_size", 1),
            "num_experts": moe.get("num_experts"),
            "moe_param_group": True,  # expert params를 별도 param group으로
        }
```

#### 3-3. Config 사용 예시

```yaml
# config-deepseek-v3.yaml
compute:
  gpus: 8
  distributed:
    strategy: deepspeed
    ds_config:
      zero_optimization:
        stage: 3
      bf16:
        enabled: true
      moe:
        enabled: true
        ep_size: 4
```

**Phase 3 결과**: DeepSeek-V3 같은 초대형 MoE 모델을 ZeRO-3 + EP 조합으로 학습할 수 있다. 검증된 DeepSpeed 인프라 위에 구축되므로 안정성이 높다.

---

## 파일별 변경 상세

### `settings/schema.py`

**변경 범위**: 5줄 추가

```python
# line 170 부근, ComputeConfig 클래스 내
class ComputeConfig(BaseModel):
    # ... 기존 필드 ...
    distributed: dict[str, Any] | None = None
```

`distributed`가 untyped dict이므로 스키마 변경 없이 `moe`, `auto_wrap_cls`, `ep_size` 등을 넣을 수 있다. Phase 1에서는 스키마를 건드리지 않는다.

검증이 필요해지면(Phase 2+) `MoEDistributedConfig`를 추가:

```python
class MoEDistributedConfig(BaseModel):
    """MoE 분산 학습 설정. distributed.moe 섹션에 대응."""
    enabled: bool = False
    expert_parallel_size: int = 1
    auto_wrap_cls: str | None = None
    expert_module_pattern: str = "experts"
```

### `training/strategies/fsdp.py`

**변경 범위**: ~40줄 수정/추가

1. `__init__`에 `auto_wrap_cls`, `moe` 파라미터 추가 (3줄)
2. `setup()`에 transformer_auto_wrap_policy 분기 추가 (15줄)
3. MoE 감지 시 HYBRID_SHARD 자동 전환 (5줄)
4. `_resolve_layer_class` 헬퍼 추가 (15줄)

### `training/strategies/moe.py` (신규)

**예상 규모**: ~300줄

1. `MoEStrategy` 클래스 (BaseStrategy 상속)
2. Process group 생성 (EP group, DP group)
3. Expert 분배 로직
4. all-to-all token dispatch hook
5. 공유 레이어 FSDP wrapping
6. Checkpoint save/load (expert별 분산 저장)

### `training/trainer.py`

**변경 범위**: 2줄

```python
STRATEGY_MAP["moe"] = "mdp.training.strategies.moe.MoEStrategy"
```

### `models/pretrained.py`

**변경 범위**: 0줄 (Phase 1-2)

HuggingFace `AutoModelForCausalLM`이 Mixtral, DeepSeek-V2 등을 이미 지원한다. 모델 로딩 자체는 변경 불필요. 다만 DeepSeek-V3처럼 HF에 아직 완전 통합되지 않은 모델은 `class_path`로 커스텀 클래스를 지정하여 로딩한다:

```yaml
model:
  class_path: transformers.AutoModelForCausalLM
  pretrained: hf://deepseek-ai/DeepSeek-V3
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2
```

### `factory/factory.py`

**변경 범위**: ~15줄 (Phase 2)

MoE 모델 감지 및 카탈로그에서 MoE 설정 추출:

```python
def _build_model(self) -> nn.Module:
    # ... 기존 로직 ...
    model = self._load_pretrained(model_spec)

    # MoE 감지: 카탈로그 또는 모델 구조에서
    if self._is_moe_model(model):
        moe_info = self._extract_moe_info(model)
        self._cache["moe_info"] = moe_info
        logger.info(
            "MoE 모델 감지: %d experts, top-%d",
            moe_info["num_experts"], moe_info["top_k"],
        )

    # ... head, adapter 적용 ...

@staticmethod
def _is_moe_model(model: nn.Module) -> bool:
    """모델이 MoE 아키텍처인지 감지한다."""
    config = getattr(model, "config", None)
    if config is None:
        return False
    # HuggingFace MoE 모델은 config에 num_local_experts 또는 num_experts를 가진다
    return hasattr(config, "num_local_experts") or hasattr(config, "num_experts")
```

### `aliases.yaml`

**변경 범위**: 1줄

```yaml
strategy:
  # ... 기존 ...
  MoEStrategy: mdp.training.strategies.moe.MoEStrategy
```

---

## 테스트 전략

### Phase 1 테스트

```python
# tests/e2e/test_fsdp_moe_wrap.py

def test_fsdp_transformer_auto_wrap():
    """auto_wrap_cls 지정 시 transformer_auto_wrap_policy가 적용되는지."""
    strategy = FSDPStrategy(auto_wrap_cls="torch.nn.TransformerEncoderLayer")
    # 작은 TransformerEncoder로 검증
    model = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=16, nhead=4), num_layers=2)
    # setup 후 각 layer가 개별 FSDP unit으로 래핑되었는지 확인

def test_fsdp_hybrid_shard_auto_switch():
    """MoE 설정 시 FULL_SHARD → HYBRID_SHARD 자동 전환."""
    strategy = FSDPStrategy(
        sharding_strategy="FULL_SHARD",
        moe={"enabled": True},
    )
    # 내부 actual_strategy가 HYBRID_SHARD인지 확인
```

### Phase 2 테스트

```python
# tests/e2e/test_moe_strategy.py

def test_moe_process_group_creation():
    """EP/DP process group이 올바르게 생성되는지 (gloo CPU 2프로세스)."""

def test_moe_expert_distribution():
    """Expert가 GPU별로 올바르게 분배되는지."""

def test_moe_all_to_all_dispatch():
    """토큰이 올바른 GPU로 dispatch되고 결과가 복원되는지."""
```

분산 테스트는 `torch.multiprocessing.spawn` + gloo CPU backend로 실행.

---

## 구현 순서 및 의존관계

```
Phase 1 (MoE-Aware FSDP):
  1. fsdp.py — auto_wrap_cls + HYBRID_SHARD         의존: 없음
  2. catalog YAML — mixtral, deepseek 카탈로그         의존: 없음
  3. 테스트 — transformer_auto_wrap 검증               의존: 1

Phase 2 (Expert Parallelism):
  4. moe.py — MoEStrategy 전체 구현                   의존: 없음 (독립)
  5. trainer.py — STRATEGY_MAP 등록                    의존: 4
  6. factory.py — MoE 감지                            의존: 없음
  7. 테스트 — EP process group + all-to-all            의존: 4

Phase 3 (DeepSpeed MoE):
  8. deepspeed.py — MoE config 확장                    의존: 없음
  9. 테스트 — DeepSpeed MoE 통합                       의존: 8
```

Phase 1과 Phase 2는 병렬 진행 가능. Phase 1이 더 작고 즉시 가치를 제공하므로 먼저 완성한다.

---

## 예상 작업량

| Phase | 파일 수 | 코드 (줄) | 테스트 (줄) | 난이도 |
|-------|--------|----------|------------|--------|
| Phase 1 | 3 수정 + 2 카탈로그 | ~60 | ~40 | 하 |
| Phase 2 | 1 신규 + 3 수정 | ~300 | ~100 | **상** |
| Phase 3 | 1 수정 | ~30 | ~40 | 중 |
| **합계** | | ~390 | ~180 | |

Phase 2의 all-to-all dispatch가 핵심 난관이다. 특히 backward pass에서의 gradient 역전파가 올바르게 동작하려면 `torch.autograd.Function`으로 커스텀 backward를 구현해야 한다. 이 부분이 전체 구현의 ~40%를 차지한다.
