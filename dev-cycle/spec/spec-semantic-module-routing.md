# spec-semantic-module-routing

**Status**: draft (2026-04-13)
**Scope**: mdp Recipe의 "모델 내부 이름 의존" 제거. 사용자는 의미적(dot-path) 이름만 쓰고, mdp가 family별 실제 이름으로 번역한다.

---

## 0. 배경 (Why)

### 0.1. 현재 상태의 문제

mdp Recipe가 **프레임워크 내부 속성명**을 사용자에게 노출한다. 예:

```yaml
adapter:
  target_modules: ["q_proj", "v_proj"]   # Llama 전용
```

같은 "attention Q/V에 LoRA 붙이기"를 표현할 때:
- Llama: `[q_proj, v_proj]`
- BERT: `[query, value]`
- T5: `[q, v]` (접미사 없음)
- GPT-2: `[c_attn]` (fused QKV)
- ViT-timm: `[qkv]` (fused)
- CLIP: `[q_proj, v_proj]` (출력은 `out_proj`, NOT `o_proj`)

**현재까지 발견된 실제 피해**:
1. catalog 5개(`dinov2-base`, `segformer-b0`, `efficientnet-b0`, `convnext-base`, `mixtral-8x7b`)가 모델 실제 모듈명과 불일치 → LoRA 적용 시 "target modules not found" 실패. (commit `262100e`에서 개별 수정)
2. `target_modules="all_linear"` default가 PEFT의 `"all-linear"` 표준과 충돌. (commit `f9cb76a`에서 수정)
3. transformers 5.x에서 Mixtral expert MLP가 `nn.Linear(w1/w2/w3)` → `nn.Parameter(gate_up_proj)`로 변경 — 카탈로그가 조용히 깨짐.

이 세 사건은 모두 **"사용자/카탈로그가 프레임워크 내부 관례를 외워서 쓰는"** 설계의 동일 증상이다.

### 0.2. LoRA만의 문제가 아니다

Recipe에서 "모델 내부 이름에 의존"하는 영역 전수:

| 필드 | 의존하는 내부 개념 | 예시 variation |
|------|-------------------|---------------|
| `adapter.target_modules` | Linear 속성명 | q_proj / query / q / c_attn / qkv / qkv_proj |
| `head._target_attr` | head 속성명 | classifier / head / lm_head / fc |
| `adapter.modules_to_save` | 저장 대상 이름 | lm_head / embed_tokens / wte / wpe / ... |
| `data.augmentation` mean/std | 사전학습 통계 | ImageNet / CLIP / SigLIP / CIFAR (각기 다름) |
| `generation.*` | 모델 추천 config | 모델별 temperature/top_p 권장값 |
| PEFT task_type | 태스크 enum | CAUSAL_LM / SEQ_2_SEQ_LM / SEQ_CLS / ... |

같은 구조적 문제가 여러 필드에 퍼져 있다.

### 0.3. 이 spec의 목표

**모든 "모델 내부 이름 의존"을 의미적 레이어로 추상화**한다:

1. 사용자 Recipe는 **family에 독립적**인 dot-path 의미 이름만 사용
2. mdp가 **단일 진실의 원천**(`family_routing`)에서 family → actual 번역
3. **원자 단위 커스텀** 가능 (`attn.q`, `mlp.gate` 등 개별 지정)
4. **Escape hatch** 유지 — 기존 raw 이름 리스트도 그대로 작동 (breaking change 없음)
5. **자동 검증** — 번역 테이블이 실제 HF/timm 모델 구조와 일치하는지 CI에서 매 빌드 검증

---

## 1. 핵심 설계 결정

### 1.1. Factory proxy layer 필요성 검토

**질문**: semantic → actual 번역을 어디에 둘 것인가?

**후보**:

| 후보 | 위치 | 장 | 단 |
|------|------|---|---|
| A. apply_lora 내부 | adapter 함수마다 개별 | 최소 변경 | 중복 (여러 adapter에서 반복) |
| B. Factory가 미리 번역 | `_assemble_model` 단계 | 단일 진입점 | Factory의 책임 과밀 (SRP 위반) |
| C. ComponentResolver 확장 | 범용 resolver | 전역 훅 | resolver의 범용성 훼손. adapter 특화 로직 주입 어색 |
| D. 독립 모듈 (신규) | `mdp/models/family_routing.py` | 단일 책임. 여러 소비자 재사용 | 파일 하나 추가 |

**결정: D**. 이유:

1. semantic resolution은 adapter target 외에도 **head slot, modules_to_save, augmentation auto, generation auto** 등 **여러 소비자**에서 필요하다. 각 소비자가 직접 구현하면 같은 매핑 로직이 중복되어 drift 위험.
2. Factory는 이미 조립자 역할(`_load_pretrained` / `_attach_head` / adapter 적용 5단계)을 하고 있다. 여기에 번역 책임까지 얹으면 Factory가 비대해진다. 반면 Factory는 **소비자로서** family_routing을 호출하면 되므로, **Factory 자체는 변경 최소**.
3. ComponentResolver는 "임의 dict → Python 객체" 범용 팩토리. family-specific 지식을 여기에 넣으면 범용성 훼손.
4. 독립 모듈은 stateless 함수 집합이라 **테스트가 가장 쉽다**. 드리프트 검증 테스트도 이 모듈만 대상으로 작성하면 됨.

**결론: 새 proxy layer(Factory)는 만들지 않는다.** 기존 Factory 구조 유지, 신규 모듈 `mdp/models/family_routing.py`가 옆에서 서비스한다.

### 1.2. 아키텍처 도식

```
Recipe YAML (semantic names)
    │
    ▼
SettingsFactory.for_training (Pydantic 검증)
    │
    ▼
Factory._assemble_model ──(호출)──> family_routing.resolve_*
    │                                   │
    ├─ _load_pretrained                 │
    ├─ _attach_head (slot 해석 시)      │
    │                                   │
    └─ adapter.resolve(model)           │
           ↓                            │
        apply_lora / apply_qlora ───────┘
           ↓
        family_routing.resolve_targets(target, model)
           ↓
        LoraConfig(target_modules=[실제 이름])
```

`family_routing`은 **pure function 모듈** — 내부 상태 없음. 테스트 극단적으로 단순.

### 1.3. Semantic 네임스페이스 규칙

Dot-path 형식으로 표현한다.

**Attention**:
- `attn.q` / `attn.k` / `attn.v` / `attn.o` — 분리된 Q/K/V/O
- `attn.qkv` — fused QKV (GPT-2, Phi-3, ViT-timm)

**MLP**:
- `mlp.gate` / `mlp.up` / `mlp.down` — SwiGLU
- `mlp.fc1` / `mlp.fc2` — 2-layer MLP (BERT, ViT-timm, CLIP)
- `mlp.gate_up` / `mlp.down` — fused gate+up (Phi-3)

**Head**:
- `head.cls` — classification head
- `head.lm` — language modeling head
- `head.det` — detection head
- `head.seg` — segmentation head

**Embedding**:
- `embed.token` / `embed.pos`

**Wildcard 축약**:
- `attn.*` → family에 존재하는 모든 attention 컴포넌트
- `mlp.*` → 모든 MLP 컴포넌트
- `*` → PEFT `"all-linear"` 패스스루 (모든 Linear)

**Escape hatch (raw)**:
- 리스트의 각 원소에 dot(`.`)이 없으면 raw 모듈명으로 취급 — 번역 생략, PEFT에 그대로 전달
- 예: `target: [q_proj, gate_proj]` 는 기존과 동일하게 작동

### 1.4. 사용 예시 (사용자 Recipe)

```yaml
# Llama 학습 — semantic
adapter:
  _component_: LoRA
  r: 16
  target: [attn.q, attn.v]        # → [q_proj, v_proj]로 자동 번역

# BERT 학습 — 동일한 표기
adapter:
  _component_: LoRA
  r: 16
  target: [attn.q, attn.v]        # → [query, value]로 자동 번역

# T5 학습 — 동일
adapter:
  _component_: LoRA
  r: 16
  target: [attn.q, attn.v]        # → [q, v]로 자동 번역

# 전체 attention + MLP
adapter:
  target: [attn.*, mlp.*]

# 세밀 제어 — gate만
adapter:
  target: [mlp.gate]

# 가장 쉬움 — PEFT 자동 family 매핑 (target 생략)
adapter:
  _component_: LoRA
  r: 16
  # target 없음 → PEFT의 TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING 위임

# Escape hatch — 실제 이름 직접 (고급 사용자)
adapter:
  target_modules: [q_proj, gate_proj]    # 기존 필드명 그대로
```

### 1.5. 왜 `target` (신규)과 `target_modules` (기존)을 둘 다 받는가

- **신규**: `target` 키는 semantic name 전용 (dot-path + wildcard)
- **기존**: `target_modules`는 raw name 전용 (하위 호환)
- 두 필드 **동시 지정 금지** — ValueError로 차단
- 둘 다 없으면 PEFT 자동 매핑 (`target_modules=None`)

이유: semantic/raw 경로를 한 필드에 섞으면 파싱 규칙이 애매해진다. dot 유무로 분기하는 것은 escape hatch 수준의 편의로만 제한.

---

## 2. 소비자 변경 명세

### 2.1. `mdp/models/family_routing.py` (신규)

```python
"""Family별 semantic name → actual module name 매핑.

단일 진실의 원천. 새 모델 family 추가 시 여기만 업데이트한다.
"""

# alias: 특정 family가 다른 family와 동일한 네이밍을 쓸 때 문자열로 참조
_FAMILY_ROUTING: dict[str, dict[str, str] | str] = {
    "llama": {
        "attn.q": "q_proj", "attn.k": "k_proj",
        "attn.v": "v_proj", "attn.o": "o_proj",
        "mlp.gate": "gate_proj", "mlp.up": "up_proj", "mlp.down": "down_proj",
        "head.lm": "lm_head",
        "embed.token": "embed_tokens",
    },
    "mistral": "llama",
    "qwen2": "llama",
    "gemma2": "llama",

    "phi3": {
        "attn.qkv": "qkv_proj", "attn.o": "o_proj",
        "mlp.gate_up": "gate_up_proj", "mlp.down": "down_proj",
        "head.lm": "lm_head",
    },

    "bert": {
        "attn.q": "query", "attn.k": "key", "attn.v": "value",
        "attn.o": "output.dense",
        "mlp.fc1": "intermediate.dense", "mlp.fc2": "output.dense",
        "head.cls": "classifier",
        "embed.token": "embeddings.word_embeddings",
        "embed.pos": "embeddings.position_embeddings",
    },
    "roberta": "bert",
    "dinov2": "bert",
    "segformer": "bert",

    "t5": {
        "attn.q": "q", "attn.k": "k", "attn.v": "v", "attn.o": "o",
        "mlp.gate": "wi_0", "mlp.up": "wi_1", "mlp.down": "wo",
        "head.lm": "lm_head",
    },

    "gpt2": {
        "attn.qkv": "c_attn", "attn.o": "c_proj",
        "mlp.fc1": "c_fc", "mlp.fc2": "c_proj",
        "head.lm": "lm_head",
        "embed.token": "wte", "embed.pos": "wpe",
    },

    "clip": {
        "attn.q": "q_proj", "attn.k": "k_proj", "attn.v": "v_proj",
        "attn.o": "out_proj",
        "mlp.fc1": "fc1", "mlp.fc2": "fc2",
    },
    "siglip": "clip",
    "detr": "clip",
    "florence2": "clip",
    "blip2": "clip",

    "vit": {
        "attn.qkv": "qkv", "attn.o": "proj",
        "mlp.fc1": "fc1", "mlp.fc2": "fc2",
        "head.cls": "head",
    },
    "swin": "vit",

    "convnext": {
        "conv.dw": "conv_dw",
        "mlp.fc1": "fc1", "mlp.fc2": "fc2",
        "head.cls": "head",
    },
    "efficientnet": {
        "head.cls": "classifier",
    },
    "resnet": {
        "head.cls": "fc",
    },

    "mixtral": {
        "attn.q": "q_proj", "attn.k": "k_proj",
        "attn.v": "v_proj", "attn.o": "o_proj",
        # MoE expert MLP는 transformers 5.x에서 nn.Parameter라 PEFT LoRA 불가.
        # 의도적으로 mlp.* 매핑 없음.
        "head.lm": "lm_head",
    },
}


def detect_family(model: nn.Module) -> str:
    """모델 인스턴스 → family 문자열.

    우선순위:
    1. HF: model.config.model_type (가장 권위 있음)
    2. timm: model.default_cfg["architecture"]에서 프리픽스로 family 추출
    3. torchvision: type(model).__name__으로 폴백
    """
    ...


def resolve_targets(
    targets: list[str] | str | None,
    model: nn.Module,
) -> list[str] | None:
    """Semantic target 리스트를 actual module name 리스트로 번역.

    None 입력 → None 반환 (호출부가 PEFT 자동 매핑에 위임).
    '*' 또는 'all-linear' → 'all-linear' 패스스루.
    리스트의 각 원소:
    - dot(.) 포함 + 알려진 semantic: 번역
    - dot(.) 포함 + .* 접미 (wildcard): family의 해당 prefix 하위 모든 키 확장
    - dot(.) 없음: raw name으로 취급하여 그대로 유지
    - dot 포함하지만 unknown semantic: ValueError (오타 방지)
    """
    ...


def resolve_head_slot(slot: str | None, model: nn.Module) -> str | None:
    """head.cls / head.lm / head.det / head.seg → 실제 attribute name."""
    ...


def resolve_save_modules(
    saves: list[str] | None,
    model: nn.Module,
) -> list[str] | None:
    """modules_to_save의 semantic 이름을 actual name으로 번역."""
    ...
```

**테스트 대상 인터페이스**: `detect_family`, `resolve_targets`, `resolve_head_slot`, `resolve_save_modules`.

### 2.2. `mdp/models/adapters/lora.py` 변경

```python
def apply_lora(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int | None = None,
    lora_dropout: float | None = None,
    target: list[str] | str | None = None,          # 신규: semantic
    target_modules: list[str] | str | None = None,  # 기존: raw (하위 호환)
    task_type: str | None = None,
    alpha: int | None = None,
    dropout: float | None = None,
    **kwargs: Any,
) -> nn.Module:
    # target과 target_modules 동시 지정 방지
    if target is not None and target_modules is not None:
        raise ValueError("target과 target_modules를 동시에 지정할 수 없습니다")

    # semantic 번역
    if target is not None:
        from mdp.models.family_routing import resolve_targets
        target_modules = resolve_targets(target, model)
    # 기존 behavior 유지:
    # - target_modules가 명시되면 그대로 PEFT에 전달
    # - 둘 다 None이면 target_modules=None → PEFT 자동 매핑

    ...  # LoraConfig 생성은 기존 그대로
```

`apply_qlora`도 동일 패턴. 단 QLoRA는 model_name_or_path 기반 로딩이라 model 인스턴스가 함수 초반에 없음 — `get_peft_model` 호출 직전에 `resolve_targets`를 수행한다.

### 2.3. `mdp/factory/factory.py` 변경 (최소)

`_attach_head`의 `target_attr` 파라미터가 semantic `slot` 입력도 받도록 확장:

```python
@staticmethod
def _attach_head(
    model: nn.Module,
    head: nn.Module,
    target_attr: str | None,  # raw 이름 (기존 방식)
    slot: str | None = None,  # semantic 이름 (신규)
) -> None:
    if slot is not None and target_attr is not None:
        raise ValueError("slot과 _target_attr를 동시에 지정할 수 없습니다")
    if slot is not None:
        from mdp.models.family_routing import resolve_head_slot
        target_attr = resolve_head_slot(slot, model)
    # 이후 기존 로직 그대로
    ...
```

호출부 `_assemble_model`의 head 단계:

```python
if head_config is not None:
    hc = dict(head_config)
    target_attr = hc.pop("_target_attr", None)
    slot = hc.pop("slot", None)
    head = self.resolver.resolve(hc)
    self._attach_head(model, head, target_attr, slot=slot)
```

### 2.4. `mdp/settings/schema.py` — 스키마 영향 없음

`adapter`, `head`가 `dict[str, Any]`이므로 `target` / `slot` 같은 새 키를 Pydantic 수준에서 받아들이는 데 변경 불필요. 검증은 런타임(apply_lora, _attach_head)에서 수행.

### 2.5. AGENT.md / docs 변경

- **AGENT.md** Adapter 섹션에 `target` 필드 문서화 (dot-path 설명 + 4가지 경우)
- **AGENT.md** Head 섹션에 `slot` 필드 문서화
- **docs/extending.md** 커스텀 어댑터 섹션에 semantic 규약 추가

docs/ 갱신은 doc-refine 단계에서 일괄 수행 (code 단계에서는 AGENT.md만 동기화).

### 2.6. Catalog 변경

현재 `target_modules` 하드코딩을 `target` semantic으로 치환. 예:

```yaml
# Before (llama3-8b.yaml)
adapter_defaults:
  lora:
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# After
adapter_defaults:
  lora:
    target: [attn.q, attn.k, attn.v, attn.o]
```

**원칙**: catalog의 `target`은 **권장 subset**을 의미적으로 표현. 사용자는 catalog를 복사하거나, Recipe에서 생략하여 PEFT 자동 매핑을 쓸 수 있다.

---

## 3. Unit 분할

총 4개 Unit. 의존 관계: U1 → U2, U3 (병렬) → U4.

| Unit | 범위 | 핵심 파일 | 의존 Unit | verify 기준 |
|------|------|---------|----------|------------|
| **U1** | `family_routing.py` 신규 + 자동 검증 테스트 | `mdp/models/family_routing.py`, `tests/unit/test_family_routing.py` | — | 모든 family에서 `resolve_targets` 왕복 검증 통과. `detect_family` 13개 family 커버 |
| **U2** | adapter 소비자 갱신 | `mdp/models/adapters/lora.py`, `qlora.py`, `tests/unit/test_adapter_semantic.py` | U1 | `target` 인자로 Llama/BERT/T5 recipe 3종 apply_lora 통과. `target_modules` 경로 여전히 작동 (회귀 테스트) |
| **U3** | Factory `_attach_head` slot 지원 | `mdp/factory/factory.py`, `tests/e2e/test_factory_e2e.py` 확장 | U1 | `slot="head.cls"`가 classifier/head/fc로 올바르게 번역. 기존 `_target_attr` 경로 회귀 없음 |
| **U4** | Catalog 50+ 레시피 semantic 전환 + AGENT.md 동기화 | `mdp/models/catalog/**/*.yaml`, `AGENT.md` | U2, U3 | 전 catalog를 `SettingsFactory.for_estimation`으로 파싱 (기존 테스트 확장). AGENT.md는 `target`/`slot` 신규 필드만 반영 |

### 3.1. U1 verify 상세

**자동 검증 테스트** (`test_family_routing.py`):

```python
# tiny 모델로 실제 구조 검증 (가중치 다운로드 없음)
FAMILY_TINY_MODELS = [
    ("llama", "HuggingFaceTB/SmolLM2-135M"),      # Llama-style
    ("bert", "prajjwal1/bert-tiny"),
    ("t5", "google/t5-efficient-tiny"),
    ("gpt2", "sshleifer/tiny-gpt2"),
    ("clip", "openai/clip-vit-base-patch32"),     # config만
    # timm: create_model(pretrained=False)
    ("vit", "vit_base_patch16_224"),
    ("convnext", "convnext_base"),
    ("efficientnet", "efficientnet_b0"),
]

@pytest.mark.parametrize("family,model_id", FAMILY_TINY_MODELS)
def test_family_mapping_matches_actual_modules(family, model_id):
    """FAMILY_ROUTING의 모든 actual name이 실제 모델의 named_modules에 존재."""
    model = _load_model_struct_only(model_id)
    mapping = resolve_family(family)
    actual_names = {n for n, _ in model.named_modules()}
    for semantic, actual in mapping.items():
        assert any(actual in n for n in actual_names), (
            f"family={family} semantic={semantic} actual={actual} not found"
        )
```

이 테스트가 CI에서 매 빌드 실행되어 transformers/timm 업데이트로 인한 drift를 즉시 감지.

### 3.2. U2 verify 상세

- `target: [attn.q, attn.v]`가 Llama에서 `[q_proj, v_proj]`로 번역되는지 단위 테스트
- 같은 recipe를 BERT tiny/T5 tiny에 주면 `[query, value]` / `[q, v]`로 번역되는지
- `target`과 `target_modules` 동시 지정 → `ValueError`
- `target=None` + `target_modules=None` → PEFT `target_modules=None`으로 전달 (자동 매핑)
- **회귀 확인**: 기존 `tests/e2e/test_adapter_e2e.py`가 수정 없이 통과 (raw name 경로)

### 3.3. U3 verify 상세

- `slot: head.cls` + ViT (HF) → `target_attr="classifier"`
- `slot: head.cls` + ViT (timm) → `target_attr="head"`
- `slot: head.lm` + Llama → `target_attr="lm_head"`
- 기존 `_target_attr` 경로 회귀 없음

### 3.4. U4 verify 상세

- `tests/unit/test_recipe_fixtures.py::test_fixture_parses_via_factory`가 **모든** catalog를 `SettingsFactory`로 성공 파싱
- `test_family_routing.py`의 semantic round-trip이 catalog에서 사용한 모든 `target` 값을 커버

---

## 4. 범위 밖 (후속 spec에서 다룰 것)

이번 spec은 **target_modules, head slot, modules_to_save** 3개 영역에 집중. 다음 영역은 **후속 spec으로 분리**:

| 후속 영역 | 왜 분리하는가 |
|----------|-------------|
| `augmentation: auto` (image_processor 자동 로드) | 데이터 파이프라인 영역. family_routing과 별개의 HF AutoImageProcessor 통합 필요 |
| `generation: auto` (generation_config 자동 로드) | CLI 경로에서 HF GenerationConfig 로딩. 추론/생성 파이프라인 수정 |
| PEFT `task_type` 자동 결정 | `recipe.task` 기반 단순 매핑. 분리된 spec에서 일괄 처리 가능 |

이번 spec 범위를 좁혀 각 Unit이 깨끗이 수렴하도록 한다. 후속 spec은 본 구조(family_routing) 위에 얹는 형태로 설계.

---

## 5. 위험 및 트레이드오프

### 5.1. 위험

| 위험 | 가능성 | 완화 |
|------|--------|------|
| family 감지 실패 (알 수 없는 모델) | 중 | `detect_family`에서 명확한 ValueError + 사용자에게 `target_modules` raw 경로 안내 |
| 새 transformers 버전에서 모듈명 변경 | 중 | 자동 검증 테스트가 CI에서 잡음 |
| semantic 이름 오타 (`attn.que`) | 낮음 | resolver가 unknown dot-path는 ValueError로 거부 (raw와 구분) |
| catalog 전환 중 일부 레시피가 drift | 낮음 | U4 verify에서 전 catalog 파싱 검증 |
| 사용자 혼란 (`target` vs `target_modules`) | 중 | AGENT.md에 명확한 가이드 + 동시 지정 방지 |

### 5.2. 트레이드오프

**얻는 것**:
- Recipe의 모델 family 독립성 (Qwen Recipe를 Llama로 바꾸려면 `pretrained`만 수정)
- Catalog drift 위험 소멸 (자동 검증)
- 커스텀 단위를 `attn.q` 수준으로 정밀화
- LoRA 외 영역(head, save, augmentation, generation)으로 확장 가능한 패턴 확립

**잃는 것** (허용):
- `family_routing.py` 유지보수 (새 model family 추가 시 매핑 추가 필요)
- 신규 모듈 1개 + 신규 키 2개(`target`, `slot`) → 학습 곡선

**되돌림 가능성**: `target`/`slot`은 선택적 신규 키. `target_modules`/`_target_attr`는 그대로 지원. 이번 변경을 롤백해도 기존 recipe는 전부 작동. Catalog는 semantic 변환 후 되돌리기 쉽지 않지만, 문법적으로 YAML 값만 바뀌므로 git revert로 복구 가능.

---

## 6. Verify 기준 (spec 수렴)

### 6.1. 기능 검증

- [ ] 13개 family 모두 `detect_family`로 정확히 식별
- [ ] `resolve_targets`가 모든 family에서 알려진 semantic name을 올바른 actual name으로 번역
- [ ] `resolve_head_slot`, `resolve_save_modules` 동일 검증
- [ ] wildcard(`attn.*`, `mlp.*`, `*`) 정상 확장
- [ ] raw name(dot 없음)은 번역 없이 패스스루
- [ ] unknown semantic dot-path는 ValueError (오타 방지)
- [ ] `target`과 `target_modules` 동시 지정 시 ValueError
- [ ] 둘 다 None → PEFT 자동 매핑 (target_modules=None 전달)

### 6.2. 회귀 검증

- [ ] 기존 `test_adapter_e2e.py` (raw target_modules 경로) 수정 없이 통과
- [ ] 기존 `test_factory_e2e.py` (raw _target_attr 경로) 수정 없이 통과
- [ ] 전 catalog 파싱 테스트 통과
- [ ] unit test 전체 통과 (현 baseline: 134 passed, 6 skipped)

### 6.3. 드리프트 방지

- [ ] `test_family_routing.py`가 tiny 모델 로딩으로 실제 모듈 구조를 검증
- [ ] 이 테스트가 CI 기본 suite에 포함

---

## 7. 문서 갱신 예정 영역 (doc-refine 단계에서 수행)

`docs/` 갱신은 code/review/fix 사이클 수렴 후 doc-refine이 담당:

- `docs/configuration.md` — Recipe의 `adapter.target` 섹션 신규
- `docs/extending.md` — 커스텀 family 추가 방법 (`family_routing`에 엔트리 추가하는 워크플로)
- `docs/training.md` — LoRA 사용 예시를 semantic으로 갱신
- `AGENT.md` — code 단계에서 즉시 동기화 (U4에서 포함)

---

## 8. 실행 순서 (code phase 오케스트레이션)

1. **U1 착수** — `family_routing.py` + 자동 검증 테스트 구현. CI에서 통과 확인
2. **U2/U3 병렬** — adapter 소비자 + Factory head slot 갱신. 기존 테스트 회귀 없음 확인
3. **U4** — catalog 전환 + AGENT.md 동기화. 전 catalog 파싱 검증
4. **review → fix 사이클** — 수렴까지 반복
5. **doc-refine** — `docs/` 일괄 갱신, 본 spec archive

수렴 후 `_dev-cycle.md` "산출물"과 "이력"에 반영.
