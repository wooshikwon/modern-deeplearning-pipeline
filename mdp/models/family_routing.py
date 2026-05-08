"""Family별 semantic name → actual module name 매핑.

단일 진실의 원천. 새 모델 family 추가 시 여기만 업데이트한다.

모듈 구조
--------
- ``_FAMILY_ROUTING``: family 이름 → semantic→actual 매핑 dict 또는 alias 문자열.
  alias는 다른 family와 동일한 네이밍 규약을 공유할 때 사용한다 (예: ``"mistral": "llama"``).
- ``detect_family(model)``: 모델 인스턴스에서 family 문자열을 추론한다.
- ``detect_family_from_pretrained_uri(pretrained, component)``:
  HF pretrained URI에서 AutoConfig로 family를 추정한다 (QLoRA 경로용).
- ``resolve_family(role)``: family 이름(또는 alias)을 최종 매핑 dict로 해석한다.
- ``resolve_targets(targets, family)``, ``resolve_head_slot(slot, family)``,
  ``resolve_save_modules(saves, family)``:
  semantic dot-path 리스트를 실제 모듈 이름 리스트로 번역하는 순수 함수.
  인자가 model이 아닌 family 문자열 — AssemblyMaterializer가 family를 추출해서 전달한다.
"""

from __future__ import annotations

from torch import nn

# ──────────────────────────────────────────────────────────────────────
# Family routing table
# ──────────────────────────────────────────────────────────────────────
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
        "attn.o": "attention.output.dense",
        "mlp.fc1": "intermediate.dense",
        # mlp.fc2 의도적 미지원: BERT의 MLP output은 "output.dense"이나,
        # PEFT suffix 매칭에서 endswith(".output.dense")가 attention의
        # "attention.output.dense"까지 매칭한다. 구분 불가능하므로 제거.
        # MLP LoRA가 필요하면 mlp.fc1 (intermediate.dense)을 사용하거나
        # raw target_modules로 직접 지정.
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
        # GPT-2의 attention output은 "attn.c_proj",
        # MLP output은 "mlp.c_proj"이다. 짧은 "c_proj"는
        # PEFT suffix 매칭에서 양쪽 모두 매칭되므로,
        # 더 구체적인 경로를 사용하여 두 매핑이 구분되도록 한다.
        "attn.qkv": "c_attn", "attn.o": "attn.c_proj",
        "mlp.fc1": "c_fc", "mlp.fc2": "mlp.c_proj",
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
    "blip-2": "clip",

    # HF ViT (google/vit-*): BERT-style 모듈명.
    # config.model_type="vit"으로 감지되며, query/key/value/attention.output.dense를 사용한다.
    # mlp.fc2 미지원 — bert family와 동일 사유 (PEFT suffix 충돌).
    "vit": {
        "attn.q": "query", "attn.k": "key",
        "attn.v": "value", "attn.o": "attention.output.dense",
        "mlp.fc1": "intermediate.dense",
        "head.cls": "classifier",
    },
    # timm ViT (vit_base_patch16_224 등): timm 고유 모듈명.
    # _TIMM_PREFIX_MAP을 통해 detect_family가 "vit_timm"을 반환한다.
    "vit_timm": {
        "attn.qkv": "qkv", "attn.o": "proj",
        "mlp.fc1": "fc1", "mlp.fc2": "fc2",
        "head.cls": "head",
    },
    # HF Swin (microsoft/swin-*): BERT-style 모듈명.
    # config.model_type="swin"으로 감지된다.
    # mlp.fc2 미지원 — bert family와 동일 사유 (PEFT suffix 충돌).
    "swin": {
        "attn.q": "query", "attn.k": "key",
        "attn.v": "value", "attn.o": "attention.output.dense",
        "mlp.fc1": "intermediate.dense",
        "head.cls": "classifier",
    },
    # timm Swin은 timm ViT와 동일한 모듈명 규약을 사용한다.
    "swin_timm": "vit_timm",

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

    "llava": "llama",

    "mixtral": {
        "attn.q": "q_proj", "attn.k": "k_proj",
        "attn.v": "v_proj", "attn.o": "o_proj",
        # MoE expert MLP는 transformers 5.x에서 nn.Parameter라 PEFT LoRA 불가.
        # 의도적으로 mlp.* 매핑 없음.
        "head.lm": "lm_head",
    },
}

# timm architecture prefix → family 매핑
_TIMM_PREFIX_MAP: dict[str, str] = {
    "vit": "vit_timm",
    "swin": "swin_timm",
    "convnext": "convnext",
    "efficientnet": "efficientnet",
    "tf_efficientnet": "efficientnet",
}


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def detect_family(model: nn.Module) -> str:
    """모델 인스턴스 → family 문자열.

    우선순위:
    1. HF: model.config.model_type (가장 권위 있는 소스)
    2. timm: model.default_cfg["architecture"]에서 프리픽스로 family 추출
    3. torchvision: type(model).__name__으로 폴백

    알려진 family에 매핑되지 않으면 ValueError를 발생시킨다.
    이 경우 사용자는 target_modules에 raw name 리스트를 직접 지정하여 우회할 수 있다.
    """
    # 1) HF transformers: config.model_type
    config = getattr(model, "config", None)
    if config is not None:
        model_type = getattr(config, "model_type", None)
        if model_type is not None and model_type in _FAMILY_ROUTING:
            return model_type

    # 2) timm: default_cfg["architecture"]
    default_cfg = getattr(model, "default_cfg", None)
    if default_cfg is not None and isinstance(default_cfg, dict):
        arch = default_cfg.get("architecture", "")
        arch_lower = arch.lower()
        for prefix, family in _TIMM_PREFIX_MAP.items():
            if arch_lower.startswith(prefix):
                return family

    # 3) torchvision: class name fallback
    class_name = type(model).__name__.lower()
    if "resnet" in class_name:
        return "resnet"

    raise ValueError(
        f"모델 family를 감지할 수 없습니다: {type(model).__name__}. "
        "target_modules에 raw module name 리스트를 직접 지정하여 우회하세요."
    )


def detect_family_from_pretrained_uri(
    pretrained: str, component: str | None = None,
) -> str:
    """model 인스턴스 없이 pretrained URI에서 family를 추정한다 (QLoRA 경로용).

    QLoRA는 양자화+로딩+어댑터를 한 번에 수행하므로 model 인스턴스가 resolve
    시점에 존재하지 않는다. 대신 HF AutoConfig에서 model_type을 추출한다.

    Args:
        pretrained: HF 모델 URI (``hf://meta-llama/Meta-Llama-3-8B`` 형식).
        component: model config의 ``_component_`` 값 (현재 미사용, 향후 확장용).

    Returns:
        family 문자열 (예: ``"llama"``).

    Raises:
        ValueError: ``hf://`` 프로토콜이 아닌 경우, 또는 model_type이
            ``_FAMILY_ROUTING``에 없는 경우.
    """
    if pretrained is None:
        raise ValueError(
            "QLoRA에서 semantic target/save를 사용하려면 pretrained URI가 필요합니다. "
            "Recipe의 model.pretrained를 지정하거나, raw target_modules로 우회하세요."
        )

    if not pretrained.startswith("hf://"):
        raise ValueError(
            f"detect_family_from_pretrained_uri는 hf:// 프로토콜만 지원합니다: "
            f"{pretrained!r}. raw target_modules를 직접 지정하여 우회하세요."
        )

    model_id = pretrained[len("hf://"):]

    try:
        from transformers import AutoConfig
    except ImportError as e:
        raise ImportError(
            "transformers 패키지가 필요합니다: pip install transformers"
        ) from e

    config = AutoConfig.from_pretrained(model_id)
    model_type = getattr(config, "model_type", None)

    if model_type is None or model_type not in _FAMILY_ROUTING:
        raise ValueError(
            f"pretrained URI {pretrained!r}의 model_type={model_type!r}이 "
            f"_FAMILY_ROUTING에 없습니다. "
            "target_modules에 raw module name 리스트를 직접 지정하여 우회하세요."
        )

    return model_type


def resolve_family(role: str) -> dict[str, str]:
    """family 이름(또는 alias)을 최종 semantic→actual 매핑 dict로 해석한다.

    alias 체인을 따라가되 순환 참조는 ValueError로 거부한다.

    Args:
        role: family 이름 문자열 (예: ``"llama"``, ``"mistral"``).

    Returns:
        semantic dot-path → actual module name 매핑 dict.

    Raises:
        ValueError: 알 수 없는 family이거나 순환 alias인 경우.
    """
    seen: set[str] = set()
    current = role
    while isinstance(current, str):
        if current in seen:
            raise ValueError(f"순환 alias 감지: {role!r}")
        seen.add(current)
        entry = _FAMILY_ROUTING.get(current)
        if entry is None:
            raise ValueError(f"알 수 없는 family: {role!r}")
        current = entry
    return current


def resolve_targets(
    targets: list[str] | str | None,
    family: str,
) -> list[str] | str | None:
    """Semantic target 리스트를 actual module name 리스트로 번역한다.

    **인자가 model이 아닌 family 문자열** -- AssemblyMaterializer가 family를 추출해서 전달한다.
    이렇게 resolve 함수를 model에 무관한 순수 함수로 유지한다.

    변환 규칙:
    - ``None`` → ``None`` 반환 (호출부가 PEFT 자동 매핑에 위임).
    - ``"*"`` 또는 ``"all-linear"`` → ``"all-linear"`` 패스스루.
    - 리스트의 각 원소:
      - dot(``.``) 포함 + ``.*`` 접미 (wildcard): family의 해당 prefix 아래
        모든 semantic 키를 확장.
      - dot 포함 + 알려진 semantic: 해당 actual name으로 번역.
      - dot 없음: raw name으로 취급하여 그대로 유지.
      - dot 포함하지만 unknown semantic: ``ValueError`` (오타 방지).

    Args:
        targets: semantic target 리스트, 단일 문자열, 또는 None.
        family: family 이름 문자열 (예: ``"llama"``, ``"bert"``).

    Returns:
        번역된 actual module name 리스트, ``"all-linear"`` 문자열, 또는 None.
    """
    if targets is None:
        return None

    # 전체 와일드: 문자열 "all-linear" 또는 "*"
    if isinstance(targets, str):
        if targets in ("*", "all-linear"):
            return "all-linear"
        # 단일 문자열을 리스트로 정규화
        targets = [targets]

    # 리스트 내에 "*" 또는 "all-linear"가 유일 원소로 들어온 경우
    if len(targets) == 1 and targets[0] in ("*", "all-linear"):
        return "all-linear"

    # 전체 Linear 와일드("*" 또는 "all-linear")는 단독으로만 사용 가능.
    # 다른 semantic과 함께 리스트에 들어오면 의미적으로 모순이므로 거부한다.
    wild = {"*", "all-linear"} & set(targets)
    if wild and len(targets) > 1:
        raise ValueError(
            "전체 Linear 와일드('*' 또는 'all-linear')는 단독으로만 사용할 수 있습니다. "
            "다른 semantic과 혼용할 수 없습니다."
        )

    mapping = resolve_family(family)

    result: list[str] = []
    for target in targets:
        if "." not in target:
            # raw name: 번역 없이 그대로 유지
            result.append(target)
            continue

        if target.endswith(".*"):
            # wildcard: prefix 아래 모든 매핑 확장
            prefix = target[:-2]  # "attn.*" → "attn"
            expanded = [
                actual
                for semantic, actual in mapping.items()
                if semantic.startswith(prefix + ".")
            ]
            if not expanded:
                raise ValueError(
                    f"family={family!r}에서 prefix={prefix!r}에 해당하는 "
                    f"semantic 매핑이 없습니다."
                )
            result.extend(expanded)
        elif target in mapping:
            # 알려진 semantic: 번역
            result.append(mapping[target])
        else:
            raise ValueError(
                f"family={family!r}에서 알 수 없는 semantic target: {target!r}. "
                f"사용 가능한 semantic: {sorted(mapping.keys())}"
            )

    return result


def resolve_head_slot(slot: str | None, family: str) -> str | None:
    """head.cls / head.lm / head.det / head.seg → 실제 attribute name.

    Args:
        slot: semantic head slot 이름 (예: ``"head.cls"``), 또는 None.
        family: family 이름 문자열 (예: ``"llama"``, ``"bert"``).

    Returns:
        실제 attribute name 문자열, 또는 None (slot이 None인 경우).

    Raises:
        ValueError: slot이 family 매핑에 존재하지 않는 경우.
    """
    if slot is None:
        return None

    mapping = resolve_family(family)

    actual = mapping.get(slot)
    if actual is None:
        raise ValueError(
            f"family={family!r}에서 알 수 없는 head slot: {slot!r}. "
            f"사용 가능한 head slot: "
            f"{sorted(k for k in mapping if k.startswith('head.'))}"
        )
    return actual


def resolve_save_modules(
    saves: list[str] | None,
    family: str,
) -> list[str] | None:
    """modules_to_save의 semantic 이름을 actual name으로 번역한다.

    변환 규칙은 ``resolve_targets``와 동일하다:
    - dot 포함 + 알려진 semantic → 번역
    - dot 포함 + wildcard → 확장
    - dot 없음 → raw name 패스스루
    - dot 포함 + unknown → ValueError

    Args:
        saves: semantic module name 리스트 또는 None.
        family: family 이름 문자열 (예: ``"llama"``, ``"bert"``).

    Returns:
        번역된 actual module name 리스트 또는 None.
    """
    if saves is None:
        return None

    mapping = resolve_family(family)

    result: list[str] = []
    for name in saves:
        if "." not in name:
            result.append(name)
            continue

        if name.endswith(".*"):
            prefix = name[:-2]
            expanded = [
                actual
                for semantic, actual in mapping.items()
                if semantic.startswith(prefix + ".")
            ]
            if not expanded:
                raise ValueError(
                    f"family={family!r}에서 prefix={prefix!r}에 해당하는 "
                    f"semantic 매핑이 없습니다."
                )
            result.extend(expanded)
        elif name in mapping:
            result.append(mapping[name])
        else:
            raise ValueError(
                f"family={family!r}에서 알 수 없는 semantic module: {name!r}. "
                f"사용 가능한 semantic: {sorted(mapping.keys())}"
            )

    return result
