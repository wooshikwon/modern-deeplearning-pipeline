"""family_routing 모듈 테스트.

3단계 검증 구조:
1. 정적 검증: _FAMILY_ROUTING 데이터 구조 자체의 무결성
2. 함수 동작 검증: Mock 모델을 사용한 detect_family + family 문자열 기반 resolve_* 함수 테스트
3. 실모델 검증 (skip-by-default): tiny 모델로 매핑 대조 (@pytest.mark.slow)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from torch import nn

from mdp.models.family_routing import (
    _FAMILY_ROUTING,
    detect_family,
    resolve_family,
    resolve_head_slot,
    resolve_save_modules,
    resolve_targets,
)


# ──────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────


def _mock_hf_model(model_type: str) -> nn.Module:
    """HF model.config.model_type를 흉내내는 Mock 모델."""
    model = MagicMock(spec=nn.Module)
    model.config = MagicMock()
    model.config.model_type = model_type
    # timm/torchvision 경로가 작동하지 않도록 제거
    del model.default_cfg
    return model


def _mock_timm_model(architecture: str) -> nn.Module:
    """timm model.default_cfg["architecture"]를 흉내내는 Mock 모델."""
    model = MagicMock(spec=nn.Module)
    # HF config가 없도록 설정
    del model.config
    model.default_cfg = {"architecture": architecture}
    return model


class _FakeResNet(nn.Module):
    """torchvision ResNet을 흉내내는 클래스. 이름에 'resnet'이 포함된다."""
    def forward(self, x: Any) -> Any:
        return x


# ──────────────────────────────────────────────────────────────────────
# 1. 정적 검증: _FAMILY_ROUTING 데이터 구조
# ──────────────────────────────────────────────────────────────────────


class TestFamilyRoutingStructure:
    """_FAMILY_ROUTING 테이블의 구조적 무결성을 검증한다."""

    def test_all_entries_are_dict_or_str(self) -> None:
        """모든 entry가 dict(실제 매핑) 또는 str(alias)이어야 한다."""
        for family, entry in _FAMILY_ROUTING.items():
            assert isinstance(entry, (dict, str)), (
                f"family={family!r}의 entry 타입이 유효하지 않다: {type(entry)}"
            )

    def test_alias_references_resolve_to_existing_family(self) -> None:
        """모든 string alias가 실재하는 family key를 참조해야 한다."""
        for family, entry in _FAMILY_ROUTING.items():
            if isinstance(entry, str):
                assert entry in _FAMILY_ROUTING, (
                    f"alias {family!r} → {entry!r}이지만 "
                    f"{entry!r}이 _FAMILY_ROUTING에 없다"
                )

    def test_alias_targets_are_dicts_not_alias_chains(self) -> None:
        """alias가 가리키는 대상은 dict여야 한다 (2-hop 이상 체인 방지).

        현재 설계에서 alias 체인을 resolve_family가 처리하지만,
        데이터 자체는 1-hop alias를 유지하는 것이 가독성에 좋다.
        """
        for family, entry in _FAMILY_ROUTING.items():
            if isinstance(entry, str):
                target = _FAMILY_ROUTING.get(entry)
                assert isinstance(target, dict), (
                    f"alias {family!r} → {entry!r}이지만, "
                    f"{entry!r}의 entry가 또 alias다: {target!r}"
                )

    def test_semantic_keys_use_dot_path_format(self) -> None:
        """dict entry의 모든 semantic 키가 dot-path 형식(prefix.suffix)이다."""
        for family, entry in _FAMILY_ROUTING.items():
            if isinstance(entry, dict):
                for key in entry:
                    assert "." in key, (
                        f"family={family!r}의 semantic 키 {key!r}에 "
                        f"dot이 없다 (dot-path 형식이어야 함)"
                    )

    def test_semantic_prefixes_are_known(self) -> None:
        """semantic 키의 prefix가 알려진 namespace에 속해야 한다."""
        known_prefixes = {"attn", "mlp", "head", "embed", "conv"}
        for family, entry in _FAMILY_ROUTING.items():
            if isinstance(entry, dict):
                for key in entry:
                    prefix = key.split(".")[0]
                    assert prefix in known_prefixes, (
                        f"family={family!r}의 semantic {key!r}의 prefix "
                        f"{prefix!r}가 알려진 namespace에 없다: {known_prefixes}"
                    )

    def test_expected_families_present(self) -> None:
        """spec에 명시된 모든 family가 테이블에 존재해야 한다."""
        expected = {
            "llama", "mistral", "qwen2", "gemma2",
            "phi3",
            "bert", "roberta", "dinov2", "segformer",
            "t5",
            "gpt2",
            "clip", "siglip", "detr", "florence2", "blip2",
            "vit", "swin",
            "convnext",
            "efficientnet",
            "resnet",
            "mixtral",
        }
        actual = set(_FAMILY_ROUTING.keys())
        missing = expected - actual
        assert not missing, f"누락된 family: {missing}"

    def test_actual_values_are_strings(self) -> None:
        """dict entry의 모든 actual value가 비어있지 않은 문자열이어야 한다."""
        for family, entry in _FAMILY_ROUTING.items():
            if isinstance(entry, dict):
                for semantic, actual in entry.items():
                    assert isinstance(actual, str) and actual, (
                        f"family={family!r} semantic={semantic!r}의 "
                        f"actual 값이 유효하지 않다: {actual!r}"
                    )


# ──────────────────────────────────────────────────────────────────────
# 2. resolve_family 함수 검증
# ──────────────────────────────────────────────────────────────────────


class TestResolveFamily:
    """resolve_family 함수의 동작을 검증한다."""

    def test_direct_family_returns_dict(self) -> None:
        result = resolve_family("llama")
        assert isinstance(result, dict)
        assert "attn.q" in result

    def test_alias_resolves_to_target_dict(self) -> None:
        """alias(mistral→llama)가 llama의 dict를 반환해야 한다."""
        llama = resolve_family("llama")
        mistral = resolve_family("mistral")
        assert mistral is llama  # 동일 dict 객체

    def test_all_aliases_resolve(self) -> None:
        """모든 alias가 dict로 최종 resolve되어야 한다."""
        for family in _FAMILY_ROUTING:
            result = resolve_family(family)
            assert isinstance(result, dict), (
                f"family={family!r}의 resolve 결과가 dict가 아니다"
            )

    def test_unknown_family_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="알 수 없는 family"):
            resolve_family("unknown_model")


# ──────────────────────────────────────────────────────────────────────
# 3. detect_family 함수 검증 (Mock)
# ──────────────────────────────────────────────────────────────────────


class TestDetectFamily:
    """detect_family 함수의 3단계 감지 로직을 검증한다."""

    @pytest.mark.parametrize("model_type", [
        "llama", "mistral", "qwen2", "gemma2", "phi3",
        "bert", "roberta", "dinov2", "segformer",
        "t5", "gpt2",
        "clip", "siglip", "detr", "florence2", "blip2",
        "vit", "swin", "convnext", "efficientnet",
        "mixtral",
    ])
    def test_hf_model_type_detection(self, model_type: str) -> None:
        """HF config.model_type로 family를 정확히 감지해야 한다."""
        model = _mock_hf_model(model_type)
        assert detect_family(model) == model_type

    @pytest.mark.parametrize("arch,expected_family", [
        ("vit_base_patch16_224", "vit"),
        ("vit_large_patch32_384", "vit"),
        ("swin_base_patch4_window7_224", "vit"),
        ("convnext_base", "convnext"),
        ("convnext_tiny", "convnext"),
        ("efficientnet_b0", "efficientnet"),
        ("tf_efficientnet_b3", "efficientnet"),
    ])
    def test_timm_detection(self, arch: str, expected_family: str) -> None:
        """timm default_cfg["architecture"]로 family를 감지해야 한다."""
        model = _mock_timm_model(arch)
        assert detect_family(model) == expected_family

    def test_torchvision_resnet_detection(self) -> None:
        """torchvision ResNet을 클래스명으로 감지해야 한다."""
        model = _FakeResNet()
        assert detect_family(model) == "resnet"

    def test_unknown_model_raises_value_error(self) -> None:
        """알 수 없는 모델은 ValueError를 발생시켜야 한다."""
        model = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="모델 family를 감지할 수 없습니다"):
            detect_family(model)


# ──────────────────────────────────────────────────────────────────────
# 4. resolve_targets 함수 검증
# ──────────────────────────────────────────────────────────────────────


class TestResolveTargets:
    """resolve_targets의 모든 변환 경로를 검증한다."""

    def test_none_returns_none(self) -> None:
        """None 입력 → None 반환 (PEFT 자동 매핑 위임)."""
        assert resolve_targets(None, "llama") is None

    def test_star_string_returns_all_linear(self) -> None:
        """'*' → 'all-linear' 패스스루."""
        assert resolve_targets("*", "llama") == "all-linear"

    def test_all_linear_string_passthrough(self) -> None:
        """'all-linear' → 'all-linear' 패스스루."""
        assert resolve_targets("all-linear", "bert") == "all-linear"

    def test_star_in_list_returns_all_linear(self) -> None:
        """['*'] → 'all-linear' 패스스루."""
        assert resolve_targets(["*"], "llama") == "all-linear"

    def test_all_linear_in_list_returns_all_linear(self) -> None:
        """['all-linear'] → 'all-linear' 패스스루."""
        assert resolve_targets(["all-linear"], "llama") == "all-linear"

    def test_semantic_to_actual_llama(self) -> None:
        """Llama: semantic [attn.q, attn.v] → [q_proj, v_proj]."""
        result = resolve_targets(["attn.q", "attn.v"], "llama")
        assert result == ["q_proj", "v_proj"]

    def test_semantic_to_actual_bert(self) -> None:
        """BERT: semantic [attn.q, attn.v] → [query, value]."""
        result = resolve_targets(["attn.q", "attn.v"], "bert")
        assert result == ["query", "value"]

    def test_semantic_to_actual_t5(self) -> None:
        """T5: semantic [attn.q, attn.v] → [q, v]."""
        result = resolve_targets(["attn.q", "attn.v"], "t5")
        assert result == ["q", "v"]

    def test_semantic_to_actual_gpt2(self) -> None:
        """GPT-2: semantic [attn.qkv] → [c_attn]."""
        result = resolve_targets(["attn.qkv"], "gpt2")
        assert result == ["c_attn"]

    def test_semantic_to_actual_clip(self) -> None:
        """CLIP: semantic [attn.q, attn.o] → [q_proj, out_proj]."""
        result = resolve_targets(["attn.q", "attn.o"], "clip")
        assert result == ["q_proj", "out_proj"]

    def test_wildcard_expansion_llama_attn(self) -> None:
        """Llama: attn.* → [q_proj, k_proj, v_proj, o_proj]."""
        result = resolve_targets(["attn.*"], "llama")
        assert set(result) == {"q_proj", "k_proj", "v_proj", "o_proj"}

    def test_wildcard_expansion_llama_mlp(self) -> None:
        """Llama: mlp.* → [gate_proj, up_proj, down_proj]."""
        result = resolve_targets(["mlp.*"], "llama")
        assert set(result) == {"gate_proj", "up_proj", "down_proj"}

    def test_wildcard_expansion_bert_attn(self) -> None:
        """BERT: attn.* → [query, key, value, output.dense]."""
        result = resolve_targets(["attn.*"], "bert")
        assert set(result) == {"query", "key", "value", "output.dense"}

    def test_wildcard_and_semantic_mixed(self) -> None:
        """wildcard와 개별 semantic을 섞어 사용할 수 있다."""
        result = resolve_targets(["attn.*", "mlp.gate"], "llama")
        assert "q_proj" in result
        assert "gate_proj" in result

    def test_raw_name_passthrough(self) -> None:
        """dot 없는 원소는 raw name으로 그대로 유지."""
        result = resolve_targets(["q_proj", "custom_layer"], "llama")
        assert result == ["q_proj", "custom_layer"]

    def test_raw_and_semantic_mixed(self) -> None:
        """raw name과 semantic dot-path를 섞어 사용할 수 있다."""
        result = resolve_targets(["attn.q", "custom_layer"], "llama")
        assert result == ["q_proj", "custom_layer"]

    def test_unknown_semantic_raises_value_error(self) -> None:
        """알 수 없는 semantic dot-path는 ValueError (오타 방지)."""
        with pytest.raises(ValueError, match="알 수 없는 semantic target"):
            resolve_targets(["attn.que"], "llama")

    def test_unknown_wildcard_prefix_raises_value_error(self) -> None:
        """존재하지 않는 prefix의 wildcard는 ValueError."""
        with pytest.raises(ValueError, match="semantic 매핑이 없습니다"):
            resolve_targets(["nonexist.*"], "llama")

    def test_single_string_semantic(self) -> None:
        """단일 문자열 semantic도 리스트로 정규화되어 번역된다."""
        result = resolve_targets("attn.q", "llama")
        assert result == ["q_proj"]

    def test_alias_family_works(self) -> None:
        """alias family(mistral→llama)에서도 번역이 정상 동작한다."""
        result = resolve_targets(["attn.q", "attn.v"], "mistral")
        assert result == ["q_proj", "v_proj"]

    def test_mixtral_no_mlp_mapping(self) -> None:
        """Mixtral은 mlp.* 매핑이 없으므로 mlp wildcard에서 에러."""
        with pytest.raises(ValueError, match="semantic 매핑이 없습니다"):
            resolve_targets(["mlp.*"], "mixtral")

    def test_mixtral_attn_works(self) -> None:
        """Mixtral의 attn 매핑은 정상 동작해야 한다."""
        result = resolve_targets(["attn.q", "attn.v"], "mixtral")
        assert result == ["q_proj", "v_proj"]


# ──────────────────────────────────────────────────────────────────────
# 5. resolve_head_slot 함수 검증
# ──────────────────────────────────────────────────────────────────────


class TestResolveHeadSlot:
    """resolve_head_slot의 동작을 검증한다."""

    def test_none_returns_none(self) -> None:
        assert resolve_head_slot(None, "bert") is None

    def test_bert_head_cls(self) -> None:
        assert resolve_head_slot("head.cls", "bert") == "classifier"

    def test_llama_head_lm(self) -> None:
        assert resolve_head_slot("head.lm", "llama") == "lm_head"

    def test_vit_head_cls(self) -> None:
        assert resolve_head_slot("head.cls", "vit") == "head"

    def test_resnet_head_cls(self) -> None:
        assert resolve_head_slot("head.cls", "resnet") == "fc"

    def test_unknown_slot_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="알 수 없는 head slot"):
            resolve_head_slot("head.det", "bert")


# ──────────────────────────────────────────────────────────────────────
# 6. resolve_save_modules 함수 검증
# ──────────────────────────────────────────────────────────────────────


class TestResolveSaveModules:
    """resolve_save_modules의 동작을 검증한다."""

    def test_none_returns_none(self) -> None:
        assert resolve_save_modules(None, "llama") is None

    def test_semantic_translation(self) -> None:
        result = resolve_save_modules(["head.lm", "embed.token"], "llama")
        assert result == ["lm_head", "embed_tokens"]

    def test_raw_passthrough(self) -> None:
        result = resolve_save_modules(["lm_head", "custom_module"], "llama")
        assert result == ["lm_head", "custom_module"]

    def test_wildcard_expansion(self) -> None:
        result = resolve_save_modules(["embed.*"], "llama")
        assert "embed_tokens" in result

    def test_unknown_semantic_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="알 수 없는 semantic module"):
            resolve_save_modules(["embed.unknown"], "llama")


# ──────────────────────────────────────────────────────────────────────
# 7. Integration test (실모델, skip-by-default)
# ──────────────────────────────────────────────────────────────────────

# 실모델 로딩에 네트워크/HF cache가 필요할 수 있으므로 slow 마커로 기본 skip.
# 실행: pytest -m slow

_HF_TINY_MODELS: list[tuple[str, str]] = [
    ("llama", "HuggingFaceTB/SmolLM2-135M"),
    ("bert", "hf-internal-testing/tiny-random-BertModel"),
    # google/t5-efficient-tiny는 T5 v1.0 (non-gated: wi/wo)이라 우리 매핑의
    # wi_0/wi_1 (gated T5 v1.1)과 불일치. _load_hf_model_struct_only에서
    # T5는 gated custom config로 직접 생성한다.
    ("t5", "__custom_gated_t5__"),
    ("gpt2", "sshleifer/tiny-gpt2"),
]

_TIMM_MODELS: list[tuple[str, str]] = [
    ("vit", "vit_base_patch16_224"),
    ("convnext", "convnext_base"),
    ("efficientnet", "efficientnet_b0"),
]


def _load_hf_model_struct_only(model_id: str) -> nn.Module:
    """HF 모델을 config만으로 인스턴스화한다 (가중치 없이 구조만).

    특수 케이스:
    - ``__custom_gated_t5__``: google/t5-efficient-tiny는 T5 v1.0 (non-gated FFN:
      wi/wo)이라 우리 매핑(wi_0/wi_1, gated T5 v1.1)과 불일치한다. 네트워크
      의존 없이 gated FFN 구조를 검증하기 위해 최소 T5 config을 직접 생성한다.
    """
    try:
        import transformers
    except ImportError:
        pytest.skip("transformers 미설치")

    # T5 gated 특수 처리: 최소 config으로 wi_0/wi_1 구조를 보장
    if model_id == "__custom_gated_t5__":
        config = transformers.T5Config(
            vocab_size=100,
            d_model=32,
            d_ff=64,
            d_kv=8,
            num_heads=4,
            num_layers=2,
            num_decoder_layers=2,
            dense_act_fn="gelu_new",
            is_gated_act=True,
        )
        return transformers.T5ForConditionalGeneration(config)

    config = transformers.AutoConfig.from_pretrained(model_id)
    architectures = getattr(config, "architectures", None) or []
    if not architectures:
        pytest.skip(f"{model_id}: config.architectures 없음")
    cls_name = architectures[0]

    # head.cls 등 task head 매핑 검증을 위해, base 모델(예: BertModel)은
    # task-specific 변종(BertForSequenceClassification)으로 교체한다.
    # base 모델에는 classifier head가 없기 때문이다.
    _TASK_MODEL_OVERRIDE: dict[str, str] = {
        "BertModel": "BertForSequenceClassification",
    }
    cls_name = _TASK_MODEL_OVERRIDE.get(cls_name, cls_name)

    model_cls = getattr(transformers, cls_name, None)
    if model_cls is None:
        pytest.skip(f"transformers에 {cls_name} 클래스 없음")
    # _no_init_weights로 빠르게 빈 모델 생성 (가중치 다운로드 없음)
    return model_cls(config)


def _load_timm_model(model_name: str) -> nn.Module:
    """timm 모델을 가중치 없이 인스턴스화한다."""
    try:
        import timm
    except ImportError:
        pytest.skip("timm 미설치")
    return timm.create_model(model_name, pretrained=False)


@pytest.mark.slow
class TestIntegrationHF:
    """HF tiny 모델로 실제 named_modules와 매핑을 대조한다."""

    @pytest.mark.parametrize("family,model_id", _HF_TINY_MODELS)
    def test_family_detection(self, family: str, model_id: str) -> None:
        """실제 모델에서 detect_family가 올바른 family를 반환해야 한다."""
        model = _load_hf_model_struct_only(model_id)
        assert detect_family(model) == family

    @pytest.mark.parametrize("family,model_id", _HF_TINY_MODELS)
    def test_mapping_matches_actual_modules(
        self, family: str, model_id: str,
    ) -> None:
        """FAMILY_ROUTING의 모든 actual name이 실제 모델의 named_modules에 존재."""
        model = _load_hf_model_struct_only(model_id)
        mapping = resolve_family(family)
        module_names = {n for n, _ in model.named_modules()}
        # named_modules()는 전체 경로(예: "model.layers.0.self_attn.q_proj")를
        # 반환하므로, actual name이 경로의 일부(suffix)로 존재하는지 확인한다.
        for semantic, actual in mapping.items():
            found = any(
                name == actual or name.endswith("." + actual)
                for name in module_names
            )
            assert found, (
                f"family={family} model={model_id} semantic={semantic} "
                f"actual={actual} not found in named_modules"
            )


@pytest.mark.slow
class TestIntegrationTimm:
    """timm 모델로 실제 named_modules와 매핑을 대조한다."""

    @pytest.mark.parametrize("family,model_name", _TIMM_MODELS)
    def test_family_detection(self, family: str, model_name: str) -> None:
        model = _load_timm_model(model_name)
        assert detect_family(model) == family

    @pytest.mark.parametrize("family,model_name", _TIMM_MODELS)
    def test_mapping_matches_actual_modules(
        self, family: str, model_name: str,
    ) -> None:
        model = _load_timm_model(model_name)
        mapping = resolve_family(family)
        module_names = {n for n, _ in model.named_modules()}
        for semantic, actual in mapping.items():
            found = any(
                name == actual or name.endswith("." + actual)
                for name in module_names
            )
            assert found, (
                f"family={family} model={model_name} semantic={semantic} "
                f"actual={actual} not found in named_modules"
            )
