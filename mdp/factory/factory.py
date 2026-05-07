"""Factory 파사드 — Settings를 받아 모든 컴포넌트를 생성하는 중앙 관리자."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable

import torch.nn as nn

from mdp.settings.resolver import ComponentResolver
from mdp.settings.schema import Settings

logger = logging.getLogger(__name__)


class _ModelLoadRoute(Enum):
    COMPONENT_FROM_PRETRAINED = "component_from_pretrained"
    COMPONENT_CONSTRUCTOR = "component_constructor"
    PRETRAINED_RESOLVER = "pretrained_resolver"
    QLORA = "qlora"


def _decide_model_load_route(
    model_config: dict[str, Any],
    adapter_config: dict[str, Any] | None,
) -> _ModelLoadRoute:
    component = model_config.get("_component_")
    pretrained = model_config.get("pretrained")

    if adapter_config is not None and _is_qlora_adapter_config(adapter_config):
        return _ModelLoadRoute.QLORA
    if component is not None and pretrained is not None:
        return _ModelLoadRoute.COMPONENT_FROM_PRETRAINED
    if component is not None:
        return _ModelLoadRoute.COMPONENT_CONSTRUCTOR
    if pretrained is not None:
        return _ModelLoadRoute.PRETRAINED_RESOLVER
    raise ValueError("model에 _component_ 또는 pretrained가 필요합니다")


def _is_qlora_adapter_config(adapter_config: dict[str, Any]) -> bool:
    component = adapter_config.get("_component_", "")
    return component in {"QLoRA", "mdp.models.adapters.qlora.apply_qlora"}


class Factory:
    """컴포넌트 생성 중앙 파사드.

    모든 create_* 메서드는 동일한 Settings에서 동일한 컴포넌트를 요청하면
    캐싱된 인스턴스를 반환한다 (싱글턴 보장).
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.resolver = ComponentResolver()
        self._cache: dict[str, Any] = {}

    _SENTINEL = object()

    def _get_or_create(self, key: str, creator: Callable) -> Any:
        """캐시에서 key를 찾고, 없으면 creator()를 호출하여 캐싱한다."""
        cached = self._cache.get(key, self._SENTINEL)
        if cached is not self._SENTINEL:
            return cached
        instance = creator()
        if instance is not None:
            self._cache[key] = instance
        return instance

    # ── Phase 2: 모델 생성 ──

    def create_model(self, skip_base_check: bool = False) -> nn.Module:
        """Recipe의 model 설정에 따라 모델을 생성한다.

        순서: pretrained 로딩 → MoE 감지 → head 교체 → BaseModel 검증 → adapter 적용.
        QLoRA는 양자화+로딩+어댑터가 결합된 특수 경로를 탄다.
        """
        return self._get_or_create("model", lambda: self._build_model(skip_base_check=skip_base_check))

    def _build_model(self, skip_base_check: bool = False) -> nn.Module:
        recipe = self.settings.recipe
        return self._assemble_model(
            model_config=recipe.model,
            head_config=recipe.head,
            adapter_config=recipe.adapter,
            skip_base_check=skip_base_check,
        )

    def _assemble_model(
        self,
        model_config: dict[str, Any],
        head_config: dict[str, Any] | None = None,
        adapter_config: dict[str, Any] | None = None,
        skip_base_check: bool = False,
    ) -> nn.Module:
        """모델 dict로부터 5단계 조립을 수행한다.

        1. 모델 로딩 (_component_ 유무에 따라 경로 결정)
        2. MoE 감지 + moe_info 모델 인스턴스 부착
        3. Head 교체 (head_config가 있을 때)
        4. BaseModel 검증 (커스텀 모델은 BaseModel 필수)
        5. Adapter 적용 (adapter_config가 있을 때)

        QLoRA는 1+5를 한 번에 수행하는 특수 경로를 탄다.
        """
        route = _decide_model_load_route(model_config, adapter_config)

        # QLoRA 특수 경로: 양자화 + 로딩 + 어댑터가 한 번에
        if route is _ModelLoadRoute.QLORA:
            adapter_config = self._resolve_semantic_from_config(model_config, adapter_config)
            return self._build_qlora_model(model_config, adapter_config)

        # 일반 경로
        # 단계 1: pretrained 로딩
        model = self._load_pretrained(model_config, route)

        # 단계 1.5: semantic resolve (일방향 — 여기서 한 번만, 이후 raw만 흐름)
        head_config, adapter_config = self._resolve_semantic(
            model, head_config, adapter_config,
        )

        # 단계 2: MoE 감지
        if self._is_moe_model(model):
            moe_info = self._extract_moe_info(model)
            model._moe_info = moe_info
            self._cache["moe_info"] = moe_info
            logger.info(
                "MoE 모델 감지: %s experts, top-%s",
                moe_info.get("num_experts", "?"),
                moe_info.get("top_k", "?"),
            )

        # 단계 3: head 교체 (설정이 있을 때만)
        if head_config is not None:
            hc = dict(head_config)
            target_attr = hc.pop("_target_attr", None)
            head = self.resolver.resolve(hc)
            self._attach_head(model, head, target_attr)

        # 단계 4: BaseModel 검증 (커스텀 모델은 BaseModel 필수)
        # HF 모델(from_pretrained 보유)은 자체 인터페이스를 가지므로 면제.
        # 커스텀 모델은 pretrained 유무와 무관하게 BaseModel을 상속해야 한다.
        from mdp.models.base import BaseModel
        if not skip_base_check and not hasattr(model, "from_pretrained") and not isinstance(model, BaseModel):
            raise TypeError(
                f"{model.__class__.__name__}이 BaseModel을 상속하지 않습니다. "
                "커스텀 모델은 BaseModel을 상속하여 forward를 구현하고, "
                "SFT loss는 recipe.loss 또는 forward output `loss`로 제공하세요. "
                "HF 모델이라면 pretrained: hf://... 를 사용하세요."
            )

        # 단계 5: adapter 적용 (설정이 있을 때만, QLoRA는 위에서 처리)
        if adapter_config is not None:
            model = self.resolver.resolve(adapter_config, model)

        return model

    def _resolve_semantic(
        self,
        model: nn.Module,
        head_config: dict[str, Any] | None,
        adapter_config: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """semantic 키를 raw 키로 번역한다. model 생성 후, head/adapter 적용 전 1회 호출.

        변환:
        - head_config["slot"] → head_config["_target_attr"]
        - adapter_config["target"] → adapter_config["target_modules"]
        - adapter_config["save"] → adapter_config["modules_to_save"]

        semantic 키가 없으면 config를 그대로 반환 (기존 경로 호환).
        semantic 키와 대응하는 raw 키가 동시에 존재하면 ValueError.
        """
        needs_resolve = (
            (head_config is not None and "slot" in head_config)
            or (adapter_config is not None and ("target" in adapter_config or "save" in adapter_config))
        )
        if not needs_resolve:
            return head_config, adapter_config

        from mdp.models.family_routing import (
            detect_family, resolve_targets, resolve_head_slot, resolve_save_modules,
        )
        family = detect_family(model)

        if head_config is not None and "slot" in head_config:
            head_config = dict(head_config)
            if "_target_attr" in head_config:
                raise ValueError(
                    "head 설정에 'slot'과 '_target_attr'을 동시에 지정할 수 없습니다. "
                    "semantic 이름('slot')과 raw 이름('_target_attr') 중 하나만 사용하세요."
                )
            slot = head_config.pop("slot")
            head_config["_target_attr"] = resolve_head_slot(slot, family)

        if adapter_config is not None:
            ac = dict(adapter_config)
            target = ac.pop("target", None)
            if target is not None:
                if "target_modules" in ac:
                    raise ValueError(
                        "adapter 설정에 'target'과 'target_modules'를 동시에 지정할 수 없습니다. "
                        "semantic 이름('target')과 raw 이름('target_modules') 중 하나만 사용하세요."
                    )
                ac["target_modules"] = resolve_targets(target, family)
            save = ac.pop("save", None)
            if save is not None:
                if "modules_to_save" in ac:
                    raise ValueError(
                        "adapter 설정에 'save'와 'modules_to_save'를 동시에 지정할 수 없습니다. "
                        "semantic 이름('save')과 raw 이름('modules_to_save') 중 하나만 사용하세요."
                    )
                ac["modules_to_save"] = resolve_save_modules(save, family)
            adapter_config = ac

        return head_config, adapter_config

    def _resolve_semantic_from_config(
        self, model_config: dict[str, Any], adapter_config: dict[str, Any],
    ) -> dict[str, Any]:
        """QLoRA 분기 전용: model 없이 config 기반 family 추정 후 semantic resolve.

        QLoRA는 model 인스턴스가 resolve 시점에 존재하지 않으므로,
        pretrained URI에서 AutoConfig로 family를 추정한다.
        """
        ac = dict(adapter_config)
        target = ac.pop("target", None)
        save = ac.pop("save", None)

        if target is None and save is None:
            return ac  # semantic 키 없음 → 기존 경로

        from mdp.models.family_routing import (
            detect_family_from_pretrained_uri, resolve_targets, resolve_save_modules,
        )
        pretrained = model_config.get("pretrained")
        component = model_config.get("_component_")
        family = detect_family_from_pretrained_uri(pretrained, component)

        if target is not None:
            if "target_modules" in ac:
                raise ValueError(
                    "adapter 설정에 'target'과 'target_modules'를 동시에 지정할 수 없습니다. "
                    "semantic 이름('target')과 raw 이름('target_modules') 중 하나만 사용하세요."
                )
            ac["target_modules"] = resolve_targets(target, family)
        if save is not None:
            if "modules_to_save" in ac:
                raise ValueError(
                    "adapter 설정에 'save'와 'modules_to_save'를 동시에 지정할 수 없습니다. "
                    "semantic 이름('save')과 raw 이름('modules_to_save') 중 하나만 사용하세요."
                )
            ac["modules_to_save"] = resolve_save_modules(save, family)

        return ac

    def _load_pretrained(
        self,
        model_config: dict[str, Any],
        route: _ModelLoadRoute | None = None,
    ) -> nn.Module:
        """_component_와 pretrained 조합에 따라 모델을 로딩한다.

        분기 기준은 _component_ 유무이다:
        - _component_ 있음: 그 클래스가 인스턴스화를 소유한다.
          duck typing(hasattr from_pretrained)으로 HF 클래스와 커스텀 클래스를 구분.
        - _component_ 없음: PretrainedResolver가 URI에서 클래스까지 추론한다.
        """
        from mdp.models.pretrained import PretrainedResolver

        config = dict(model_config)
        component = config.pop("_component_", None)
        pretrained = config.pop("pretrained", None)
        torch_dtype_str = config.pop("torch_dtype", None)
        attn_impl = config.pop("attn_implementation", None)

        kwargs = config  # 나머지는 전부 kwargs

        if torch_dtype_str is not None:
            import torch
            kwargs["torch_dtype"] = getattr(torch, torch_dtype_str)
        if attn_impl is not None:
            kwargs["attn_implementation"] = attn_impl

        route = route or _decide_model_load_route(model_config, None)

        # _component_ 명시: 그 클래스가 인스턴스화를 소유한다
        if route in {
            _ModelLoadRoute.COMPONENT_FROM_PRETRAINED,
            _ModelLoadRoute.COMPONENT_CONSTRUCTOR,
        }:
            klass = self.resolver.import_class(
                self.resolver._resolve_alias(component)
            )
            if (
                route is _ModelLoadRoute.COMPONENT_FROM_PRETRAINED
                and hasattr(klass, "from_pretrained")
            ):
                # HF 호환 클래스: URI에서 identifier를 추출하고 from_pretrained
                _, identifier = PretrainedResolver._parse_uri(pretrained)
                return klass.from_pretrained(identifier, **kwargs)
            elif route is _ModelLoadRoute.COMPONENT_FROM_PRETRAINED:
                # 커스텀 클래스: pretrained는 생성자 인자
                return klass(pretrained=pretrained, **kwargs)
            else:
                return klass(**kwargs)

        # _component_ 없음: PretrainedResolver가 클래스까지 추론
        if route is _ModelLoadRoute.PRETRAINED_RESOLVER:
            return PretrainedResolver.load(pretrained, **kwargs)

        raise ValueError(f"지원하지 않는 모델 로딩 경로입니다: {route.value}")

    def _build_qlora_model(self, model_config: dict[str, Any], adapter_config: dict[str, Any]) -> nn.Module:
        """QLoRA 특수 경로: 양자화 + 로딩 + 어댑터를 한 번에 수행한다."""
        # pretrained URI에서 identifier 추출
        uri = model_config.get("pretrained")
        if uri is None:
            raise ValueError("QLoRA에는 pretrained 모델 URI가 필요합니다")

        # hf:// 접두사 제거
        if uri.startswith("hf://"):
            model_name = uri[len("hf://"):]
        else:
            model_name = uri

        ac = dict(adapter_config)
        # _component_ 키 제거 — apply_qlora를 직접 호출하므로
        ac.pop("_component_", None)
        ac["model_name_or_path"] = model_name
        ac["class_path"] = model_config.get("_component_", "transformers.AutoModelForCausalLM")

        # torch_dtype, attn_implementation 전달
        torch_dtype_str = model_config.get("torch_dtype")
        if torch_dtype_str is not None:
            import torch
            ac["torch_dtype"] = getattr(torch, torch_dtype_str)
        attn_impl = model_config.get("attn_implementation")
        if attn_impl is not None:
            ac["attn_implementation"] = attn_impl

        from mdp.models.adapters.qlora import apply_qlora
        return apply_qlora(**ac)

    @staticmethod
    def _attach_head(
        model: nn.Module, head: nn.Module, target_attr: str | None = None
    ) -> None:
        """모델에 새 head를 부착한다.

        Args:
            model: 기반 모델.
            head: 부착할 head 모듈.
            target_attr: head를 부착할 모델 속성명. None이면 ValueError.
        """
        if target_attr is None:
            raise ValueError(
                "head 설정에 '_target_attr'이 지정되지 않았습니다. "
                "recipe의 head 섹션에 '_target_attr'을 추가하세요."
            )
        if not hasattr(model, target_attr):
            children = [name for name, _ in model.named_children()]
            logger.warning(
                "모델에 '%s' 속성이 없어 새로 추가합니다. "
                "model의 children: %s",
                target_attr, children,
            )
        setattr(model, target_attr, head)
        logger.info("Head 교체: model.%s → %s", target_attr, type(head).__name__)

    @staticmethod
    def _is_moe_model(model: nn.Module) -> bool:
        """모델이 MoE 아키텍처인지 감지한다."""
        config = getattr(model, "config", None)
        if config is None:
            return False
        return hasattr(config, "num_local_experts") or hasattr(config, "num_experts")

    @staticmethod
    def _extract_moe_info(model: nn.Module) -> dict:
        """MoE 모델의 expert 정보를 추출한다."""
        config = model.config
        return {
            "num_experts": getattr(config, "num_local_experts", None)
            or getattr(config, "num_experts", None),
            "top_k": getattr(config, "num_experts_per_tok", None),
        }

    # ── Phase 3: 데이터 ──

    def create_dataloaders(self) -> dict:
        """train/val DataLoader를 생성한다.

        DataSpec의 dataset/collator/val_dataset을 _component_로 resolve한다.
        """
        def _create() -> dict:
            from mdp.data.dataloader import create_dataloaders
            from mdp.settings.distributed import has_distributed_intent

            recipe = self.settings.recipe
            data = recipe.data
            distributed = has_distributed_intent(self.settings)

            return create_dataloaders(
                dataset_config=data.dataset,
                collator_config=data.collator,
                dataloader_config=data.dataloader.model_dump(),
                val_dataset_config=data.val_dataset,
                sampler_config=data.sampler,
                distributed=distributed,
            )

        return self._get_or_create("dataloaders", _create)

    # ── RL: 복수 모델 생성 ──

    def create_models(self, skip_base_check: bool = False) -> dict[str, nn.Module]:
        """RL Recipe의 models 설정에 따라 역할별 모델을 생성한다.

        각 모델은 SFT와 동일한 5단계 조립을 거친다:
        pretrained 로딩 → MoE 감지 → head 교체 → BaseModel 검증 → adapter 적용.
        """
        def _create() -> dict[str, nn.Module]:
            recipe = self.settings.recipe
            if recipe.rl is None:
                raise ValueError("create_models()는 recipe.rl이 필요합니다")

            models = {}
            for name, spec in recipe.rl.models.items():
                spec = dict(spec)  # 원본 변이 방지
                head_config = spec.pop("head", None)
                adapter_config = spec.pop("adapter", None)
                # optimizer, scheduler는 RLTrainer가 처리 — Factory는 모델만
                spec.pop("optimizer", None)
                spec.pop("scheduler", None)
                spec.pop("freeze", None)
                model = self._assemble_model(
                    model_config=spec,
                    head_config=head_config,
                    adapter_config=adapter_config,
                    skip_base_check=skip_base_check,
                )
                models[name] = model
            return models

        return self._get_or_create("models", _create)
