"""Factory 파사드 — Settings를 받아 모든 컴포넌트를 생성하는 중앙 관리자."""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch.nn as nn

from mdp.settings.resolver import ComponentResolver
from mdp.settings.schema import Settings

logger = logging.getLogger(__name__)


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

        1. Pretrained 로딩 (또는 _component_ 인스턴스화)
        2. MoE 감지 + moe_info 모델 인스턴스 부착
        3. Head 교체 (head_config가 있을 때)
        4. BaseModel 검증 (pretrained가 없는 커스텀 모델)
        5. Adapter 적용 (adapter_config가 있을 때)

        QLoRA는 1+5를 한 번에 수행하는 특수 경로를 탄다.
        """
        # QLoRA 특수 경로: 양자화 + 로딩 + 어댑터가 한 번에
        if adapter_config is not None and self._is_qlora_adapter(adapter_config):
            return self._build_qlora_model(model_config, adapter_config)

        # 일반 경로
        # 단계 1: pretrained 로딩
        model = self._load_pretrained(model_config)

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

        # 단계 4: BaseModel 검증 (pretrained가 없는 커스텀 모델)
        from mdp.models.base import BaseModel
        if not skip_base_check and model_config.get("pretrained") is None and not isinstance(model, BaseModel):
            raise TypeError(
                f"{model.__class__.__name__}이 BaseModel을 상속하지 않습니다. "
                "커스텀 모델은 BaseModel을 상속하여 forward, training_step, "
                "validation_step을 구현하세요. HF 모델이라면 pretrained: hf://... 를 사용하세요."
            )

        # 단계 5: adapter 적용 (설정이 있을 때만, QLoRA는 위에서 처리)
        if adapter_config is not None:
            model = self.resolver.resolve(adapter_config, model)

        return model

    def _load_pretrained(self, model_config: dict[str, Any]) -> nn.Module:
        """PretrainedResolver를 통해 pretrained 모델을 로딩한다."""
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

        if pretrained is not None:
            return PretrainedResolver.load(
                pretrained,
                class_path=component,
                **kwargs,
            )
        elif component is not None:
            klass = self.resolver.import_class(
                self.resolver._resolve_alias(component)
            )
            return klass(**kwargs)
        else:
            raise ValueError("model에 _component_ 또는 pretrained가 필요합니다")

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

    def _is_qlora_adapter(self, adapter_config: dict[str, Any]) -> bool:
        """adapter 설정이 QLoRA인지 판별한다."""
        component = adapter_config.get("_component_", "")
        resolved = self.resolver._resolve_alias(component) if component else ""
        return resolved == "mdp.models.adapters.qlora.apply_qlora"

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

            recipe = self.settings.recipe
            data = recipe.data
            distributed = self.settings.config.compute.distributed is not None

            return create_dataloaders(
                dataset_config=data.dataset,
                collator_config=data.collator,
                dataloader_config=data.dataloader.model_dump(),
                val_dataset_config=data.val_dataset,
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
