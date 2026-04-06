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
            model_spec=recipe.model,
            head_config=recipe.head,
            adapter_spec=recipe.adapter,
            skip_base_check=skip_base_check,
        )

    def _assemble_model(
        self,
        model_spec: Any,
        head_config: dict[str, Any] | None = None,
        adapter_spec: Any | None = None,
        skip_base_check: bool = False,
    ) -> nn.Module:
        """모델 스펙으로부터 5단계 조립을 수행한다.

        1. Pretrained 로딩 (또는 class_path 인스턴스화)
        2. MoE 감지 + moe_info 모델 인스턴스 부착
        3. Head 교체 (head_config가 있을 때)
        4. BaseModel 검증 (pretrained=None인 커스텀 모델)
        5. Adapter 적용 (adapter_spec가 있을 때)

        QLoRA는 1+5를 한 번에 수행하는 특수 경로를 탄다.
        """
        # QLoRA 특수 경로: 양자화 + 로딩 + 어댑터가 한 번에
        if adapter_spec is not None and adapter_spec.method == "qlora":
            return self._build_qlora_model(model_spec, adapter_spec)

        # 일반 경로
        # 단계 1: pretrained 로딩
        model = self._load_pretrained(model_spec)

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

        # 단계 4: BaseModel 검증 (pretrained=None인 커스텀 모델)
        from mdp.models.base import BaseModel
        if not skip_base_check and model_spec.pretrained is None and not isinstance(model, BaseModel):
            raise TypeError(
                f"{model.__class__.__name__}이 BaseModel을 상속하지 않습니다. "
                "커스텀 모델은 BaseModel을 상속하여 forward, training_step, "
                "validation_step을 구현하세요. HF 모델이라면 pretrained: hf://... 를 사용하세요."
            )

        # 단계 5: adapter 적용 (설정이 있을 때만, QLoRA는 위에서 처리)
        if adapter_spec is not None and adapter_spec.method != "qlora":
            from mdp.models.adapters import apply_adapter

            adapter_config = adapter_spec.model_dump(exclude_none=True)
            model = apply_adapter(model, adapter_config)

        return model

    def _load_pretrained(self, model_spec: Any) -> nn.Module:
        """PretrainedResolver를 통해 pretrained 모델을 로딩한다."""
        from mdp.models.pretrained import PretrainedResolver

        kwargs = dict(model_spec.init_args)

        # torch_dtype 처리
        if model_spec.torch_dtype is not None:
            import torch
            kwargs["torch_dtype"] = getattr(torch, model_spec.torch_dtype)

        # attn_implementation 처리
        if model_spec.attn_implementation is not None:
            kwargs["attn_implementation"] = model_spec.attn_implementation

        if model_spec.pretrained is not None:
            return PretrainedResolver.load(
                model_spec.pretrained,
                class_path=model_spec.class_path,
                **kwargs,
            )
        else:
            # pretrained 없음 → class_path에서 직접 인스턴스화 (랜덤 초기화)
            klass = self.resolver.import_class(model_spec.class_path)
            return klass(**kwargs)

    def _build_qlora_model(self, model_spec: Any, adapter_spec: Any) -> nn.Module:
        """QLoRA 특수 경로: 양자화 + 로딩 + 어댑터를 한 번에 수행한다."""
        from mdp.models.adapters import apply_adapter

        # pretrained URI에서 identifier 추출
        uri = model_spec.pretrained
        if uri is None:
            raise ValueError("QLoRA에는 pretrained 모델 URI가 필요합니다")

        # hf:// 접두사 제거
        if uri.startswith("hf://"):
            model_name = uri[len("hf://"):]
        else:
            model_name = uri

        adapter_config = adapter_spec.model_dump(exclude_none=True)
        adapter_config["model_name_or_path"] = model_name
        adapter_config["class_path"] = model_spec.class_path

        # torch_dtype, attn_implementation 전달
        if model_spec.torch_dtype is not None:
            import torch
            adapter_config["torch_dtype"] = getattr(torch, model_spec.torch_dtype)
        if model_spec.attn_implementation is not None:
            adapter_config["attn_implementation"] = model_spec.attn_implementation

        return apply_adapter(None, adapter_config)

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

        DataSpec을 분해하여 create_dataloaders에 전달한다.
        """
        def _create() -> dict:
            from mdp.data.dataloader import create_dataloaders

            recipe = self.settings.recipe
            data = recipe.data
            distributed = self.settings.config.compute.distributed is not None

            return create_dataloaders(
                source=data.source,
                fields=data.fields or None,
                split=data.split,
                subset=data.subset,
                streaming=data.streaming,
                data_files=data.data_files,
                fmt=data.format,
                label_strategy=data.label_strategy,
                aug_config=data.augmentation,
                tokenizer_config=data.tokenizer,
                loader_config=data.dataloader.model_dump(),
                distributed=distributed,
                val_split=data.val_split,
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
                model = self._assemble_model(
                    model_spec=spec,
                    head_config=getattr(spec, "head", None),
                    adapter_spec=spec.adapter,
                    skip_base_check=skip_base_check,
                )
                models[name] = model
            return models

        return self._get_or_create("models", _create)

    # ── Phase 4: 학습 ──

    def create_trainer(self) -> Any:
        """Trainer를 생성한다."""
        def _create() -> Any:
            from mdp.training.trainer import Trainer

            model = self.create_model()
            loaders = self.create_dataloaders()
            return Trainer(
                settings=self.settings,
                model=model,
                train_loader=loaders["train"],
                val_loader=loaders.get("val"),
            )

        return self._get_or_create("trainer", _create)


