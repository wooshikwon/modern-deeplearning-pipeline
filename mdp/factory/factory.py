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

    def _get_or_create(self, key: str, creator: Callable) -> Any:
        """캐시에서 key를 찾고, 없으면 creator()를 호출하여 캐싱한다."""
        if key not in self._cache:
            instance = creator()
            if instance is not None:
                self._cache[key] = instance
            return instance
        return self._cache[key]

    # ── Phase 2: 모델 생성 ──

    def create_model(self) -> nn.Module:
        """Recipe의 model 설정에 따라 모델을 생성한다.

        순서: pretrained 로딩 → head 교체 → adapter 적용.
        QLoRA는 양자화+로딩+어댑터가 결합된 특수 경로를 탄다.
        """
        return self._get_or_create("model", self._build_model)

    def _build_model(self) -> nn.Module:
        recipe = self.settings.recipe
        model_spec = recipe.model
        adapter_spec = recipe.adapter

        # QLoRA 특수 경로: 양자화 + 로딩 + 어댑터가 한 번에
        if adapter_spec is not None and adapter_spec.method == "qlora":
            return self._build_qlora_model(model_spec, adapter_spec)

        # 일반 경로
        # 단계 1: pretrained 로딩
        model = self._load_pretrained(model_spec)

        # 단계 2: head 교체 (설정이 있을 때만)
        if recipe.head is not None:
            head_config = dict(recipe.head)
            target_attr = head_config.pop("_target_attr", None)
            head = self.resolver.resolve(head_config)
            self._attach_head(model, head, target_attr)

        # 단계 3: adapter 적용 (설정이 있을 때만)
        if adapter_spec is not None and adapter_spec.method == "lora":
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
            raise AttributeError(
                f"모델에 '{target_attr}' 속성이 없습니다. "
                f"model의 children: {children}."
            )
        setattr(model, target_attr, head)
        logger.info("Head 교체: model.%s → %s", target_attr, type(head).__name__)

    # ── Phase 3: 데이터 ──

    def create_dataloaders(self) -> dict:
        """train/val DataLoader를 생성한다."""
        def _create() -> dict:
            from mdp.data.dataloader import create_dataloaders

            recipe = self.settings.recipe
            distributed = self.settings.config.compute.distributed is not None
            return create_dataloaders(
                data_spec=recipe.data,
                fields=recipe.data.fields,
                distributed=distributed,
            )

        return self._get_or_create("dataloaders", _create)

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

    def create_callbacks(self) -> list:
        """Recipe의 callbacks 설정에서 콜백 리스트를 생성한다."""
        callbacks = []
        for cfg in self.settings.recipe.callbacks:
            try:
                cb = self.resolver.resolve(cfg)
                callbacks.append(cb)
            except Exception as e:
                logger.warning("콜백 생성 실패: %s", e)
        return callbacks

