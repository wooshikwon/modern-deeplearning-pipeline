"""Factory 파사드 — Settings를 받아 모든 컴포넌트를 생성하는 중앙 관리자."""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch.nn as nn

from mdp.settings.resolver import ComponentResolver
from mdp.settings.schema import Settings

logger = logging.getLogger(__name__)

# head를 부착할 수 있는 일반적인 속성명 (우선순위 순)
_HEAD_ATTR_CANDIDATES = [
    "classifier",  # HuggingFace 분류 모델
    "lm_head",     # HuggingFace 언어 모델
    "head",        # timm 모델
    "fc",          # torchvision ResNet 등
    "heads",       # 일부 검출 모델
]


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
            head = self.resolver.resolve(recipe.head)
            self._attach_head(model, head)

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
            klass = self.resolver._import_class(model_spec.class_path)
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

        # torch_dtype, attn_implementation 전달
        if model_spec.torch_dtype is not None:
            import torch
            adapter_config["torch_dtype"] = getattr(torch, model_spec.torch_dtype)
        if model_spec.attn_implementation is not None:
            adapter_config["attn_implementation"] = model_spec.attn_implementation

        return apply_adapter(None, adapter_config)

    @staticmethod
    def _attach_head(model: nn.Module, head: nn.Module) -> None:
        """모델에 새 head를 부착한다.

        모델 아키텍처별로 head가 붙는 위치가 다르므로,
        일반적인 속성명을 순회하며 교체할 위치를 찾는다.
        """
        for attr in _HEAD_ATTR_CANDIDATES:
            if hasattr(model, attr):
                setattr(model, attr, head)
                logger.info("Head 교체: model.%s → %s", attr, type(head).__name__)
                return

        # 후보에 없는 경우 — 모델의 named_children을 출력하여 디버깅 지원
        children = [name for name, _ in model.named_children()]
        raise AttributeError(
            f"모델에서 head를 교체할 위치를 찾을 수 없습니다. "
            f"model의 children: {children}. "
            f"모델 클래스에 맞는 head 속성명을 확인하세요."
        )

    def create_head(self) -> nn.Module | None:
        """Recipe의 head 설정에서 head를 생성한다."""
        head_config = self.settings.recipe.head
        if head_config is None:
            return None
        return self._get_or_create(
            "head", lambda: self.resolver.resolve(head_config)
        )

    # ── Phase 3: 데이터 ──

    def create_dataloaders(self) -> dict:
        """train/val DataLoader를 생성한다."""
        def _create() -> dict:
            from mdp.data.dataloader import create_dataloaders

            recipe = self.settings.recipe
            distributed = self.settings.config.compute.distributed is not None
            return create_dataloaders(
                dataset_config=recipe.data.dataset,
                aug_config=recipe.data.augmentation,
                tokenizer_config=recipe.data.tokenizer,
                loader_config=recipe.data.dataloader.model_dump(),
                recipe_task=recipe.task,
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

    def create_strategy(self) -> Any:
        """Config의 distributed 설정에서 전략을 생성한다."""
        dist = self.settings.config.compute.distributed
        if dist is None:
            return None
        strategy_name = dist.get("strategy") if isinstance(dist, dict) else None
        if strategy_name is None:
            return None

        from mdp.training.trainer import STRATEGY_MAP

        class_path = STRATEGY_MAP.get(strategy_name)
        if class_path is None:
            raise ValueError(f"알 수 없는 분산 전략: {strategy_name}")

        kwargs = {k: v for k, v in dist.items() if k != "strategy"}
        return self.resolver.resolve({"_component_": class_path, **kwargs})

    # ── Phase 5: 실행 레이어 ──

    def create_executor(self) -> Any:
        """config.compute.target에 맞는 Executor를 생성한다."""
        def _create() -> Any:
            from mdp.compute import get_executor

            target = self.settings.config.compute.target
            return get_executor(target)

        return self._get_or_create("executor", _create)
