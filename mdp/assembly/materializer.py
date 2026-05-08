"""AssemblyPlan materialization into concrete runtime components."""

from __future__ import annotations

import logging
from dataclasses import replace
from enum import Enum
from typing import Any

import torch.nn as nn

from mdp.assembly.assembly_plan import AssemblyPlan
from mdp.assembly.bundles import (
    RLTrainingBundle,
    SFTTrainingBundle,
    build_rl_training_bundle,
    build_sft_training_bundle,
)
from mdp.assembly.specs import ComponentSpec, DataNode, ModelNode
from mdp.settings.resolver import ComponentResolver
from mdp.training.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)


class _ModelLoadRoute(Enum):
    COMPONENT_FROM_PRETRAINED = "component_from_pretrained"
    COMPONENT_CONSTRUCTOR = "component_constructor"
    PRETRAINED_RESOLVER = "pretrained_resolver"
    QLORA = "qlora"


def _decide_model_load_route(
    model_config: ComponentSpec,
    adapter_config: ComponentSpec | None,
) -> _ModelLoadRoute:
    component = model_config.resolved_component or model_config.component
    pretrained = model_config.pretrained

    if adapter_config is not None and _is_qlora_adapter_config(adapter_config):
        return _ModelLoadRoute.QLORA
    if component is not None and pretrained is not None:
        return _ModelLoadRoute.COMPONENT_FROM_PRETRAINED
    if component is not None:
        return _ModelLoadRoute.COMPONENT_CONSTRUCTOR
    if pretrained is not None:
        return _ModelLoadRoute.PRETRAINED_RESOLVER
    raise ValueError("model에 component 또는 pretrained가 필요합니다")


def _is_qlora_adapter_config(adapter_config: ComponentSpec) -> bool:
    component = adapter_config.resolved_component or adapter_config.component or ""
    return component in {"QLoRA", "mdp.models.adapters.qlora.apply_qlora"}


class AssemblyMaterializer:
    """Materialize model/data nodes from an AssemblyPlan."""

    _SENTINEL = object()

    def __init__(
        self,
        assembly_plan: AssemblyPlan | None = None,
        *,
        resolver: ComponentResolver | None = None,
        cache: dict[str, Any] | None = None,
    ) -> None:
        self.assembly_plan = assembly_plan
        self.resolver = resolver or ComponentResolver()
        self._cache = cache if cache is not None else {}

    @property
    def settings(self):
        if self.assembly_plan is None:
            raise ValueError("AssemblyMaterializer에는 AssemblyPlan이 필요합니다")
        return self.assembly_plan.run_plan.settings

    def _get_or_create(self, key: str, creator):
        cached = self._cache.get(key, self._SENTINEL)
        if cached is not self._SENTINEL:
            return cached
        instance = creator()
        if instance is not None:
            self._cache[key] = instance
        return instance

    def materialize_policy_model(self, skip_base_check: bool = False) -> nn.Module:
        """Materialize the SFT policy node while preserving cache semantics."""
        def _create() -> nn.Module:
            policy = self._model_node("policy")
            return self.materialize_model_node(policy, skip_base_check=skip_base_check)

        return self._get_or_create("model", _create)

    def materialize_models(self, skip_base_check: bool = False) -> dict[str, nn.Module]:
        """Materialize RL role models with the legacy role->model output shape."""
        def _create() -> dict[str, nn.Module]:
            if self.assembly_plan is None or self.assembly_plan.kind != "rl_training":
                raise ValueError("create_models()는 recipe.rl이 필요합니다")
            return {
                node.role: self.materialize_model_node(
                    node, skip_base_check=skip_base_check,
                )
                for node in self.assembly_plan.models
            }

        return self._get_or_create("models", _create)

    def materialize_dataloaders(self) -> dict:
        """Materialize train/val dataloaders from the plan DataNode."""
        return self._get_or_create(
            "dataloaders",
            lambda: self.materialize_data_node(self._data_node()),
        )

    def materialize_callbacks(self) -> list[BaseCallback]:
        """Materialize callback nodes from the plan's raw callback configs."""
        def _create() -> list[BaseCallback]:
            if self.assembly_plan is None:
                raise ValueError("AssemblyMaterializer에는 AssemblyPlan이 필요합니다")
            from mdp.training._common import create_callbacks

            configs = [
                node.config.to_dict()
                for node in self.assembly_plan.callbacks
            ]
            return create_callbacks(configs, self.resolver)

        return self._get_or_create("callbacks", _create)

    def materialize_sft_training_bundle(
        self,
        *,
        callbacks: list[BaseCallback] | None = None,
        skip_base_check: bool = False,
    ) -> SFTTrainingBundle:
        """Materialize an SFTTrainingBundle for Trainer.from_bundle()."""
        if self.assembly_plan is None or self.assembly_plan.kind != "sft_training":
            raise ValueError("SFTTrainingBundle에는 sft_training AssemblyPlan이 필요합니다")
        dataloaders = self.materialize_dataloaders()
        callback_instances = callbacks
        if callback_instances is None:
            callback_instances = self.materialize_callbacks()
        return build_sft_training_bundle(
            settings=self.settings,
            model=self.materialize_policy_model(skip_base_check=skip_base_check),
            train_loader=dataloaders["train"],
            val_loader=dataloaders.get("val"),
            callbacks=callback_instances,
            resolver=self.resolver,
        )

    def materialize_rl_training_bundle(
        self,
        *,
        callbacks: list[BaseCallback] | None = None,
        skip_base_check: bool = False,
    ) -> RLTrainingBundle:
        """Materialize an RLTrainingBundle for RLTrainer.from_bundle()."""
        if self.assembly_plan is None or self.assembly_plan.kind != "rl_training":
            raise ValueError("RLTrainingBundle에는 rl_training AssemblyPlan이 필요합니다")
        dataloaders = self.materialize_dataloaders()
        callback_instances = callbacks
        if callback_instances is None:
            callback_instances = self.materialize_callbacks()
        return build_rl_training_bundle(
            settings=self.settings,
            models=self.materialize_models(skip_base_check=skip_base_check),
            train_loader=dataloaders["train"],
            val_loader=dataloaders.get("val"),
            callbacks=callback_instances,
            resolver=self.resolver,
        )

    def materialize_model_node(
        self,
        node: ModelNode,
        *,
        skip_base_check: bool = False,
    ) -> nn.Module:
        """Materialize a single ModelNode."""
        return self._assemble_model(
            model_config=node.model,
            head_config=node.head,
            adapter_config=node.adapter,
            skip_base_check=skip_base_check,
        )

    def materialize_data_node(self, node: DataNode) -> dict:
        """Materialize dataloaders from a DataNode."""
        from mdp.data.dataloader import create_dataloaders

        return create_dataloaders(
            dataset_config=node.dataset,
            collator_config=node.collator,
            dataloader_config=dict(node.dataloader_config),
            val_dataset_config=node.val_dataset,
            sampler_config=node.sampler,
            distributed=node.distributed_intent,
        )

    def _model_node(self, role: str) -> ModelNode:
        if self.assembly_plan is None:
            raise ValueError("AssemblyMaterializer에는 AssemblyPlan이 필요합니다")
        for node in self.assembly_plan.models:
            if node.role == role:
                return node
        raise ValueError(f"AssemblyPlan에 {role!r} model node가 없습니다")

    def _data_node(self) -> DataNode:
        if self.assembly_plan is None:
            raise ValueError("AssemblyMaterializer에는 AssemblyPlan이 필요합니다")
        return self.assembly_plan.data

    def _assemble_model(
        self,
        model_config: ComponentSpec,
        head_config: ComponentSpec | None = None,
        adapter_config: ComponentSpec | None = None,
        skip_base_check: bool = False,
    ) -> nn.Module:
        """Assemble a model from model/head/adapter component specs."""
        route = _decide_model_load_route(model_config, adapter_config)

        if route is _ModelLoadRoute.QLORA:
            adapter_config = self._resolve_semantic_from_config(
                model_config, adapter_config,
            )
            return self._build_qlora_model(model_config, adapter_config)

        model = self._load_pretrained(model_config, route)

        head_config, adapter_config = self._resolve_semantic(
            model, head_config, adapter_config,
        )

        if self._is_moe_model(model):
            moe_info = self._extract_moe_info(model)
            model._moe_info = moe_info
            self._cache["moe_info"] = moe_info
            logger.info(
                "MoE 모델 감지: %s experts, top-%s",
                moe_info.get("num_experts", "?"),
                moe_info.get("top_k", "?"),
        )

        if head_config is not None:
            target_attr = head_config.kwargs.get("_target_attr")
            head_kwargs = {
                key: value
                for key, value in head_config.kwargs.items()
                if key != "_target_attr"
            }
            head = self.resolver.resolve(replace(head_config, kwargs=head_kwargs))
            self._attach_head(model, head, target_attr)

        from mdp.models.base import BaseModel
        if (
            not skip_base_check
            and not hasattr(model, "from_pretrained")
            and not isinstance(model, BaseModel)
        ):
            raise TypeError(
                f"{model.__class__.__name__}이 BaseModel을 상속하지 않습니다. "
                "커스텀 모델은 BaseModel을 상속하여 forward를 구현하고, "
                "SFT loss는 recipe.loss 또는 forward output `loss`로 제공하세요. "
                "HF 모델이라면 pretrained: hf://... 를 사용하세요."
            )

        if adapter_config is not None:
            model = self.resolver.resolve(adapter_config, model)

        return model

    def _resolve_semantic(
        self,
        model: nn.Module,
        head_config: ComponentSpec | None,
        adapter_config: ComponentSpec | None,
    ) -> tuple[ComponentSpec | None, ComponentSpec | None]:
        needs_resolve = (
            (head_config is not None and "slot" in head_config.kwargs)
            or (
                adapter_config is not None
                and (
                    "target" in adapter_config.kwargs
                    or "save" in adapter_config.kwargs
                )
            )
        )
        if not needs_resolve:
            return head_config, adapter_config

        from mdp.models.family_routing import (
            detect_family, resolve_targets, resolve_head_slot, resolve_save_modules,
        )
        family = detect_family(model)

        if head_config is not None and "slot" in head_config.kwargs:
            head_kwargs = dict(head_config.kwargs)
            if "_target_attr" in head_kwargs:
                raise ValueError(
                    "head 설정에 'slot'과 '_target_attr'을 동시에 지정할 수 없습니다. "
                    "semantic 이름('slot')과 raw 이름('_target_attr') 중 하나만 사용하세요."
                )
            slot = head_kwargs.pop("slot")
            head_kwargs["_target_attr"] = resolve_head_slot(slot, family)
            head_config = replace(head_config, kwargs=head_kwargs)

        if adapter_config is not None:
            ac = dict(adapter_config.kwargs)
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
            adapter_config = replace(adapter_config, kwargs=ac)

        return head_config, adapter_config

    def _resolve_semantic_from_config(
        self, model_config: ComponentSpec, adapter_config: ComponentSpec,
    ) -> ComponentSpec:
        ac = dict(adapter_config.kwargs)
        target = ac.pop("target", None)
        save = ac.pop("save", None)

        if target is None and save is None:
            return replace(adapter_config, kwargs=ac)

        from mdp.models.family_routing import (
            detect_family_from_pretrained_uri, resolve_targets, resolve_save_modules,
        )
        pretrained = model_config.pretrained
        component = model_config.resolved_component or model_config.component
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

        return replace(adapter_config, kwargs=ac)

    def _load_pretrained(
        self,
        model_config: ComponentSpec,
        route: _ModelLoadRoute | None = None,
    ) -> nn.Module:
        from mdp.models.pretrained import PretrainedResolver

        component = model_config.resolved_component or model_config.component
        pretrained = model_config.pretrained
        kwargs = dict(model_config.kwargs)
        torch_dtype_str = kwargs.pop("torch_dtype", None)
        attn_impl = kwargs.pop("attn_implementation", None)

        if torch_dtype_str is not None:
            import torch
            kwargs["torch_dtype"] = getattr(torch, torch_dtype_str)
        if attn_impl is not None:
            kwargs["attn_implementation"] = attn_impl

        route = route or _decide_model_load_route(model_config, None)

        if route in {
            _ModelLoadRoute.COMPONENT_FROM_PRETRAINED,
            _ModelLoadRoute.COMPONENT_CONSTRUCTOR,
        }:
            if component is None:
                raise ValueError("component model route requires component")
            klass = self.resolver.import_class(self.resolver._resolve_alias(component))
            if (
                route is _ModelLoadRoute.COMPONENT_FROM_PRETRAINED
                and hasattr(klass, "from_pretrained")
            ):
                _, identifier = PretrainedResolver._parse_uri(pretrained)
                return klass.from_pretrained(identifier, **kwargs)
            if route is _ModelLoadRoute.COMPONENT_FROM_PRETRAINED:
                return klass(pretrained=pretrained, **kwargs)
            return klass(**kwargs)

        if route is _ModelLoadRoute.PRETRAINED_RESOLVER:
            return PretrainedResolver.load(pretrained, **kwargs)

        raise ValueError(f"지원하지 않는 모델 로딩 경로입니다: {route.value}")

    def _build_qlora_model(
        self, model_config: ComponentSpec, adapter_config: ComponentSpec
    ) -> nn.Module:
        uri = model_config.pretrained
        if uri is None:
            raise ValueError("QLoRA에는 pretrained 모델 URI가 필요합니다")

        if uri.startswith("hf://"):
            model_name = uri[len("hf://"):]
        else:
            model_name = uri

        ac = dict(adapter_config.kwargs)
        ac["model_name_or_path"] = model_name
        ac["class_path"] = (
            model_config.resolved_component
            or model_config.component
            or "transformers.AutoModelForCausalLM"
        )

        torch_dtype_str = model_config.kwargs.get("torch_dtype")
        if torch_dtype_str is not None:
            import torch
            ac["torch_dtype"] = getattr(torch, torch_dtype_str)
        attn_impl = model_config.kwargs.get("attn_implementation")
        if attn_impl is not None:
            ac["attn_implementation"] = attn_impl

        from mdp.models.adapters.qlora import apply_qlora
        return apply_qlora(**ac)

    @staticmethod
    def _attach_head(
        model: nn.Module, head: nn.Module, target_attr: str | None = None
    ) -> None:
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
        config = getattr(model, "config", None)
        if config is None:
            return False
        return hasattr(config, "num_local_experts") or hasattr(config, "num_experts")

    @staticmethod
    def _extract_moe_info(model: nn.Module) -> dict:
        config = model.config
        return {
            "num_experts": getattr(config, "num_local_experts", None)
            or getattr(config, "num_experts", None),
            "top_k": getattr(config, "num_experts_per_tok", None),
        }
