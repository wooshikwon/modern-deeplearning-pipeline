"""SettingsPlan to AssemblyPlan translation."""

from __future__ import annotations

from typing import Any, Mapping

from mdp.factory.assembly_plan import AssemblyPlan
from mdp.factory.specs import (
    CallbackNode,
    ComponentSpec,
    DataNode,
    ModelNode,
    StrategyNode,
    TrainerNode,
)
from mdp.settings.plan import SettingsPlan


class AssemblyPlanner:
    """Build serializable factory assembly graphs from SettingsPlan."""

    _MODEL_OWNED_KEYS = frozenset(
        {"head", "adapter", "optimizer", "scheduler", "loss", "freeze", "trainable"}
    )
    _STRATEGY_CAPABILITIES = (
        "setup",
        "setup_models",
        "checkpoint",
        "custom_method_dispatch",
        "cleanup",
    )

    @classmethod
    def from_settings_plan(cls, plan: SettingsPlan) -> AssemblyPlan:
        """Translate a validated SettingsPlan into an SFT or RL AssemblyPlan."""
        if plan.mode == "rl" or plan.settings.recipe.rl is not None:
            return cls()._build_rl(plan)
        if plan.mode == "sft":
            return cls()._build_sft(plan)
        raise ValueError(
            "AssemblyPlanner는 training SettingsPlan만 지원합니다 "
            f"(mode={plan.mode!r})"
        )

    def _build_sft(self, plan: SettingsPlan) -> AssemblyPlan:
        recipe = plan.settings.recipe
        model = ModelNode(
            role="policy",
            trainable=True,
            model=self._component(recipe.model),
            head=self._optional_component(recipe.head),
            adapter=self._optional_component(recipe.adapter),
            optimizer=self._optional_component(recipe.optimizer),
            scheduler=self._optional_component(recipe.scheduler),
            loss=self._optional_component(recipe.loss),
        )
        return AssemblyPlan(
            kind="sft_training",
            settings_plan=plan,
            models=(model,),
            data=self._data_node(plan),
            trainer=TrainerNode(
                kind="sft",
                training_config=recipe.training.model_dump(),
                generation_config=(
                    recipe.generation.model_dump() if recipe.generation is not None else None
                ),
                evaluation_config=recipe.evaluation.model_dump(),
                monitoring_config=recipe.monitoring.model_dump(),
            ),
            strategy=self._strategy_node(plan),
            callbacks=self._callback_nodes(plan),
        )

    def _build_rl(self, plan: SettingsPlan) -> AssemblyPlan:
        recipe = plan.settings.recipe
        if recipe.rl is None:
            raise ValueError("RL AssemblyPlan에는 recipe.rl이 필요합니다")

        models = tuple(
            self._rl_model_node(role, spec)
            for role, spec in recipe.rl.models.items()
        )
        return AssemblyPlan(
            kind="rl_training",
            settings_plan=plan,
            models=models,
            data=self._data_node(plan),
            trainer=TrainerNode(
                kind="rl",
                training_config=recipe.training.model_dump(),
                algorithm=self._component(recipe.rl.algorithm),
                generation_config=(
                    recipe.rl.generation.model_dump()
                    if recipe.rl.generation is not None
                    else None
                ),
                evaluation_config=recipe.evaluation.model_dump(),
                monitoring_config=recipe.monitoring.model_dump(),
            ),
            strategy=self._strategy_node(plan),
            callbacks=self._callback_nodes(plan),
        )

    def _rl_model_node(self, role: str, spec: Mapping[str, Any]) -> ModelNode:
        model_config = {
            key: value for key, value in spec.items()
            if key not in self._MODEL_OWNED_KEYS
        }
        return ModelNode(
            role=role,
            trainable=self._is_rl_model_trainable(role, spec),
            model=self._component(model_config),
            head=self._optional_component(spec.get("head")),
            adapter=self._optional_component(spec.get("adapter")),
            optimizer=self._optional_component(spec.get("optimizer")),
            scheduler=self._optional_component(spec.get("scheduler")),
            loss=self._optional_component(spec.get("loss")),
        )

    @staticmethod
    def _is_rl_model_trainable(role: str, spec: Mapping[str, Any]) -> bool:
        if "trainable" in spec:
            return bool(spec["trainable"])
        if bool(spec.get("freeze", False)):
            return False
        if spec.get("optimizer") is not None:
            return True
        return role == "policy"

    def _data_node(self, plan: SettingsPlan) -> DataNode:
        data = plan.settings.recipe.data
        return DataNode(
            dataset=self._component(data.dataset),
            val_dataset=self._optional_component(data.val_dataset),
            collator=self._component(data.collator),
            sampler=self._optional_component(data.sampler),
            dataloader_config=data.dataloader.model_dump(),
            distributed_intent=plan.distributed_intent,
        )

    def _strategy_node(self, plan: SettingsPlan) -> StrategyNode | None:
        distributed = plan.settings.config.compute.distributed
        if distributed is None or not plan.distributed_intent:
            return None

        config = dict(distributed)
        strategy = config.get("strategy", "auto")
        return StrategyNode(
            config=config,
            strategy=strategy,
            capability_boundary=self._STRATEGY_CAPABILITIES,
            distributed_intent=plan.distributed_intent,
        )

    def _callback_nodes(self, plan: SettingsPlan) -> tuple[CallbackNode, ...]:
        return tuple(
            CallbackNode(config=self._component(config))
            for config in plan.callback_configs
        )

    @staticmethod
    def _component(config: Mapping[str, Any]) -> ComponentSpec:
        return ComponentSpec.from_config(config)  # type: ignore[return-value]

    @staticmethod
    def _optional_component(config: Mapping[str, Any] | None) -> ComponentSpec | None:
        return ComponentSpec.from_config(config)
