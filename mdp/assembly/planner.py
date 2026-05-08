"""RunPlan to AssemblyPlan translation."""

from __future__ import annotations

from mdp.assembly.assembly_plan import AssemblyPlan
from mdp.assembly.specs import (
    CallbackNode,
    ComponentSpec,
    DataNode,
    ModelNode,
    StrategyNode,
    TrainerNode,
)
from mdp.settings.components import ComponentSpec as SettingsComponentSpec
from mdp.settings.components import ModelComponentSpec, RoleModelSpec
from mdp.settings.run_plan import RunPlan
from mdp.settings.resolver import ComponentResolver


class AssemblyPlanner:
    """Build serializable component assembly graphs from RunPlan."""

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
    def from_run_plan(cls, plan: RunPlan) -> AssemblyPlan:
        """Translate a validated RunPlan into an SFT or RL AssemblyPlan."""
        if plan.mode == "rl" or plan.settings.recipe.rl is not None:
            return cls()._build_rl(plan)
        if plan.mode in {"sft", "inference", "serving", "export"}:
            return cls()._build_sft(plan)
        raise ValueError(
            "AssemblyPlanner는 training RunPlan만 지원합니다 "
            f"(mode={plan.mode!r})"
        )

    def _build_sft(self, plan: RunPlan) -> AssemblyPlan:
        recipe = plan.settings.recipe
        model = ModelNode(
            role="policy",
            trainable=True,
            model=self._model_component(recipe.model, path="recipe.model"),
            head=self._optional_component(recipe.head, path="recipe.head"),
            adapter=self._optional_component(recipe.adapter, path="recipe.adapter"),
            optimizer=self._optional_component(recipe.optimizer, path="recipe.optimizer"),
            scheduler=self._optional_component(recipe.scheduler, path="recipe.scheduler"),
            loss=self._optional_component(recipe.loss, path="recipe.loss"),
        )
        return AssemblyPlan(
            kind="sft_training",
            run_plan=plan,
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

    def _build_rl(self, plan: RunPlan) -> AssemblyPlan:
        recipe = plan.settings.recipe
        if recipe.rl is None:
            raise ValueError("RL AssemblyPlan에는 recipe.rl이 필요합니다")

        models = tuple(
            self._rl_model_node(role, spec)
            for role, spec in recipe.rl.models.items()
        )
        return AssemblyPlan(
            kind="rl_training",
            run_plan=plan,
            models=models,
            data=self._data_node(plan),
            trainer=TrainerNode(
                kind="rl",
                training_config=recipe.training.model_dump(),
                algorithm=self._component(recipe.rl.algorithm, path="recipe.rl.algorithm"),
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

    def _rl_model_node(self, role: str, spec: RoleModelSpec) -> ModelNode:
        prefix = f"recipe.rl.models.{role}"
        return ModelNode(
            role=role,
            trainable=self._is_rl_model_trainable(role, spec),
            model=self._model_component(spec.model, path=prefix),
            head=self._optional_component(spec.head, path=f"{prefix}.head"),
            adapter=self._optional_component(spec.adapter, path=f"{prefix}.adapter"),
            optimizer=self._optional_component(spec.optimizer, path=f"{prefix}.optimizer"),
            scheduler=self._optional_component(spec.scheduler, path=f"{prefix}.scheduler"),
            loss=self._optional_component(spec.loss, path=f"{prefix}.loss"),
        )

    @staticmethod
    def _is_rl_model_trainable(role: str, spec: RoleModelSpec) -> bool:
        if spec.trainable is not None:
            return spec.trainable
        if spec.freeze:
            return False
        if spec.optimizer is not None:
            return True
        return role == "policy"

    def _data_node(self, plan: RunPlan) -> DataNode:
        data = plan.settings.recipe.data
        return DataNode(
            dataset=self._component(data.dataset, path="recipe.data.dataset"),
            val_dataset=self._optional_component(data.val_dataset, path="recipe.data.val_dataset"),
            collator=self._component(data.collator, path="recipe.data.collator"),
            sampler=self._optional_component(data.sampler, path="recipe.data.sampler"),
            dataloader_config=data.dataloader.model_dump(),
            distributed_intent=plan.distributed_intent,
        )

    def _strategy_node(self, plan: RunPlan) -> StrategyNode | None:
        distributed = plan.settings.config.compute.distributed
        if distributed is None or not plan.distributed_intent:
            return None

        config = (
            distributed.model_dump(mode="json")
            if hasattr(distributed, "model_dump")
            else dict(distributed)
        )
        strategy = config.get("strategy", "auto")
        return StrategyNode(
            config=config,
            strategy=strategy,
            capability_boundary=self._STRATEGY_CAPABILITIES,
            distributed_intent=plan.distributed_intent,
        )

    def _callback_nodes(self, plan: RunPlan) -> tuple[CallbackNode, ...]:
        return tuple(
            CallbackNode(config=self._component(config, path=config.path))
            for config in plan.callback_configs
        )

    def _component(self, spec: SettingsComponentSpec, *, path: str) -> ComponentSpec:
        self._check_semantic_conflicts(spec, path=path)
        return ComponentSpec.from_component(
            spec,
            path=path,
            resolved_component=self._resolve_alias(spec.component, path=path),
        )

    def _optional_component(
        self,
        spec: SettingsComponentSpec | None,
        *,
        path: str,
    ) -> ComponentSpec | None:
        if spec is None:
            return None
        return self._component(spec, path=path)

    def _model_component(self, spec: ModelComponentSpec, *, path: str) -> ComponentSpec:
        resolved = (
            self._resolve_alias(spec.component, path=path)
            if spec.component is not None
            else None
        )
        return ComponentSpec.from_model(spec, path=path, resolved_component=resolved)

    @staticmethod
    def _resolve_alias(component: str, *, path: str) -> str:
        try:
            return ComponentResolver()._resolve_alias(component)
        except ValueError as exc:
            raise ValueError(f"{path}._component_: {exc}") from exc

    @staticmethod
    def _check_semantic_conflicts(spec: SettingsComponentSpec, *, path: str) -> None:
        conflict_pairs = (("target", "target_modules"), ("save", "modules_to_save"))
        for semantic_key, raw_key in conflict_pairs:
            if semantic_key in spec.kwargs and raw_key in spec.kwargs:
                raise ValueError(
                    f"{path}: {semantic_key!r} and {raw_key!r} cannot be specified together"
                )
