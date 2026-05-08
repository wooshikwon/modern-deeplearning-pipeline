"""Typed component traversal helpers for settings validators."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

from mdp.settings.components import ComponentSpec, ModelComponentSpec
from mdp.settings.schema import Settings


def with_component_path(
    spec: ComponentSpec | ModelComponentSpec,
    path: str,
) -> ComponentSpec | ModelComponentSpec:
    """Return ``spec`` with a validator-local dot path attached."""
    if spec.path == path:
        return spec
    return replace(spec, path=path)


def iter_component_specs(settings: Settings) -> Iterator[ComponentSpec | ModelComponentSpec]:
    """Yield all typed component envelopes owned by ``settings``."""
    recipe = settings.recipe
    yield with_component_path(recipe.model, "recipe.model")

    for path, spec in (
        ("recipe.head", recipe.head),
        ("recipe.adapter", recipe.adapter),
        ("recipe.optimizer", recipe.optimizer),
        ("recipe.scheduler", recipe.scheduler),
        ("recipe.loss", recipe.loss),
        ("recipe.data.dataset", recipe.data.dataset),
        ("recipe.data.val_dataset", recipe.data.val_dataset),
        ("recipe.data.collator", recipe.data.collator),
        ("recipe.data.sampler", recipe.data.sampler),
    ):
        if spec is not None:
            yield with_component_path(spec, path)

    if recipe.evaluation is not None:
        for index, metric in enumerate(recipe.evaluation.metrics):
            if isinstance(metric, ComponentSpec):
                yield with_component_path(metric, f"recipe.evaluation.metrics[{index}]")

    if recipe.rl is None:
        return

    yield with_component_path(recipe.rl.algorithm, "recipe.rl.algorithm")
    for name, role in recipe.rl.models.items():
        prefix = f"recipe.rl.models.{name}"
        yield with_component_path(role.model, prefix)
        for key in ("head", "adapter", "optimizer", "scheduler", "loss"):
            spec = getattr(role, key)
            if spec is not None:
                yield with_component_path(spec, f"{prefix}.{key}")
