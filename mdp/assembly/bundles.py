"""Training bundle containers and shared trainer dependency preparation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mdp.settings.components import ComponentSpec
from mdp.settings.resolver import ComponentResolver
from mdp.settings.schema import Settings
from mdp.training._common import create_expert_parallel, create_strategy
from mdp.training._schedulers import (
    create_scheduler_with_warmup,
    parse_warmup_config,
)
from mdp.training.callbacks.base import BaseCallback
from mdp.training.callbacks.early_stopping import EarlyStopping
from mdp.training.callbacks.ema import EMACallback


@dataclass
class SFTTrainingBundle:
    """Concrete dependencies needed to construct an SFT Trainer."""

    settings: Settings
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader | None = None
    callbacks: list[BaseCallback] | None = None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: Any | None = None
    scheduler_interval: str = "step"
    loss_fn: nn.Module | None = None
    strategy: Any | None = None
    expert_parallel: Any | None = None


@dataclass
class RLTrainingBundle:
    """Concrete dependencies needed to construct an RLTrainer."""

    settings: Settings
    models: dict[str, nn.Module]
    train_loader: DataLoader
    val_loader: DataLoader | None = None
    callbacks: list[BaseCallback] | None = None
    algorithm: Any | None = None
    trainable: dict[str, nn.Module] | None = None
    frozen: dict[str, nn.Module] | None = None
    optimizers: dict[str, torch.optim.Optimizer] | None = None
    schedulers: dict[str, Any] | None = None
    scheduler_intervals: dict[str, str] | None = None
    strategy: Any | None = None
    expert_parallel: Any | None = None


def estimate_total_steps(
    *,
    max_steps: int | None,
    epochs: int | float | None,
    grad_accum_steps: int,
    train_loader: Any,
) -> int:
    """Estimate total optimizer steps from trainer duration settings."""
    if max_steps:
        return max_steps
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    return int(steps_per_epoch * (epochs or 1))


def promote_training_callbacks(
    settings: Settings,
    callbacks: Sequence[BaseCallback] | None,
) -> list[BaseCallback]:
    """Append training-spec callbacks using the legacy Trainer/RLTrainer rules."""
    promoted = list(callbacks) if callbacks else []
    training = settings.recipe.training
    if training.early_stopping is not None:
        promoted.append(EarlyStopping(**training.early_stopping.model_dump()))
    if training.ema is not None:
        promoted.append(EMACallback(**training.ema.model_dump()))
    return promoted


def create_sft_optimizer(
    model: nn.Module,
    config: ComponentSpec,
    resolver: ComponentResolver,
) -> torch.optim.Optimizer:
    """Create the single SFT optimizer while preserving model override priority."""
    custom = (
        model.configure_optimizers()
        if hasattr(model, "configure_optimizers")
        else None
    )
    if custom and isinstance(custom, dict) and "optimizer" in custom:
        return custom["optimizer"]

    klass, kwargs = resolver.resolve_partial(config)
    weight_decay = kwargs.get("weight_decay", 0.0)

    if weight_decay > 0:
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                "bias" in name
                or "norm" in name
                or "layernorm" in name
                or "ln_" in name
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        kwargs.pop("weight_decay", None)
        return klass(param_groups, **kwargs)

    return klass(model.parameters(), **kwargs)


def create_sft_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ComponentSpec | None,
    *,
    total_steps: int,
    resolver: ComponentResolver,
) -> tuple[Any | None, str]:
    """Create the SFT scheduler and its step/epoch interval."""
    if config is None:
        return None, "step"

    scheduler_kwargs = dict(config.kwargs)
    warmup = parse_warmup_config(scheduler_kwargs, total_steps)
    klass, kwargs = resolver.resolve_partial(replace(config, kwargs=scheduler_kwargs))
    base_scheduler = klass(optimizer, **kwargs)
    scheduler = create_scheduler_with_warmup(optimizer, base_scheduler, warmup)
    return scheduler, warmup.interval


def create_loss(
    config: ComponentSpec | None,
    resolver: ComponentResolver,
) -> nn.Module | None:
    """Resolve an optional SFT loss component."""
    if config is None:
        return None
    return resolver.resolve(config)


def build_sft_training_bundle(
    *,
    settings: Settings,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    callbacks: Sequence[BaseCallback] | None = None,
    resolver: ComponentResolver | None = None,
) -> SFTTrainingBundle:
    """Prepare the legacy SFT trainer dependencies as a bundle."""
    resolver = resolver or ComponentResolver()
    training = settings.recipe.training
    total_steps = estimate_total_steps(
        max_steps=training.max_steps,
        epochs=training.epochs,
        grad_accum_steps=training.gradient_accumulation_steps,
        train_loader=train_loader,
    )
    optimizer = create_sft_optimizer(model, settings.recipe.optimizer, resolver)
    scheduler, scheduler_interval = create_sft_scheduler(
        optimizer,
        settings.recipe.scheduler,
        total_steps=total_steps,
        resolver=resolver,
    )
    return SFTTrainingBundle(
        settings=settings,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=promote_training_callbacks(settings, callbacks),
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_interval=scheduler_interval,
        loss_fn=create_loss(settings.recipe.loss, resolver),
        strategy=create_strategy(settings, resolver),
        expert_parallel=create_expert_parallel(settings),
    )


def build_rl_training_bundle(
    *,
    settings: Settings,
    models: dict[str, nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    callbacks: Sequence[BaseCallback] | None = None,
    resolver: ComponentResolver | None = None,
) -> RLTrainingBundle:
    """Prepare RLTrainer dependencies while preserving per-model optimizer rules."""
    resolver = resolver or ComponentResolver()
    if settings.recipe.rl is None:
        raise ValueError("RLTrainingBundle에는 recipe.rl이 필요합니다")

    training = settings.recipe.training
    total_steps = estimate_total_steps(
        max_steps=training.max_steps,
        epochs=training.epochs,
        grad_accum_steps=training.gradient_accumulation_steps,
        train_loader=train_loader,
    )

    trainable: dict[str, nn.Module] = {}
    frozen: dict[str, nn.Module] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}
    schedulers: dict[str, Any] = {}
    scheduler_intervals: dict[str, str] = {}

    for name, spec in settings.recipe.rl.models.items():
        model = models[name]
        if spec.optimizer is not None:
            trainable[name] = model
            klass, kwargs = resolver.resolve_partial(spec.optimizer)
            optimizers[name] = klass(model.parameters(), **kwargs)
            if spec.scheduler is not None:
                scheduler_kwargs = dict(spec.scheduler.kwargs)
                warmup = parse_warmup_config(scheduler_kwargs, total_steps)
                s_klass, s_kwargs = resolver.resolve_partial(
                    replace(spec.scheduler, kwargs=scheduler_kwargs)
                )
                base_scheduler = s_klass(optimizers[name], **s_kwargs)
                schedulers[name] = create_scheduler_with_warmup(
                    optimizers[name], base_scheduler, warmup
                )
                scheduler_intervals[name] = warmup.interval
        else:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            frozen[name] = model

    return RLTrainingBundle(
        settings=settings,
        models=models,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=promote_training_callbacks(settings, callbacks),
        algorithm=resolver.resolve(settings.recipe.rl.algorithm),
        trainable=trainable,
        frozen=frozen,
        optimizers=optimizers,
        schedulers=schedulers,
        scheduler_intervals=scheduler_intervals,
        strategy=create_strategy(settings, resolver),
        expert_parallel=create_expert_parallel(settings),
    )
