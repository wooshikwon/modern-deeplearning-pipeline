"""Worker-side runtime setup for torchrun training processes."""

from __future__ import annotations

import os

from mdp.runtime.context import RuntimeContext


def init_distributed_if_torchrun(settings) -> RuntimeContext:
    """Initialize torch.distributed before any dataloader materialization."""
    context = RuntimeContext.from_env()
    if not context.is_torchrun:
        return context

    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        return context

    if torch.cuda.is_available():
        torch.cuda.set_device(context.local_rank)

    dist_cfg = settings.config.compute.distributed
    backend = (
        dist_cfg.backend
        if hasattr(dist_cfg, "backend")
        else dist_cfg.get("backend") if isinstance(dist_cfg, dict) else None
    )
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend)
    return context


def apply_liger_patches_for_training() -> None:
    """Apply Liger monkey patches before model materialization."""
    from mdp._liger_patch import apply_liger_patches

    apply_liger_patches()


def is_main_process() -> bool:
    """Return whether the current process owns rank-0 side effects."""
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return int(os.environ.get("RANK", "0")) == 0
