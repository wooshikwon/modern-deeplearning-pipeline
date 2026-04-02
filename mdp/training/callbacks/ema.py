"""Exponential Moving Average (EMA) callback for model parameters."""

from __future__ import annotations

import copy
import logging

import torch

from mdp.training.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)


class EMACallback(BaseCallback):
    """Maintain an exponential moving average of model parameters.

    During training, a shadow copy of every trainable parameter is
    updated with:

        ema_param = decay * ema_param + (1 - decay) * current_param

    At the end of training the EMA weights are copied back into the
    model so that subsequent evaluation / export uses the averaged
    weights.

    Parameters
    ----------
    decay:
        EMA decay factor.  Values close to 1.0 (e.g. 0.9999) produce
        a very slow-moving average.
    update_after_step:
        Only start updating the shadow parameters after this many
        training steps.
    update_every:
        Update the shadow parameters every *n* steps.
    """

    def __init__(
        self,
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
    ) -> None:
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every

        self._shadow_params: list[torch.Tensor] = []
        self._model: torch.nn.Module | None = None

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_train_start(self, **kwargs) -> None:
        model: torch.nn.Module | None = kwargs.get("model")
        if model is None:
            logger.warning("EMACallback: 'model' not provided in on_train_start kwargs.")
            return

        self._model = model
        self._shadow_params = [
            p.data.clone().cpu() for p in model.parameters() if p.requires_grad
        ]
        logger.info(
            "EMACallback: initialised shadow copy for %d parameter tensors.",
            len(self._shadow_params),
        )

    def on_batch_end(
        self,
        step: int,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        if self._model is None or not self._shadow_params:
            return
        if step < self.update_after_step:
            return
        if step % self.update_every != 0:
            return

        trainable = [p for p in self._model.parameters() if p.requires_grad]
        if len(trainable) != len(self._shadow_params):
            logger.warning(
                "EMACallback: trainable parameter count changed (%d → %d), skipping update.",
                len(self._shadow_params),
                len(trainable),
            )
            return
        for shadow, param in zip(self._shadow_params, trainable):
            shadow.mul_(self.decay).add_(param.data.cpu(), alpha=1.0 - self.decay)

    def on_train_end(
        self,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        if self._model is None or not self._shadow_params:
            return

        trainable = [p for p in self._model.parameters() if p.requires_grad]
        for shadow, param in zip(self._shadow_params, trainable):
            param.data.copy_(shadow)

        logger.info("EMACallback: copied EMA weights back to model.")
