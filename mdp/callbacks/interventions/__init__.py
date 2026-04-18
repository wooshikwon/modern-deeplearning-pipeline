"""Intervention callbacks -- 모델 출력을 직접 수정하는 개입 콜백 모음."""

from __future__ import annotations

import json
import logging
from typing import Any

from mdp.callbacks.interventions.logit_bias import LogitBias
from mdp.callbacks.interventions.residual_add import ResidualAdd

__all__ = ["ResidualAdd", "LogitBias", "apply_intervention_tags"]

logger = logging.getLogger(__name__)

_MLFLOW_TAG_MAX_LEN = 5000


def apply_intervention_tags(callbacks: list[Any]) -> None:
    """intervention callback 메타데이터를 stdout/log에 출력하고, 활성 MLflow run이 있으면 tag를 붙인다.

    Parameters
    ----------
    callbacks:
        추론 콜백 리스트. ``is_intervention=True`` 인 콜백만 처리된다.
        리스트가 비어 있거나 intervention callback이 없으면 아무것도 하지 않는다.
    """
    intervention_cbs = [cb for cb in callbacks if getattr(cb, "is_intervention", False)]
    if not intervention_cbs:
        return

    try:
        import mlflow
    except ImportError:
        # mlflow 미설치: stdout 로깅만 수행
        for i, cb in enumerate(intervention_cbs):
            logger.info("intervention[%d]=%s", i, cb.metadata)
        return

    if mlflow.active_run() is None:
        logger.info(
            "Interventions present but no active MLflow run; logging to stdout only."
        )
        for i, cb in enumerate(intervention_cbs):
            logger.info("intervention[%d]=%s", i, cb.metadata)
        return

    for i, cb in enumerate(intervention_cbs):
        md: dict[str, Any] = cb.metadata
        for k, v in md.items():
            val_str = json.dumps(v) if isinstance(v, list) else str(v)
            # MLflow tag 값은 5000자 제한
            if len(val_str) > _MLFLOW_TAG_MAX_LEN:
                val_str = val_str[:4950] + "..."
            mlflow.set_tag(f"intervention.{i}.{k}", val_str)
