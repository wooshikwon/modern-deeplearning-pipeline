"""CLI JSON 출력 스키마 — 에이전트가 파싱하는 구조화된 결과 모델.

각 CLI 명령의 JSON 출력 구조를 Pydantic 모델로 정의한다.
모든 필드는 실제 데이터 생산자(trainer, estimator, drift detector)가
반환하는 값과 1:1 대응한다.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── mdp train ──


class TrainResult(BaseModel):
    """mdp train / rl-train --format json 출력 스키마.

    데이터 출처:
    - metrics, total_epochs, total_steps, stopped_reason, duration_seconds
      → Trainer.train() / RLTrainer.train() 반환 dict
    - checkpoint_dir, output_dir → Settings.config.storage
    - monitoring → Trainer._maybe_compute_baseline() 반환값
    - algorithm → RLTrainer.train() 반환 dict (SFT에서는 None)
    """

    checkpoint_dir: str | None = None
    output_dir: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    total_epochs: int | None = None
    total_steps: int | None = None
    stopped_reason: str | None = None
    duration_seconds: float | None = None
    monitoring: dict[str, Any] | None = None
    algorithm: str | None = None
    run_id: str | None = None


# ── mdp inference ──


class InferenceResult(BaseModel):
    """mdp inference --format json 출력 스키마.

    데이터 출처:
    - output_path → run_batch_inference() 반환값
    - task → checkpoint recipe.task
    - monitoring → _detect_drift() → compare_baselines() 반환 dict
    - evaluation_metrics → metric.compute() 결과 집계
    """

    output_path: str | None = None
    task: str | None = None
    run_id: str | None = None
    monitoring: dict[str, Any] | None = None
    evaluation_metrics: dict[str, Any] | None = None


# ── mdp generate ──


class GenerateResult(BaseModel):
    """mdp generate --format json 출력 스키마."""

    output_path: str | None = None
    num_prompts: int | None = None
    num_generated: int | None = None
    num_samples: int | None = None


# ── mdp serve ──


class ServeResult(BaseModel):
    """mdp serve --format json 출력 스키마."""

    run_id: str | None = None
    port: int | None = None
    status: str = "starting"


# ── mdp estimate ──


class EstimateResult(BaseModel):
    """mdp estimate --format json 출력 스키마.

    데이터 출처: MemoryEstimator.estimate() 반환 dict 전체.
    """

    model_mem_gb: float | None = None
    gradient_mem_gb: float | None = None
    optimizer_mem_gb: float | None = None
    activation_mem_gb: float | None = None
    total_mem_gb: float | None = None
    suggested_gpus: int | None = None
    suggested_strategy: str | None = None


# ── mdp export ──


class ExportResult(BaseModel):
    """mdp export --format json 출력 스키마.

    데이터 출처:
    - output_dir → 내보내기 대상 디렉토리
    - model_class → type(target).__name__
    - merged → adapter merge 여부
    """

    output_dir: str | None = None
    model_class: str | None = None
    merged: bool = True


# ── mdp list ──


class ListModelsResult(BaseModel):
    """mdp list models --format json 출력 스키마."""

    what: str = "models"
    models: list[dict[str, Any]] = Field(default_factory=list)


class ListTasksResult(BaseModel):
    """mdp list tasks --format json 출력 스키마."""

    what: str = "tasks"
    tasks: list[dict[str, Any]] = Field(default_factory=list)


class ListStrategiesResult(BaseModel):
    """mdp list strategies --format json 출력 스키마."""

    what: str = "strategies"
    strategies: list[dict[str, Any]] = Field(default_factory=list)


class ListCallbacksResult(BaseModel):
    """mdp list callbacks --format json 출력 스키마."""

    what: str = "callbacks"
    callbacks: list[str] = Field(default_factory=list)
