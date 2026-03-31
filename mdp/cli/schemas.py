"""CLI JSON 출력 스키마 — 에이전트가 파싱하는 구조화된 결과 모델.

각 CLI 명령의 JSON 출력 구조를 Pydantic 모델로 정의한다.
이 스키마는 --format json 옵션 사용 시 출력의 계약(contract)이다.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── 공통 ──


class ErrorDetail(BaseModel):
    """구조화된 에러 정보."""

    type: str = Field(description="에러 유형: ValidationError, RuntimeError, TimeoutError 등")
    message: str = Field(description="사람이 읽을 수 있는 에러 메시지")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="에러 컨텍스트 (field, value, allowed 등)",
    )


# ── mdp train ──


class TrainMetricsFinal(BaseModel):
    """학습 종료 시점 메트릭."""

    train_loss: float | None = None
    val_loss: float | None = None
    val_accuracy: float | None = None
    val_f1: float | None = None


class TrainMetricsBest(BaseModel):
    """전체 학습 중 최고 성능 메트릭."""

    epoch: int | None = None
    val_accuracy: float | None = None
    val_loss: float | None = None


class TrainMetrics(BaseModel):
    """학습 메트릭 요약."""

    final: TrainMetricsFinal = Field(default_factory=TrainMetricsFinal)
    best: TrainMetricsBest = Field(default_factory=TrainMetricsBest)


class TrainingSummary(BaseModel):
    """학습 실행 요약."""

    total_epochs: int | None = None
    stopped_epoch: int | None = None
    stopped_reason: str | None = None
    duration_seconds: float | None = None
    samples_per_second: float | None = None


class MLflowInfo(BaseModel):
    """MLflow 실험 추적 정보."""

    tracking_uri: str | None = None
    experiment_name: str | None = None
    run_id: str | None = None
    artifact_uri: str | None = None


class MonitoringBaselineInfo(BaseModel):
    """학습 시 저장된 baseline 정보."""

    baseline_saved: bool = False
    baseline_path: str | None = None


class TrainResult(BaseModel):
    """mdp train --format json 출력 스키마."""

    run_id: str | None = None
    experiment_name: str | None = None
    checkpoint_dir: str | None = None
    output_dir: str | None = None
    metrics: TrainMetrics = Field(default_factory=TrainMetrics)
    training_summary: TrainingSummary = Field(default_factory=TrainingSummary)
    mlflow: MLflowInfo = Field(default_factory=MLflowInfo)
    monitoring: MonitoringBaselineInfo = Field(default_factory=MonitoringBaselineInfo)


# ── mdp inference ──


class PredictionDistribution(BaseModel):
    """예측 엔트로피 분포."""

    entropy_mean: float | None = None
    entropy_std: float | None = None
    baseline_entropy_mean: float | None = None
    baseline_entropy_std: float | None = None


class ClassShift(BaseModel):
    """개별 클래스의 분포 변화."""

    class_id: int | str
    train_ratio: float
    inference_ratio: float


class ClassDistributionShift(BaseModel):
    """클래스 분포 변화 분석."""

    kl_divergence: float | None = None
    top_shifted_classes: list[ClassShift] = Field(default_factory=list)


class ConfidenceAnalysis(BaseModel):
    """확신도 분석."""

    mean_confidence: float | None = None
    low_confidence_ratio: float | None = None
    baseline_mean_confidence: float | None = None
    baseline_low_confidence_ratio: float | None = None


class InferenceMonitoring(BaseModel):
    """추론 시 drift 모니터링 결과."""

    drift_detected: bool = False
    drift_score: float | None = None
    drift_threshold: float | None = None
    prediction_distribution: PredictionDistribution = Field(
        default_factory=PredictionDistribution
    )
    class_distribution_shift: ClassDistributionShift = Field(
        default_factory=ClassDistributionShift
    )
    confidence_analysis: ConfidenceAnalysis = Field(default_factory=ConfidenceAnalysis)
    alerts: list[str] = Field(default_factory=list)


class InferenceResult(BaseModel):
    """mdp inference --format json 출력 스키마."""

    output_path: str | None = None
    rows: int | None = None
    duration_seconds: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    task: str | None = None
    monitoring: InferenceMonitoring = Field(default_factory=InferenceMonitoring)
    # 생성 태스크 전용
    generation_config: dict[str, Any] | None = None


# ── mdp estimate ──


class ModelInfo(BaseModel):
    """모델 정보 요약."""

    class_path: str | None = None
    pretrained: str | None = None
    params_total: int | None = None
    params_trainable: int | None = None
    adapter: str | None = None


class MemoryEstimate(BaseModel):
    """메모리 추정 결과."""

    precision: str | None = None
    model_memory_gb: float | None = None
    gradient_memory_gb: float | None = None
    optimizer_memory_gb: float | None = None
    activation_memory_gb_estimate: float | None = None
    total_memory_gb: float | None = None


class Recommendation(BaseModel):
    """GPU/전략 추천."""

    suggested_gpus: int | None = None
    suggested_strategy: str | None = None
    gpu_memory_assumption_gb: int = 80
    fits_single_gpu: bool | None = None
    notes: list[str] = Field(default_factory=list)


class EstimateResult(BaseModel):
    """mdp estimate --format json 출력 스키마."""

    model_info: ModelInfo = Field(default_factory=ModelInfo)
    memory_estimate: MemoryEstimate = Field(default_factory=MemoryEstimate)
    recommendation: Recommendation = Field(default_factory=Recommendation)


# ── mdp list ──


class CatalogModel(BaseModel):
    """카탈로그 모델 정보."""

    name: str
    family: str
    class_path: str
    tasks: list[str] = Field(default_factory=list)
    pretrained: list[str] = Field(default_factory=list)
    params_million: float | None = None
    compatible_heads: list[str] = Field(default_factory=list)
    compatible_adapters: list[str] = Field(default_factory=list)


class TaskInfo(BaseModel):
    """태스크 정보."""

    name: str
    modality: str
    default_metrics: list[str] = Field(default_factory=list)
    compatible_heads: list[str] = Field(default_factory=list)
    description: str = ""


class StrategyInfo(BaseModel):
    """분산 전략 정보."""

    name: str
    strategy: str
    description: str = ""


class ListModelsResult(BaseModel):
    """mdp list models --format json 출력 스키마."""

    what: str = "models"
    models: list[CatalogModel] = Field(default_factory=list)


class ListTasksResult(BaseModel):
    """mdp list tasks --format json 출력 스키마."""

    what: str = "tasks"
    tasks: list[TaskInfo] = Field(default_factory=list)


class ListStrategiesResult(BaseModel):
    """mdp list strategies --format json 출력 스키마."""

    what: str = "strategies"
    strategies: list[StrategyInfo] = Field(default_factory=list)


class ListCallbacksResult(BaseModel):
    """mdp list callbacks --format json 출력 스키마."""

    what: str = "callbacks"
    callbacks: list[str] = Field(default_factory=list)
