"""MDP Pydantic 스키마 — Recipe, Config, Settings.

YAML 구조와 1:1 대응. 구조적 유효성(필드 존재, 타입 일치)만 검증한다.
비즈니스 검증(태스크-모델 호환성 등)은 validation/ 패키지가 담당한다.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


# ── Recipe 스키마 ──


class ModelSpec(BaseModel):
    """모델 명세. class_path + pretrained URI 패턴."""

    class_path: str
    pretrained: str | None = None
    torch_dtype: str | None = None
    attn_implementation: str | None = None
    init_args: dict[str, Any] = Field(default_factory=dict)


class AdapterSpec(BaseModel):
    """어댑터 명세. method로 lora/qlora 분기."""

    method: str  # "lora", "qlora"
    r: int | None = None
    alpha: int | None = None
    dropout: float = 0.0
    target_modules: list[str] | str = "all_linear"
    quantization: dict[str, Any] | None = None
    modules_to_save: list[str] = Field(default_factory=list)


class DataloaderSpec(BaseModel):
    """DataLoader 설정."""

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = True


class DataSpec(BaseModel):
    """데이터 파이프라인. source로 데이터를 지정하고, format/column_map으로 스키마를 맞춘다."""

    source: str  # HF Hub 이름, 또는 로컬 파일/디렉토리 경로
    fields: dict[str, str] = Field(default_factory=dict)  # {role: column_name}
    format: str = "auto"  # alpaca | sharegpt | completion | text | auto
    split: str | dict[str, Any] = "train"
    subset: str | None = None
    column_map: dict[str, str] | None = None
    streaming: bool = False
    data_files: str | dict[str, str] | None = None
    tokenizer: dict[str, Any] | None = None
    augmentation: dict[str, Any] | None = None
    dataloader: DataloaderSpec = Field(default_factory=DataloaderSpec)


class TrainingSpec(BaseModel):
    """학습 루프 설정."""

    epochs: int | None = None
    max_steps: int | None = None
    precision: str = "fp32"
    gradient_accumulation_steps: int = 1
    gradient_clip_max_norm: float | None = None
    gradient_checkpointing: bool = False
    val_check_interval: float = 1.0
    val_check_unit: str = "epoch"  # "epoch" or "step"
    compile: str | bool = False


class MonitoringSpec(BaseModel):
    """데이터 분포 모니터링."""

    enabled: bool = False
    baseline: dict[str, Any] = Field(default_factory=dict)
    drift: dict[str, Any] = Field(default_factory=dict)


class GenerationSpec(BaseModel):
    """자기회귀 생성 설정 (추론 전용)."""

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.0


class EvaluationSpec(BaseModel):
    """평가 설정.

    metrics 항목은 str(alias 이름) 또는 dict(_component_ 패턴) 형태.
    """

    metrics: list[dict[str, Any] | str] = Field(default_factory=list)


class MetadataSpec(BaseModel):
    """실험 메타데이터."""

    author: str
    description: str


class Recipe(BaseModel):
    """실험 정의서. '무엇을 학습할지'를 기술한다."""

    name: str
    task: str
    model: ModelSpec
    head: dict[str, Any] | None = None
    adapter: AdapterSpec | None = None
    data: DataSpec
    training: TrainingSpec
    optimizer: dict[str, Any]  # _component_ 패턴 (required)
    scheduler: dict[str, Any] | None = None
    loss: dict[str, Any] | None = None
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    generation: GenerationSpec | None = None
    monitoring: MonitoringSpec = Field(default_factory=MonitoringSpec)
    callbacks: list[dict[str, Any]] = Field(default_factory=list)
    metadata: MetadataSpec

    @model_validator(mode="after")
    def check_training_duration(self):
        if self.training.epochs is None and self.training.max_steps is None:
            raise ValueError("training.epochs 또는 training.max_steps 중 하나는 필수")
        return self


# ── Config 스키마 ──


class ComputeConfig(BaseModel):
    """실행 환경 설정."""

    target: str = "local"
    gpus: int | str | list[int] = "auto"
    host: str | None = None
    user: str | None = None
    ssh_key: str | None = None
    working_dir: str | None = None
    nodes: list[dict[str, Any]] | None = None
    distributed: dict[str, Any] | None = None
    # cloud 전용 필드
    provider: str | None = None
    accelerators: str | None = None
    spot: bool = False
    region: str | None = None
    disk_size: int = 256


class EnvironmentSetupConfig(BaseModel):
    """원격/클라우드 환경 준비."""

    container: str | None = None
    dependencies: str | None = None
    setup_commands: list[str] = Field(default_factory=list)


class MLflowConfig(BaseModel):
    """MLflow 실험 추적."""

    tracking_uri: str = "./mlruns"
    experiment_name: str = "default"


class StorageConfig(BaseModel):
    """체크포인트/출력 저장소."""

    checkpoint_dir: str = "./checkpoints"
    checkpoint_every_n_steps: int | None = None
    output_dir: str = "./outputs"


class ServingConfig(BaseModel):
    """서빙 설정."""

    backend: str = "torchserve"
    model_repository: str | None = None
    max_batch_size: int = 1
    instance_count: int = 1


class JobConfig(BaseModel):
    """작업 제어."""

    name: str | None = None
    resume: str = "auto"
    max_retries: int = 0


class Config(BaseModel):
    """인프라 설정서. '어디서 실행할지'를 기술한다."""

    environment: dict[str, Any] = Field(default_factory=lambda: {"name": "local"})
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    environment_setup: EnvironmentSetupConfig = Field(
        default_factory=EnvironmentSetupConfig
    )
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    serving: ServingConfig | None = None
    job: JobConfig = Field(default_factory=JobConfig)


# ── 통합 Settings ──


class Settings(BaseModel):
    """Recipe + Config를 합친 통합 설정 객체."""

    recipe: Recipe
    config: Config
