"""MDP Pydantic 스키마 — Recipe, Config, Settings.

YAML 구조와 1:1 대응. 구조적 유효성(필드 존재, 타입 일치)만 검증한다.
비즈니스 검증(태스크-모델 호환성 등)은 validation/ 패키지가 담당한다.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ── Recipe 스키마 ──




class DataloaderSpec(BaseModel):
    """DataLoader 설정."""

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 2
    drop_last: bool = True

    @model_validator(mode="after")
    def _fix_persistent_workers(self) -> "DataloaderSpec":
        if self.num_workers == 0:
            self.persistent_workers = False
            self.prefetch_factor = None
        return self


class DataSpec(BaseModel):
    """데이터 파이프라인. dataset과 collator를 _component_ 패턴으로 명시한다.

    기존 source/label_strategy/tokenizer/augmentation 고정 스키마를 제거하고,
    모든 데이터 컴포넌트를 _component_ 패턴으로 통일한다.

    Example::

        data:
          dataset:
            _component_: HuggingFaceDataset
            source: wikitext
            split: train
          collator:
            _component_: CausalLMCollator
            tokenizer: gpt2
          dataloader:
            batch_size: 32
    """

    dataset: dict[str, Any]                          # _component_ 필수
    collator: dict[str, Any]                         # _component_ 필수
    val_dataset: dict[str, Any] | None = None        # _component_, None이면 validation 비활성
    dataloader: DataloaderSpec = Field(default_factory=DataloaderSpec)


class TrainingSpec(BaseModel):
    """학습 루프 설정."""

    epochs: float | None = None
    max_steps: int | None = None
    precision: str = "fp32"
    gradient_accumulation_steps: int = 1
    gradient_clip_max_norm: float | None = None
    gradient_checkpointing: bool = False
    val_check_interval: float = 1.0
    val_check_unit: Literal["epoch", "step"] = "epoch"
    compile: str | bool = False


class MonitoringSpec(BaseModel):
    """데이터 분포 모니터링."""

    enabled: bool = False
    baseline: dict[str, Any] = Field(default_factory=dict)
    drift: dict[str, Any] = Field(default_factory=dict)


class GenerationSpec(BaseModel):
    """자기회귀 생성 설정 (서빙/추론 전용)."""

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.0


class RLGenerationSpec(GenerationSpec):
    """RL 학습 중 응답 생성 설정. GenerationSpec + RL 전용 필드."""

    group_size: int = 1  # GRPO K개 응답 생성. 1이면 기존 동작.


class RLSpec(BaseModel):
    """RL alignment 설정. None이면 SFT."""

    algorithm: dict[str, Any]  # _component_ 패턴 (DPO, GRPO, PPO)
    models: dict[str, dict[str, Any]]  # 역할별 모델 정의 (policy 필수)
    generation: RLGenerationSpec | None = None  # GRPO/PPO 전용 응답 생성 파라미터


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
    # SFT 필드
    model: dict[str, Any] = Field(
        default_factory=lambda: {"_component_": "transformers.AutoModelForCausalLM"}
    )
    head: dict[str, Any] | None = None
    adapter: dict[str, Any] | None = None
    data: DataSpec
    training: TrainingSpec
    optimizer: dict[str, Any] | None = None  # SFT용 (RL은 models.*.optimizer)
    scheduler: dict[str, Any] | None = None
    loss: dict[str, Any] | None = None
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    generation: GenerationSpec | None = None  # 서빙/추론 전용 (rl.generation과 독립)
    monitoring: MonitoringSpec = Field(default_factory=MonitoringSpec)
    callbacks: list[dict[str, Any]] = Field(default_factory=list)
    metadata: MetadataSpec
    # RL
    rl: RLSpec | None = None  # None이면 SFT

    @model_validator(mode="after")
    def check_training_duration(self):
        if self.training.epochs is None and self.training.max_steps is None:
            raise ValueError("training.epochs 또는 training.max_steps 중 하나는 필수")
        return self

    @model_validator(mode="after")
    def check_rl_consistency(self):
        if self.rl is not None:
            if "policy" not in self.rl.models:
                raise ValueError("RL 학습에는 rl.models.policy가 필수입니다")
            policy = self.rl.models["policy"]
            if policy.get("optimizer") is None:
                raise ValueError("rl.models.policy에는 optimizer가 필수입니다")
        return self


# ── Config 스키마 ──


class ComputeConfig(BaseModel):
    """실행 환경 설정.

    로컬/torchrun 기반 학습만 지원한다. 원격·클라우드 오케스트레이션
    (SSH job submission, SkyPilot 런칭 등)은 mdp의 책임이 아니다 —
    사용자가 이미 실행 환경 안에 있다고 가정한다.
    """

    target: str = "local"
    gpus: int | str | list[int] = "auto"
    distributed: dict[str, Any] | None = None


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
    batch_window_ms: float = 50.0  # 동적 배칭 시간 창 (ms)
    device_map: str | None = None  # "auto", "balanced", "sequential"
    max_memory: dict[str, str] | None = None  # {"0": "24GiB", "1": "40GiB"}


class JobConfig(BaseModel):
    """작업 제어."""

    name: str | None = None
    resume: str = "auto"
    max_retries: int = 0


class Config(BaseModel):
    """인프라 설정서. '어디서 실행할지'를 기술한다."""

    environment: dict[str, Any] = Field(default_factory=lambda: {"name": "local"})
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    serving: ServingConfig | None = None
    job: JobConfig = Field(default_factory=JobConfig)


# ── 통합 Settings ──


class Settings(BaseModel):
    """Recipe + Config를 합친 통합 설정 객체."""

    recipe: Recipe
    config: Config
