"""MDP Pydantic 스키마 — Recipe, Config, Settings.

YAML 구조와 1:1 대응. 구조적 유효성(필드 존재, 타입 일치)만 검증한다.
비즈니스 검증(태스크-모델 호환성 등)은 validation/ 패키지가 담당한다.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    sampler: dict[str, Any] | None = None            # _component_, None이면 기존 동작 보존
    dataloader: DataloaderSpec = Field(default_factory=DataloaderSpec)


class EarlyStoppingSpec(BaseModel):
    """학습 조기 종료 기준. monitor 메트릭이 patience 번 연속 개선되지 않으면 중단."""

    monitor: str = "val_loss"
    patience: int = Field(default=5, ge=1)
    mode: Literal["min", "max"] = "min"
    min_delta: float = Field(default=0.0, ge=0.0)


class EMASpec(BaseModel):
    """파라미터 지수이동평균. on_train_end에서 EMA 가중치를 모델에 복사하여 최종 평가/저장 대상으로 사용."""

    decay: float = Field(default=0.9999, gt=0.0, lt=1.0)
    update_after_step: int = Field(default=0, ge=0)
    update_every: int = Field(default=1, ge=1)


class TrainingSpec(BaseModel):
    """학습 루프 설정.

    Training duration semantics:
    - epochs (float | None): 학습할 에폭 수. None이면 max_steps만으로 종료 조건 결정.
    - max_steps (int | None): 학습할 전역 스텝 수. None이면 epochs만으로 종료 조건 결정.
    - 두 값 모두 지정되면 먼저 도달한 조건에서 종료된다 (early-hit).
      예: epochs=3, max_steps=200이면 200 step 도달이 3 epoch 도달보다 빠르면
      200 step에서 종료.
    - 최소 하나는 필수(Recipe.check_training_duration validator). 둘 다 None이면 ValueError.
    """

    epochs: float | None = Field(
        default=None,
        description="학습할 에폭 수. max_steps와 함께 지정되면 먼저 도달한 조건에서 종료됨.",
    )
    max_steps: int | None = Field(
        default=None,
        description="학습할 전역 스텝 수. epochs와 함께 지정되면 먼저 도달한 조건에서 종료됨.",
    )
    precision: str = "fp32"
    gradient_accumulation_steps: int = 1
    gradient_clip_max_norm: float | None = None
    gradient_checkpointing: bool = False
    val_check_interval: float = 1.0
    val_check_unit: Literal["epoch", "step"] = "epoch"
    compile: str | bool = False
    early_stopping: EarlyStoppingSpec | None = None
    ema: EMASpec | None = None


class MonitoringSpec(BaseModel):
    """학습 모니터링 설정.

    본 모델은 두 갈래의 기능을 하나의 네임스페이스에 모은다:

    1) **System logging** (spec-system-logging-cleanup §U4/§U5/§U6 공유 계약):
       - ``log_every_n_steps``: rank-0 step-progress 로그 간격. 파일 리다이렉트
         환경에서도 학습 진행이 stdout 에 남도록 하는 text-progress 의 주기.
       - ``memory_history``: ``torch.cuda.memory._record_memory_history`` 기반
         tensor-level snapshot on/off (U5 소비).
       - ``verbose``: True 이면 ``Rank0Filter`` 와 외부 logger downgrade 를 모두
         비활성화. 디버깅 전용.
    2) **데이터 분포 monitoring** (기존 ``baseline`` / ``drift`` 기능):
       - ``enabled``: baseline/drift 계산 활성 여부.
       - ``baseline`` / ``drift``: 세부 파라미터 dict.

    두 갈래는 서로 독립이며 extra=forbid 로 오타를 차단한다.
    """

    model_config = ConfigDict(extra="forbid")

    # ── System logging (spec-system-logging-cleanup §U4/U5/U6) ──
    log_every_n_steps: int = Field(
        default=10,
        ge=1,
        description="step progress 로그 간격 (rank-0). file redirect 환경에서도 "
        "학습 진행이 보이도록 하는 text-progress 주기.",
    )
    memory_history: bool = Field(
        default=False,
        description="torch.cuda.memory._record_memory_history snapshot on/off — "
        "U5 에서 소비. 활성 시 run 시작에 context/stacks 수집, 종료·OOM 시 "
        "_dump_snapshot 수행.",
    )
    verbose: bool = Field(
        default=False,
        description="True 이면 Rank0Filter · 외부 logger downgrade 모두 비활성. "
        "디버깅 전용. 운영 기본값은 False(조용한 로그).",
    )

    # ── 데이터 분포 monitoring (기존 baseline/drift) ──
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

    model_config = ConfigDict(extra="forbid")

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
