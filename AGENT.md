# MDP -- Modern Deeplearning Pipeline

YAML 설정으로 딥러닝 모델의 학습, 추론, 서빙을 수행하는 CLI 도구.

## Commands

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `mdp init <name> --task <task> --model <model>` | 프로젝트 스켈레톤 생성 | `--task, --model` |
| `mdp train -r recipe.yaml -c config.yaml` | 학습 실행 | `--format json` |
| `mdp inference -r recipe.yaml -c config.yaml --checkpoint <path>` | 배치 추론 | `--format json` |
| `mdp estimate -r recipe.yaml` | GPU 메모리 추정 | `--format json` |
| `mdp serve -r recipe.yaml -c config.yaml` | 실시간 서빙 시작 | (backend은 Config에서 설정) |
| `mdp list <what>` | 등록 컴포넌트 조회 | `--task`, `models`, `tasks`, `callbacks`, `strategies` |
| `mdp version` | 버전 출력 | `--format json` |

`--format json` 글로벌 옵션을 지원한다. 기본값은 `text` (Rich 테이블). 단, `mdp list`는 현재 Rich 출력만 지원한다.

## Agent Discovery Flow

에이전트가 MDP를 처음 사용할 때의 표준 플로우:

```bash
# Step 1: task + 호환 모델 조회
mdp list tasks

# Step 2: 특정 task 모델 상세
mdp list models --task text_generation

# Step 3: 프로젝트 생성 (name이 첫 번째 인자)
mdp init my_project --task text_generation --model llama3-8b

# Step 4: Recipe의 ??? 채운 후 학습 (recipe 파일은 항상 example.yaml)
mdp train -r my_project/recipes/example.yaml -c my_project/configs/local.yaml --format json
```

> `--model` 값은 카탈로그의 `name` 필드 (예: `llama3-8b`, `gpt2`, `vit-base-patch16-224`)

Recipe에서 `???`로 표시된 필드는 에이전트가 채워야 하는 부분:
- `data.source`, `data.fields.*`, `head.num_classes`, `metadata.*`

## Two-File System

- **Recipe** (실험 정의): 무엇을 학습할지 -- 모델, 데이터, 하이퍼파라미터
- **Config** (인프라 설정): 어디서 실행할지 -- GPU, 분산 전략, MLflow, 체크포인트 경로

동일 Recipe를 다른 Config로 실행하면 같은 실험을 다른 환경에서 재현할 수 있다.

---

## Recipe YAML Schema

### Top-Level Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | str | yes | - | 실험 이름. 영문+숫자+하이픈+언더스코어 |
| `task` | str | yes | - | 태스크. 아래 "Available Models by Task" 참조 |
| `model` | object | yes | - | 모델 명세 |
| `head` | object | no | null | Head 클래스. AutoModelFor* 사용 시 생략 가능 |
| `adapter` | object | no | null | 어댑터 (LoRA, QLoRA 등) |
| `data` | object | yes | - | 데이터 파이프라인 |
| `training` | object | yes | - | 학습 루프 설정 |
| `optimizer` | object | yes | - | 옵티마이저 (`_component_` 패턴) |
| `scheduler` | object | no | null | 학습률 스케줄러 (`_component_` 패턴) |
| `loss` | object | no | null | 손실 함수 (`_component_` 패턴). HF 모델은 내장 loss 사용 시 생략 |
| `evaluation` | object | no | `{}` | 평가 설정 |
| `generation` | object | no | null | 자기회귀 생성 설정 (추론 전용) |
| `monitoring` | object | no | `{enabled: false}` | 데이터 분포 모니터링 |
| `callbacks` | list | no | `[]` | 콜백 리스트 (`_component_` 패턴) |
| `metadata` | object | yes | - | 실험 메타데이터 (author, description) |

### model.*

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model.class_path` | str | yes | - | 모델 클래스 Python import path |
| `model.pretrained` | str | no | null | 사전학습 가중치 URI (`hf://`, `timm://`, `ultralytics://`, `local://`) |
| `model.torch_dtype` | str | no | null | 모델 로드 dtype: `float32`, `float16`, `bfloat16` |
| `model.attn_implementation` | str | no | null | `flash_attention_2`, `sdpa`, `eager` |
| `model.init_args` | dict | no | `{}` | 모델 생성자에 전달할 추가 인자 |

### head.*

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `head._component_` | str | conditional | - | Head 클래스. backbone 모델 사용 시 필수, `AutoModelFor*` 사용 시 생략 (내장 head) |
| `head._target_attr` | str | conditional | - | 모델에서 교체할 속성명 (예: `"fc"`, `"classifier"`). backbone 모델 사용 시 필수 |
| `head.num_classes` | int | conditional | - | 분류/검출/세그멘테이션에서 필수 |

Head alias 목록: `ClassificationHead`, `DetectionHead`, `SegmentationHead`, `TokenClassificationHead`, `CausalLMHead`, `Seq2SeqLMHead`, `DualEncoderHead`, `SequenceClassificationHead`

### adapter.*

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `adapter.method` | str | yes | - | `lora`, `qlora` |
| `adapter.r` | int | yes | - | LoRA rank |
| `adapter.alpha` | int | no | null | LoRA alpha |
| `adapter.dropout` | float | no | `0.0` | LoRA dropout |
| `adapter.target_modules` | list[str] \| str | no | `"all_linear"` | 적용 대상 모듈 |
| `adapter.quantization` | dict | conditional | null | QLoRA 시 필수. `{bits: 4}` |
| `adapter.modules_to_save` | list[str] | no | `[]` | LoRA 적용 제외하고 전체 학습할 모듈 |

### data.*

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `data.source` | str | yes | - | HF Hub 이름 (예: `cifar10`) 또는 로컬 경로 (예: `./data/train.csv`) |
| `data.fields` | dict | no | `{}` | `{role: column_name}` 매핑. role이 전처리를 결정 |
| `data.format` | str | no | `"auto"` | 로컬 파일 포맷: `auto`, `csv`, `json`, `parquet`, `imagefolder` |
| `data.split` | str | no | `"train"` | 학습 split 이름 |
| `data.subset` | str | no | null | HF 데이터셋의 config/subset 이름 |
| `data.streaming` | bool | no | `false` | 스트리밍 모드 |
| `data.data_files` | str \| dict | no | null | HF load_dataset data_files 인자 |
| `data.augmentation` | dict | no | null | 비전 태스크에서 사용. `train`/`val` 키 하위에 `steps` 리스트 |
| `data.tokenizer` | dict | no | null | 언어 태스크에서 사용 |
| `data.dataloader.batch_size` | int | no | `32` | 배치 크기 |
| `data.dataloader.num_workers` | int | no | `4` | DataLoader 워커 수 |
| `data.dataloader.pin_memory` | bool | no | `true` | GPU 전송 최적화 |
| `data.dataloader.persistent_workers` | bool | no | `true` | 워커 유지 |
| `data.dataloader.prefetch_factor` | int | no | `2` | 프리페치 배수 |
| `data.dataloader.drop_last` | bool | no | `true` | 마지막 불완전 배치 드롭 |

source 종류에 따라 `datasets.load_dataset()`로 자동 로딩된다:
- HF Hub 이름 → `load_dataset("cifar10", ...)`
- 로컬 디렉토리 → `load_dataset("imagefolder", data_dir=source, ...)`
- 로컬 `.csv`/`.json`/`.jsonl` 파일 → `load_dataset("csv"/"json", data_files=source, ...)`

collate 함수는 `fields` roles에서 파생된 label strategy로 자동 선택된다.

#### data.fields 역할명과 전처리 매핑

`fields`의 키(역할)가 전처리 파이프라인을 결정한다:

| fields roles | label strategy | collator | 대표 task |
|-------------|---------------|----------|----------|
| `{text: ...}` | `causal` | `DataCollatorForLanguageModeling` | text_generation |
| `{text: ..., target: ...}` | `seq2seq` | `DataCollatorForSeq2Seq` | seq2seq |
| `{text: ..., label: ...}` | `copy` | `DataCollatorWithPadding` | text_classification |
| `{text: ..., token_labels: ...}` | `align` | `DataCollatorWithPadding` | token_classification |
| `{image: ..., label: ...}` | `none` | None (기본 torch collate) | image_classification |

#### data.augmentation 형식

```yaml
augmentation:
  train:
    steps:
      - type: RandomResizedCrop
        params: {size: [224, 224]}
      - type: RandomHorizontalFlip
      - type: ToDtype
        params: {dtype: float32, scale: true}
      - type: Normalize
        params: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
  val:
    steps:
      - type: Resize
        params: {size: [256, 256]}
      - type: CenterCrop
        params: {size: [224, 224]}
      - type: ToDtype
        params: {dtype: float32, scale: true}
      - type: Normalize
        params: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}
```

`type`은 `torchvision.transforms.v2`의 클래스명. `_component_` 패턴은 사용하지 않는다.

#### data.tokenizer 형식

```yaml
tokenizer:
  pretrained: gpt2          # 필수. AutoTokenizer.from_pretrained()에 전달
  max_length: 512            # 선택, 기본 512
  max_target_length: 512     # seq2seq target 전용
  padding: false             # DataCollator가 동적 패딩 담당
  truncation: true
  chat_template: null        # 대화형 토큰화 (apply_chat_template)
```

`_component_` 패턴은 사용하지 않는다. `AutoTokenizer`가 자동 사용된다.

### training.*

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `training.epochs` | int | conditional | null | 학습 에폭 수. `epochs` 또는 `max_steps` 중 하나 필수 |
| `training.max_steps` | int | conditional | null | 최대 스텝 수 |
| `training.precision` | str | no | `"fp32"` | `fp32`, `fp16`, `bf16`, `fp8` |
| `training.gradient_accumulation_steps` | int | no | `1` | 그래디언트 누적 스텝 |
| `training.gradient_clip_max_norm` | float | no | null | 그래디언트 클리핑 |
| `training.gradient_checkpointing` | bool | no | `false` | 활성화 체크포인팅 (메모리 절약) |
| `training.val_check_interval` | float | no | `1.0` | 검증 주기. `val_check_unit`에 따라 해석 |
| `training.val_check_unit` | str | no | `"epoch"` | 검증 단위: `"epoch"` (0.5→에폭당 2회, 2→매 2에폭) 또는 `"step"` (500→매 500 step) |
| `training.compile` | str \| bool | no | `false` | torch.compile 모드 |

### evaluation.*

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `evaluation.metrics` | list[str] | no | `[]` | 평가 메트릭 목록 |

### generation.* (추론 전용)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `generation.max_new_tokens` | int | no | `256` | 최대 생성 토큰 수 |
| `generation.temperature` | float | no | `1.0` | 샘플링 온도 |
| `generation.top_p` | float | no | `1.0` | nucleus sampling |
| `generation.top_k` | int | no | `50` | top-k sampling |
| `generation.do_sample` | bool | no | `true` | 샘플링 사용 여부 |
| `generation.num_beams` | int | no | `1` | 빔 서치 빔 수 |
| `generation.repetition_penalty` | float | no | `1.0` | 반복 페널티 |

### monitoring.*

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `monitoring.enabled` | bool | no | `false` | 모니터링 활성화 |
| `monitoring.baseline` | dict | no | `{}` | baseline 설정 |
| `monitoring.drift` | dict | no | `{}` | drift 감지 설정 |

### metadata.*

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `metadata.author` | str | yes | - | 작성자 |
| `metadata.description` | str | yes | - | 실험 설명 |

### `_component_` Pattern

모든 플러그인(optimizer, scheduler, loss, dataset, head, callback)은 `_component_` 패턴을 사용한다.

```yaml
# alias 사용 (내장 컴포넌트)
optimizer:
  _component_: AdamW
  lr: 0.001
  weight_decay: 0.01

# 풀 경로 사용 (커스텀 컴포넌트)
optimizer:
  _component_: my_package.MyCustomOptimizer
  lr: 0.001
```

`type:` 키는 지원하지 않는다. 반드시 `_component_` 를 사용할 것.

### Alias Table

#### Callback

| Alias | Full Path |
|-------|-----------|
| `EarlyStopping` | `mdp.training.callbacks.early_stopping.EarlyStopping` |
| `ModelCheckpoint` | `mdp.training.callbacks.checkpoint.ModelCheckpoint` |
| `LRMonitor` | `mdp.training.callbacks.lr_monitor.LearningRateMonitor` |
| `ProgressBar` | `mdp.training.callbacks.progress.ProgressBar` |
| `EMACallback` | `mdp.training.callbacks.ema.EMACallback` |

#### Head

| Alias | Full Path |
|-------|-----------|
| `ClassificationHead` | `mdp.models.heads.classification.ClassificationHead` |
| `DetectionHead` | `mdp.models.heads.detection.DetectionHead` |
| `CausalLMHead` | `mdp.models.heads.causal_lm.CausalLMHead` |
| `SegmentationHead` | `mdp.models.heads.segmentation.SegmentationHead` |
| `TokenClassificationHead` | `mdp.models.heads.token_classification.TokenClassificationHead` |
| `Seq2SeqLMHead` | `mdp.models.heads.seq2seq_lm.Seq2SeqLMHead` |
| `DualEncoderHead` | `mdp.models.heads.dual_encoder.DualEncoderHead` |
| `SequenceClassificationHead` | `mdp.models.heads.classification.ClassificationHead` |

#### Dataset

데이터셋은 `_component_` 패턴을 사용하지 않는다. `data.source`에 HF Hub 이름 또는 로컬 경로를 지정하면 `datasets.load_dataset()`로 자동 로딩된다.

#### Strategy

| Alias | Full Path |
|-------|-----------|
| `DDPStrategy` | `mdp.training.strategies.ddp.DDPStrategy` |
| `FSDPStrategy` | `mdp.training.strategies.fsdp.FSDPStrategy` |
| `DeepSpeedStrategy` | `mdp.training.strategies.deepspeed.DeepSpeedStrategy` |

---

## Config YAML Schema

### Top-Level Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `environment` | dict | no | `{name: "local"}` | 환경 정보 |
| `compute` | object | no | defaults | 실행 환경 설정 |
| `environment_setup` | object | no | defaults | 원격/클라우드 환경 준비 |
| `mlflow` | object | no | defaults | MLflow 실험 추적 |
| `storage` | object | no | defaults | 체크포인트/출력 저장소 |
| `serving` | object | no | null | 서빙 설정 |
| `job` | object | no | defaults | 작업 제어 |

### compute.*

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `compute.target` | str | no | `"local"` | `local`, `remote`, `multi-node`, `cloud` |
| `compute.gpus` | int \| str \| list[int] | no | `"auto"` | GPU 수 또는 ID 리스트 |
| `compute.host` | str | no | null | 원격 호스트 (remote/multi-node 시) |
| `compute.user` | str | no | null | SSH 사용자 |
| `compute.ssh_key` | str | no | null | SSH 키 경로 |
| `compute.working_dir` | str | no | null | 원격 작업 디렉토리 |
| `compute.nodes` | list[dict] | no | null | multi-node 노드 목록 |
| `compute.distributed` | dict | no | null | 분산 전략 설정 |
| `compute.provider` | str | no | null | 클라우드 프로바이더 (cloud 시) |
| `compute.accelerators` | str | no | null | 가속기 타입 (cloud 시) |
| `compute.spot` | bool | no | `false` | spot 인스턴스 사용 |
| `compute.region` | str | no | null | 리전 |
| `compute.disk_size` | int | no | `256` | 디스크 크기 (GB) |

### Compute Target x Distributed Strategy Matrix

| Target | ddp | fsdp | deepspeed_zero2 | deepspeed_zero3 |
|--------|-----|------|-----------------|-----------------|
| local (1 GPU) | N/A | N/A | N/A | N/A |
| local (2+ GPU) | OK | OK | OK | OK |
| remote | OK | OK | OK | OK |
| multi-node | OK | OK | OK | OK |
| cloud | OK | OK | OK | OK |

### environment_setup.*

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `environment_setup.container` | str | no | null | Docker 이미지 |
| `environment_setup.dependencies` | str | no | null | requirements 파일 경로 |
| `environment_setup.setup_commands` | list[str] | no | `[]` | 환경 설정 명령어 |

### mlflow.*

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `mlflow.tracking_uri` | str | no | `"./mlruns"` | MLflow 트래킹 URI |
| `mlflow.experiment_name` | str | no | `"default"` | 실험 이름 |

### storage.*

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `storage.checkpoint_dir` | str | no | `"./checkpoints"` | 체크포인트 저장 경로 |
| `storage.checkpoint_every_n_steps` | int | no | null | 체크포인트 저장 주기 |
| `storage.output_dir` | str | no | `"./outputs"` | 출력 디렉토리 |

### serving.*

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `serving.backend` | str | no | `"torchserve"` | 서빙 백엔드 |
| `serving.model_repository` | str | no | null | 모델 리포지토리 경로 |
| `serving.max_batch_size` | int | no | `1` | 최대 배치 크기 |
| `serving.instance_count` | int | no | `1` | 인스턴스 수 |

### job.*

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `job.name` | str | no | null | 작업 이름 |
| `job.resume` | str | no | `"auto"` | 재개 모드 (`auto`, 체크포인트 경로) |
| `job.max_retries` | int | no | `0` | 최대 재시도 횟수 |

---

## Available Models by Task

### image_classification

| Model | class_path | Recommended pretrained | Params (M) | Memory (fp16 GB) |
|-------|-----------|----------------------|--------|-------------|
| ViT-Base | `timm.create_model` | `timm://vit_base_patch16_224` | 86.6 | 0.17 |
| ViT-Large | `timm.create_model` | `timm://vit_large_patch16_224` | 304.3 | 0.58 |
| ResNet-50 | `torchvision.models.resnet50` | `timm://resnet50` | 25.6 | 0.05 |
| Swin-Base | `timm.create_model` | `timm://swin_base_patch4_window7_224` | 87.8 | 0.17 |
| ConvNeXt-Base | `timm.create_model` | `timm://convnext_base.fb_in22k_ft_in1k` | 88.6 | 0.17 |
| EfficientNet-B0 | `timm.create_model` | `timm://efficientnet_b0.ra_in1k` | 5.3 | 0.01 |

### object_detection

| Model | class_path | Recommended pretrained | Params (M) | Memory (fp16 GB) |
|-------|-----------|----------------------|--------|-------------|
| DETR-ResNet50 | `transformers.AutoModelForObjectDetection` | `hf://facebook/detr-resnet-50` | 41.3 | 0.08 |
| YOLOv8n | `ultralytics.YOLO` | `ultralytics://yolov8n.pt` | 3.2 | 0.01 |

### semantic_segmentation

| Model | class_path | Recommended pretrained | Params (M) | Memory (fp16 GB) |
|-------|-----------|----------------------|--------|-------------|
| SegFormer-B0 | `transformers.AutoModelForSemanticSegmentation` | `hf://nvidia/segformer-b0-finetuned-ade-512-512` | 3.7 | 0.01 |
| Swin-Base | `timm.create_model` | `timm://swin_base_patch4_window7_224` | 87.8 | 0.17 |

### text_classification

| Model | class_path | Recommended pretrained | Params (M) | Memory (fp16 GB) |
|-------|-----------|----------------------|--------|-------------|
| BERT-Base | `transformers.AutoModel` | `hf://bert-base-uncased` | 110.0 | 0.21 |
| RoBERTa-Base | `transformers.AutoModel` | `hf://roberta-base` | 124.6 | 0.24 |

### token_classification

| Model | class_path | Recommended pretrained | Params (M) | Memory (fp16 GB) |
|-------|-----------|----------------------|--------|-------------|
| BERT-Base | `transformers.AutoModel` | `hf://bert-base-uncased` | 110.0 | 0.21 |
| RoBERTa-Base | `transformers.AutoModel` | `hf://roberta-base` | 124.6 | 0.24 |

### text_generation

| Model | class_path | Recommended pretrained | Params (M) | Memory (fp16 GB) |
|-------|-----------|----------------------|--------|-------------|
| GPT-2 | `transformers.AutoModelForCausalLM` | `hf://gpt2` | 124.4 | 0.24 |
| Phi-3-Mini | `transformers.AutoModelForCausalLM` | `hf://microsoft/Phi-3-mini-4k-instruct` | 3821 | 7.6 |
| Mistral-7B | `transformers.AutoModelForCausalLM` | `hf://mistralai/Mistral-7B-v0.3` | 7248 | 14.5 |
| Qwen2.5-7B | `transformers.AutoModelForCausalLM` | `hf://Qwen/Qwen2.5-7B` | 7616 | 15.2 |
| Llama-3-8B | `transformers.AutoModelForCausalLM` | `hf://meta-llama/Meta-Llama-3-8B` | 8030 | 16.1 |
| Gemma-2-9B | `transformers.AutoModelForCausalLM` | `hf://google/gemma-2-9b` | 9242 | 18.5 |
| Llama-3-70B | `transformers.AutoModelForCausalLM` | `hf://meta-llama/Meta-Llama-3-70B` | 70554 | 141.1 |
| Qwen2.5-72B | `transformers.AutoModelForCausalLM` | `hf://Qwen/Qwen2.5-72B` | 72710 | 145.4 |
| Florence-2-Base | `transformers.AutoModelForCausalLM` | `hf://microsoft/Florence-2-base` | 232 | 0.44 |
| BLIP-2-OPT-2.7B | `transformers.Blip2ForConditionalGeneration` | `hf://Salesforce/blip2-opt-2.7b` | 3770 | 7.5 |
| LLaVA-1.5-7B | `transformers.LlavaForConditionalGeneration` | `hf://llava-hf/llava-1.5-7b-hf` | 7063 | 14.1 |

### seq2seq

| Model | class_path | Recommended pretrained | Params (M) | Memory (fp16 GB) |
|-------|-----------|----------------------|--------|-------------|
| T5-Base | `transformers.AutoModelForSeq2SeqLM` | `hf://t5-base` | 222.9 | 0.42 |
| T5-Large | `transformers.AutoModelForSeq2SeqLM` | `hf://t5-large` | 737.7 | 1.40 |

### feature_extraction

| Model | class_path | Recommended pretrained | Params (M) | Memory (fp16 GB) |
|-------|-----------|----------------------|--------|-------------|
| CLIP-ViT-B/32 | `transformers.CLIPModel` | `hf://openai/clip-vit-base-patch32` | 151.3 | 0.29 |
| CLIP-ViT-L/14 | `transformers.CLIPModel` | `hf://openai/clip-vit-large-patch14` | 427.6 | 0.81 |
| SigLIP-Base | `transformers.AutoModel` | `hf://google/siglip-base-patch16-224` | 203 | 0.39 |
| DINOv2-Base | `transformers.AutoModel` | `hf://facebook/dinov2-base` | 86.6 | 0.17 |

---

## Validation Rules

### Task-Head Compatibility

- `image_classification` -> `ClassificationHead` (또는 `AutoModelForImageClassification` 사용 시 head 생략)
- `object_detection` -> `DetectionHead` (또는 `AutoModelForObjectDetection` 사용 시 head 생략)
- `semantic_segmentation` -> `SegmentationHead` (또는 `AutoModelForSemanticSegmentation` 사용 시 head 생략)
- `text_classification` -> `ClassificationHead` (또는 `AutoModelForSequenceClassification` 사용 시 head 생략)
- `token_classification` -> `TokenClassificationHead` 또는 `ClassificationHead` (또는 `AutoModelForTokenClassification` 사용 시 head 생략)
- `text_generation` -> `CausalLMHead` (또는 `AutoModelForCausalLM` 사용 시 head 생략)
- `seq2seq` -> `Seq2SeqLMHead` (또는 `AutoModelForSeq2SeqLM` 사용 시 head 생략)
- `image_generation` -> head 생략 권장 (모델 내장)

> `vision_language`는 별도 task가 아니다. multimodal 모델(CLIP, LLaVA, Florence-2 등)은 `text_generation` 또는 `feature_extraction` task에 `fields: {image: ..., text: ...}` 조합으로 표현한다.

### Adapter Constraints

- `adapter.method: qlora` -> `quantization.bits` 필수
- `adapter.method: qlora` -> `model.torch_dtype`는 `bfloat16` 또는 `float16`이어야 함
- `adapter.method: lora/qlora` -> `r` 필수
- `adapter.method: prefix_tuning` -> `r` 필수 (prefix 길이 = 가상 토큰 수로 해석)

### Precision Constraints

- `training.precision: bf16` -> Ampere 이상 GPU 필요 (A100, H100, RTX 3090+)
- `training.precision: fp8` -> Hopper 이상 GPU 필요 (H100)
- `model.attn_implementation: flash_attention_2` -> Ampere 이상 GPU 필요

### Distributed Constraints

- `compute.gpus: 1` -> `distributed` 섹션 불필요 (있으면 무시)
- `distributed.strategy: fsdp` + `distributed.cpu_offload: true` -> 학습 속도 크게 저하
- `distributed.strategy: deepspeed_zero3` + `offload_params: true` -> 가장 느리지만 메모리 최소

### Data Constraints

- `data.source` 필수 (HF Hub 이름 또는 로컬 경로)
- `data.tokenizer` -> 언어 태스크(`text_*`, `seq2seq`)에서만 의미 있음
- `data.augmentation` -> 비전 태스크에서 주로 사용, 언어 태스크에서는 일반적으로 불필요
- `data.fields` roles는 task의 `TASK_PRESETS.required_fields`와 일치해야 함 (BusinessValidator가 검증)

---

## Common Pitfalls

### 1. loss 섹션 생략 시 모델이 training_step()을 구현해야 함

HuggingFace 모델(`AutoModelForCausalLM` 등)은 내부에 loss 계산이 내장되어 있으므로 loss 생략 가능. 커스텀 모델은 loss를 반드시 지정하거나 `training_step()`에서 직접 계산해야 한다. 징후: "모델이 loss를 반환하지 않습니다" 에러.

### 2. DDP에서 drop_last: false로 설정

DDP는 모든 GPU가 동일한 배치 수를 처리해야 한다. 마지막 불완전 배치가 있으면 GPU 간 동기화 실패로 행(hang). 해결: `drop_last: true` (기본값이 true이므로 명시적으로 false로 바꾸지 말 것).

### 3. LoRA target_modules를 모델에 맞지 않게 지정

모델마다 어텐션 레이어 이름이 다르다:
- ViT: `qkv`, `proj`
- GPT-2: `c_attn`, `c_proj`
- LLaMA/Qwen/Mistral: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- CLIP: `q_proj`, `v_proj`

존재하지 않는 모듈 이름을 넣으면 에러 없이 LoRA가 적용되지 않는다. 안전한 선택: `target_modules: "all_linear"`.

### 4. gradient_accumulation과 실질 배치 크기 혼동

실질 배치 = `batch_size` x `gpus` x `gradient_accumulation_steps`. 예: `batch_size=8, gpus=4, accumulation=2` -> 실질 배치 = 64. LR은 실질 배치에 비례하여 조정해야 함 (linear scaling rule).

### 5. warmup_steps와 warmup_ratio 동시 지정

둘 다 지정하면 검증 에러. `warmup_steps`: 절대 step 수. `warmup_ratio`: 전체 step 대비 비율. 하나만 사용.

### 6. 대형 모델을 torch_dtype 없이 로드

7B 이상 모델을 float32로 로드하면 메모리 초과. 해결: `model.torch_dtype: bfloat16` (또는 `float16`).

### 7. vision 태스크에서 tokenizer 설정

비전 전용 태스크(`image_classification`, `object_detection`, `semantic_segmentation`)에서는 tokenizer가 무시된다. 불필요한 설정이므로 제거하는 것이 깔끔함. 단, `image_generation`은 text conditioning을 위해 tokenizer가 필요하다.

### 8. checkpoint_dir과 MLflow artifact 혼동

`checkpoint_dir`: 학습 중 주기적으로 저장하는 체크포인트 (재개용). MLflow: 학습 완료 후 best model만 저장 (실험 비교 및 복원용). 추론 시 사용할 모델은 MLflow `run_id`로 접근하는 것이 올바름.

### 9. 모든 플러그인은 `_component_` 패턴을 사용한다

`type:` 키는 없다. 내장 컴포넌트는 짧은 alias(`AdamW`, `ImageFolder`)를, 커스텀 컴포넌트는 풀 경로(`my.module.MyClass`)를 사용한다.

```yaml
# 올바른 사용
head:
  _component_: ClassificationHead
  num_classes: 10

# 잘못된 사용 -- 지원하지 않음
head:
  type: classification
  num_classes: 10
```

---

## Complete Working Examples

### Example 1: ViT LoRA Image Classification (Local Single GPU)

**Recipe** (`recipes/vit-lora-cifar10.yaml`):

```yaml
name: vit-lora-cifar10
task: image_classification

model:
  class_path: timm.create_model
  pretrained: timm://vit_base_patch16_224

head:
  _component_: ClassificationHead
  num_classes: 10

adapter:
  method: lora
  r: 16
  alpha: 32
  target_modules: ["qkv", "proj"]

data:
  source: cifar10
  fields:
    image: img
    label: label
  augmentation:
    train:
      steps:
        - {type: RandomResizedCrop, params: {size: [224, 224]}}
        - {type: RandomHorizontalFlip}
        - {type: ToDtype, params: {dtype: float32, scale: true}}
        - {type: Normalize, params: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}}
    val:
      steps:
        - {type: Resize, params: {size: [256, 256]}}
        - {type: CenterCrop, params: {size: [224, 224]}}
        - {type: ToDtype, params: {dtype: float32, scale: true}}
        - {type: Normalize, params: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}}
  dataloader:
    batch_size: 64
    num_workers: 4

training:
  epochs: 30
  precision: bf16
  gradient_clip_max_norm: 1.0

optimizer:
  _component_: AdamW
  lr: 0.001
  weight_decay: 0.01

scheduler:
  _component_: CosineAnnealingLR
  T_max: 30

loss:
  _component_: CrossEntropyLoss

evaluation:
  metrics: [accuracy, f1]

callbacks:
  - _component_: EarlyStopping
    monitor: val_loss
    patience: 5
  - _component_: ModelCheckpoint
    monitor: val_accuracy
    mode: max

metadata:
  author: agent
  description: ViT-Base LoRA fine-tuning on CIFAR-10
```

**Config** (`configs/local-1gpu.yaml`):

```yaml
compute:
  target: local
  gpus: 1

mlflow:
  tracking_uri: ./mlruns
  experiment_name: vit-cifar10-experiments

storage:
  checkpoint_dir: ./checkpoints/vit-lora-cifar10
  output_dir: ./outputs/vit-lora-cifar10
```

### Example 2: GPT-2 Text Generation Fine-tuning (Remote 4GPU DDP)

**Recipe** (`recipes/gpt2-text-gen.yaml`):

```yaml
name: gpt2-wikitext-finetune
task: text_generation

model:
  class_path: transformers.AutoModelForCausalLM
  pretrained: hf://gpt2
  # AutoModelForCausalLM은 head가 내장되어 있으므로 head 생략

data:
  source: wikitext
  subset: wikitext-103-raw-v1
  fields:
    text: text
  tokenizer:
    pretrained: gpt2
    max_length: 1024
  dataloader:
    batch_size: 8
    num_workers: 4

training:
  epochs: 5
  precision: bf16
  gradient_accumulation_steps: 4
  gradient_clip_max_norm: 1.0

optimizer:
  _component_: AdamW
  lr: 5.0e-5
  weight_decay: 0.01

scheduler:
  _component_: CosineAnnealingWarmRestarts
  T_0: 1000

evaluation:
  metrics: [perplexity]

callbacks:
  - _component_: ModelCheckpoint
    monitor: val_loss
    mode: min

metadata:
  author: agent
  description: GPT-2 full fine-tuning on WikiText-103
```

**Config** (`configs/remote-4gpu.yaml`):

```yaml
compute:
  target: remote
  gpus: 4
  host: gpu-server.example.com
  user: trainer
  ssh_key: ~/.ssh/id_rsa
  distributed:
    strategy: ddp

environment_setup:
  container: nvcr.io/nvidia/pytorch:24.01-py3

mlflow:
  tracking_uri: http://mlflow.example.com:5000
  experiment_name: gpt2-wikitext

storage:
  checkpoint_dir: /data/checkpoints/gpt2-wikitext
  output_dir: /data/outputs/gpt2-wikitext
```

### Example 3: Llama-3-8B QLoRA (Local Single GPU)

**Recipe** (`recipes/llama3-qlora.yaml`):

```yaml
name: llama3-qlora-instruct
task: text_generation

model:
  class_path: transformers.AutoModelForCausalLM
  pretrained: hf://meta-llama/Meta-Llama-3-8B
  torch_dtype: bfloat16  # 8B 모델은 반드시 dtype 지정
  attn_implementation: flash_attention_2

adapter:
  method: qlora
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  quantization:
    bits: 4

data:
  source: tatsu-lab/alpaca
  fields:
    text: instruction
  tokenizer:
    pretrained: meta-llama/Meta-Llama-3-8B
    max_length: 2048
  dataloader:
    batch_size: 4
    num_workers: 2

training:
  epochs: 3
  precision: bf16
  gradient_accumulation_steps: 8
  gradient_checkpointing: true  # 메모리 절약

optimizer:
  _component_: AdamW
  lr: 2.0e-4
  weight_decay: 0.01

scheduler:
  _component_: CosineAnnealingLR
  T_max: 3

evaluation:
  metrics: [perplexity]

generation:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  do_sample: true

callbacks:
  - _component_: ModelCheckpoint
    monitor: val_loss
    mode: min

metadata:
  author: agent
  description: Llama-3-8B QLoRA instruction tuning
```

**Config** (`configs/local-1gpu-a100.yaml`):

```yaml
compute:
  target: local
  gpus: 1

mlflow:
  tracking_uri: ./mlruns
  experiment_name: llama3-qlora

storage:
  checkpoint_dir: ./checkpoints/llama3-qlora
  output_dir: ./outputs/llama3-qlora
```

### Example 4: CLIP Vision-Language (Cloud 8GPU FSDP)

**Recipe** (`recipes/clip-finetune.yaml`):

```yaml
name: clip-custom-dataset
task: feature_extraction

model:
  class_path: transformers.CLIPModel
  pretrained: hf://openai/clip-vit-large-patch14

head:
  _component_: DualEncoderHead
  hidden_dim: 768

adapter:
  method: lora
  r: 8
  alpha: 16
  target_modules: ["q_proj", "v_proj"]

data:
  source: conceptual_captions
  fields:
    image: image_url
    text: caption
  tokenizer:
    pretrained: openai/clip-vit-large-patch14
    max_length: 77
  augmentation:
    train:
      steps:
        - {type: RandomResizedCrop, params: {size: [224, 224]}}
        - {type: RandomHorizontalFlip}
        - {type: ToDtype, params: {dtype: float32, scale: true}}
  dataloader:
    batch_size: 32
    num_workers: 8

training:
  epochs: 10
  precision: bf16
  gradient_accumulation_steps: 2

optimizer:
  _component_: AdamW
  lr: 1.0e-4
  weight_decay: 0.01

scheduler:
  _component_: CosineAnnealingLR
  T_max: 10

callbacks:
  - _component_: ModelCheckpoint
    monitor: val_loss
    mode: min

metadata:
  author: agent
  description: CLIP LoRA fine-tuning on Conceptual Captions
```

**Config** (`configs/cloud-8gpu.yaml`):

```yaml
compute:
  target: cloud
  gpus: 8
  provider: gcp
  accelerators: A100:8
  region: us-central1
  disk_size: 512
  distributed:
    strategy: fsdp

environment_setup:
  container: nvcr.io/nvidia/pytorch:24.01-py3
  dependencies: requirements.txt

mlflow:
  tracking_uri: http://mlflow.internal:5000
  experiment_name: clip-custom

storage:
  checkpoint_dir: gs://ml-checkpoints/clip-custom
  output_dir: gs://ml-outputs/clip-custom
```

---

## CLI Output Format (`--format json`)

모든 명령은 `--format json` 글로벌 옵션을 지원한다. JSON 모드에서는 Rich 테이블, progress bar, 중간 로그가 stderr로 리디렉션되고, stdout에는 구조화된 JSON만 출력된다.

### Common Wrapper

```json
{
  "status": "success | error",
  "command": "train | inference | estimate | list | version",
  "timestamp": "2026-03-31T14:30:00+00:00",
  "error": null
}
```

### Error Response

```json
{
  "status": "error",
  "command": "train",
  "timestamp": "2026-03-31T14:30:00+00:00",
  "error": {
    "type": "ValidationError",
    "message": "head._component_ 'ClassificationHead'은 task 'text_generation'과 호환되지 않습니다",
    "details": {
      "field": "head._component_",
      "value": "ClassificationHead",
      "allowed": ["CausalLMHead"]
    }
  }
}
```

에러 유형별 복구 전략:
- `ValidationError` -> YAML 수정
- `RuntimeError` -> Config (GPU 수, 메모리 설정) 조정
- `TimeoutError` -> 재시도

### `mdp train --format json`

```json
{
  "status": "success",
  "command": "train",
  "timestamp": "2026-03-31T15:30:00+00:00",
  "error": null,
  "run_id": "a1b2c3d4e5f6",
  "experiment_name": "vit-cifar10-experiments",
  "checkpoint_dir": "/checkpoints/vit-lora-cifar10/best",
  "output_dir": "./outputs/vit-lora-cifar10",
  "metrics": {
    "final": {
      "train_loss": 0.032,
      "val_loss": 0.118,
      "val_accuracy": 0.952,
      "val_f1": 0.948
    },
    "best": {
      "epoch": 24,
      "val_accuracy": 0.956,
      "val_loss": 0.105
    }
  },
  "training_summary": {
    "total_epochs": 30,
    "stopped_epoch": 27,
    "stopped_reason": "early_stopping (patience=5, monitor=val_loss)",
    "duration_seconds": 3600,
    "samples_per_second": 842.3
  },
  "mlflow": {
    "tracking_uri": "http://mlflow.example.com:5000",
    "experiment_name": "vit-cifar10-experiments",
    "run_id": "a1b2c3d4e5f6",
    "artifact_uri": "s3://mlflow-artifacts/a1b2c3d4e5f6/artifacts"
  },
  "monitoring": {
    "baseline_saved": true,
    "baseline_path": "artifacts/monitoring/baseline.json"
  }
}
```

Key fields for agent decision-making:
- `run_id`: 이후 추론, MLflow 비교, 모델 참조에 사용
- `metrics.best.val_accuracy`: 이전 실험과 비교하여 개선 여부 판단
- `training_summary.stopped_reason`: `early_stopping`이면 정상, `max_epochs`이면 추가 학습 고려
- `monitoring.baseline_saved`: true여야 이후 drift 감지 가능
- `checkpoint_dir`: 추론 시 체크포인트 경로로 사용

### `mdp inference --format json`

```json
{
  "status": "success",
  "command": "inference",
  "timestamp": "2026-03-31T16:00:00+00:00",
  "error": null,
  "output_path": "/outputs/predictions/result.parquet",
  "rows": 5000,
  "duration_seconds": 120,
  "metrics": {
    "accuracy": 0.943,
    "f1": 0.938
  },
  "task": "classification",
  "monitoring": {
    "drift_detected": true,
    "drift_score": 0.15,
    "drift_threshold": 0.10,
    "prediction_distribution": {
      "entropy_mean": 2.31,
      "entropy_std": 0.45,
      "baseline_entropy_mean": 1.82,
      "baseline_entropy_std": 0.38
    },
    "class_distribution_shift": {
      "kl_divergence": 0.087,
      "top_shifted_classes": [
        {"class_id": 3, "train_ratio": 0.12, "inference_ratio": 0.23},
        {"class_id": 7, "train_ratio": 0.08, "inference_ratio": 0.02}
      ]
    },
    "confidence_analysis": {
      "mean_confidence": 0.78,
      "low_confidence_ratio": 0.15,
      "baseline_mean_confidence": 0.89,
      "baseline_low_confidence_ratio": 0.05
    },
    "alerts": [
      "prediction_entropy elevated: 2.31 vs baseline 1.82 (threshold: 2s)",
      "class_3 distribution shift: 12% -> 23%",
      "low_confidence_ratio increased: 5% -> 15%"
    ]
  }
}
```

Drift severity levels for agent decision:
- **Level 1 (Watch)**: `drift_score` at 50-100% of threshold, `drift_detected: false`. Increase monitoring frequency.
- **Level 2 (Alert)**: `drift_detected: true`, single alert. Try minor adjustments (LR, data filtering).
- **Level 3 (Retrain)**: `drift_detected: true`, multiple alerts. Full retraining with updated data.

### `mdp estimate --format json`

```json
{
  "status": "success",
  "command": "estimate",
  "timestamp": "2026-03-31T14:00:00+00:00",
  "error": null,
  "model_info": {
    "class_path": "transformers.AutoModelForCausalLM",
    "pretrained": "hf://Qwen/Qwen2.5-7B",
    "params_total": 7616000000,
    "params_trainable": 4194304,
    "adapter": "lora (r=16)"
  },
  "memory_estimate": {
    "precision": "bf16",
    "model_memory_gb": 15.2,
    "gradient_memory_gb": 0.008,
    "optimizer_memory_gb": 0.016,
    "activation_memory_gb_estimate": 2.5,
    "total_memory_gb": 17.72
  },
  "recommendation": {
    "suggested_gpus": 1,
    "suggested_strategy": "none",
    "gpu_memory_assumption_gb": 80,
    "fits_single_gpu": true,
    "notes": [
      "LoRA adapter로 학습 가능 파라미터가 4.2M (전체의 0.06%)",
      "A100 80GB 1대로 충분. batch_size=8 기준"
    ]
  }
}
```

### `mdp list models --format json`

```json
{
  "status": "success",
  "command": "list",
  "timestamp": "2026-03-31T14:00:00+00:00",
  "error": null,
  "what": "models",
  "models": [
    {
      "name": "vit-base-patch16-224",
      "family": "vit",
      "class_path": "timm.create_model",
      "tasks": ["image_classification", "feature_extraction"],
      "pretrained": ["timm://vit_base_patch16_224", "hf://google/vit-base-patch16-224"],
      "params_million": 86.6,
      "compatible_heads": ["ClassificationHead"],
      "compatible_adapters": ["lora", "qlora"]
    }
  ]
}
```

### `mdp list tasks --format json`

```json
{
  "status": "success",
  "command": "list",
  "timestamp": "2026-03-31T14:00:00+00:00",
  "error": null,
  "what": "tasks",
  "tasks": [
    {
      "name": "image_classification",
      "modality": "vision",
      "default_metrics": ["accuracy", "f1"],
      "compatible_heads": ["ClassificationHead"],
      "description": "이미지를 카테고리로 분류"
    }
  ]
}
```
