# Extending MDP

커스텀 모델, 콜백, 데이터셋, 손실 함수 등을 작성하여 MDP를 확장하는 방법.

## 원칙

MDP의 모든 pluggable component는 `_component_` 패턴으로 주입된다. 레지스트리 등록이 필요 없으며, Python 모듈을 작성하고 import 경로를 YAML에 지정하면 된다.

**import 경로 설정 방법**:
- 프로젝트 루트에서 실행: `sys.path`에 자동 포함
- `pip install -e .`: site-packages에 등록
- `PYTHONPATH`: `PYTHONPATH=/path mdp train ...`

---

## 커스텀 모델

`BaseModel`을 상속하여 4개 메서드를 구현한다:

```python
# my_models/custom.py
import torch
import torch.nn as nn
from mdp.models.base import BaseModel

class MyModel(BaseModel):
    def __init__(self, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.encoder = nn.Linear(768, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """순전파. 최소 "logits" 키 반환 필수."""
        x = self.encoder(batch["pixel_values"])
        return {"logits": self.head(x)}

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """스칼라 loss 반환. loss: 섹션 생략 시 호출됨."""
        out = self.forward(batch)
        return nn.functional.cross_entropy(out["logits"], batch["labels"])

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """메트릭 이름 → 값 dict 반환."""
        out = self.forward(batch)
        pred = out["logits"].argmax(dim=-1)
        acc = (pred == batch["labels"]).float().mean().item()
        return {"accuracy": acc}

    # 선택 메서드:
    def generate(self, batch: dict, **kwargs) -> dict:
        """autoregressive 생성 (텍스트 모델에서 구현)."""
        ...

    def configure_optimizers(self) -> dict | None:
        """모델 전용 옵티마이저. Recipe의 optimizer를 override."""
        return {"optimizer": torch.optim.Adam(self.parameters(), lr=1e-4)}
```

Recipe에서 사용:
```yaml
model:
  _component_: my_models.custom.MyModel
  hidden_dim: 512
  num_classes: 100
```

### Pretrained backbone 조합

커스텀 BaseModel이 외부 pretrained 모델을 backbone으로 사용할 수 있다. Factory는 클래스에 `from_pretrained` 메서드가 없으면 `pretrained` 키를 생성자 인자로 전달한다 — 클래스가 내부에서 backbone을 로드하고 그 위에 자신의 컴포넌트를 얹는 패턴이다:

```python
# my_models/value_model.py
from transformers import AutoModel
from mdp.models.base import BaseModel

class ValueModel(BaseModel):
    def __init__(self, pretrained: str, value_head_dropout: float = 0.2, **kwargs):
        super().__init__()
        # URI 스킴 제거
        name = pretrained.removeprefix("hf://").removeprefix("local://")
        # backbone 로딩 (아키텍처는 불변 — 이 클래스가 적응한다)
        self.backbone = AutoModel.from_pretrained(name)
        hidden_dim = self.backbone.config.hidden_size  # backbone에 물어본다
        # 자신의 컴포넌트는 backbone 차원에 맞춰 구성
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(value_head_dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, batch):
        out = self.backbone(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
        return {"logits": self.value_head(out.last_hidden_state)}

    def training_step(self, batch): ...
    def validation_step(self, batch): ...
```

```yaml
model:
  _component_: my_models.value_model.ValueModel
  pretrained: hf://meta-llama/Meta-Llama-3-8B
  value_head_dropout: 0.2
```

**로딩 동작 정리** (Factory가 `_component_`와 `pretrained`를 어떻게 해석하는가):

| `_component_` | `pretrained` | 동작 |
|:-:|:-:|---|
| 없음 | 있음 | PretrainedResolver가 `config.architectures[0]`에서 클래스 추론 (CLI `--pretrained` 경로) |
| HF 클래스 (`from_pretrained` 있음) | 있음 | `klass.from_pretrained(identifier, **kwargs)` |
| 커스텀 클래스 (`from_pretrained` 없음) | 있음 | `klass(pretrained=uri, **kwargs)` — 위 예시 |
| 있음 | 없음 | `klass(**kwargs)` — 랜덤 초기화 (BaseModel 상속 필수) |

**제약**: pretrained backbone의 아키텍처(`hidden_size`, `num_layers` 등)는 불변이다. 커스텀 클래스는 `backbone.config`를 읽어 적응해야지, 치수를 하드코딩하면 다른 모델로 교체할 때 깨진다.

### configure_optimizers() — 파라미터 그룹별 LR

Recipe의 `optimizer:` 필드는 모든 trainable 파라미터에 단일 LR을 적용한다. **파라미터 그룹마다 다른 LR이 필요하면** `configure_optimizers()`를 오버라이드한다 — 가장 흔한 케이스는 pretrained backbone(낮은 LR)과 랜덤 초기화 head(높은 LR)의 조합이다.

```python
def configure_optimizers(self) -> dict | None:
    # pretrained LoRA 파라미터: 이미 학습된 표현을 보존 → 낮은 LR
    lora_params = [
        p for n, p in self.backbone.named_parameters()
        if "lora_" in n and p.requires_grad
    ]
    # 랜덤 초기화 head: 처음부터 학습 → 높은 LR
    head_params = list(self.value_head.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": 2e-5},
            {"params": head_params, "lr": 2e-4},
        ],
        weight_decay=0.01,
    )
    return {"optimizer": optimizer}  # ← dict 형식 필수
```

**반환 형식 주의**: trainer가 `isinstance(result, dict) and "optimizer" in result`로 판별한다. optimizer 객체를 직접 반환하면 조건 실패 → recipe fallback(`optimizer: lr`) 사용 → 오류 없이 의도와 다르게 동작한다.

| 언제 쓰는가 | 예시 |
|---|---|
| backbone(pretrained) + head(scratch) | LoRA 2e-5 + value_head 2e-4 |
| layer-wise LR decay | 하위 레이어 1e-5, 상위 레이어 5e-5 |
| 특정 파라미터 weight_decay 제외 | bias, LayerNorm 에 weight_decay=0 |

`configure_optimizers()`를 정의하면 recipe의 `optimizer:` 전체(LR, weight_decay 포함)가 무시된다. `scheduler:`는 별도로 recipe에서 그대로 적용된다.

### Loss 선택 분기

- `loss:` 지정 → `model.forward(batch)` 호출 → `loss_fn(outputs["logits"], batch["labels"])`
- `loss:` 생략 → `model.training_step(batch)` 호출 → 반환값을 loss로 사용

HuggingFace AutoModelFor\* 모델은 `forward()` 안에서 loss를 계산하므로 `loss:` 생략이 기본.
커스텀 모델은 둘 중 하나를 선택한다.

### Dict I/O 규칙

모든 입출력은 dict 기반이다:
- **입력**: Collator가 생성한 표준 키 (`input_ids`, `pixel_values`, `labels` 등)
- **출력**: `forward()`는 최소 `"logits"` 키 필요 (외부 loss 사용 시)

### export() / load_from_export() 계약

`BaseModel`은 저장/복원용 2개 메서드를 기본 제공한다.

- `export(output_dir)` — `mdp export`가 호출. 기본은 `save_pretrained`가 있으면 위임, 없으면 `model.safetensors` 단일 파일로 저장.
- `load_from_export(artifact_dir)` — `reconstruct_model()`이 `export_info.json`을 감지하면 호출. 기본은 `model.safetensors`를 `load_state_dict`로 로드.

단일 네트워크라면 기본 구현으로 충분하다. **backbone + 커스텀 head처럼 이질적 하위 구조**를 가진 모델은 두 메서드를 함께 오버라이드해야 한다:

```python
class CriticValueModel(BaseModel):
    _block_classes = None

    def export(self, output_dir: Path) -> None:
        # backbone은 HF 포맷, value_head는 별도 파일로 분리 저장
        self.backbone.save_pretrained(output_dir / "backbone")
        torch.save(self.value_head.state_dict(), output_dir / "value_head.pt")
        # reconstruct_model()이 이 파일을 감지해 load_from_export()를 호출한다
        (output_dir / "export_info.json").write_text('{"format": "split_backbone_head"}')

    def load_from_export(self, artifact_dir: Path) -> None:
        from safetensors.torch import load_file
        self.backbone.load_state_dict(
            load_file(artifact_dir / "backbone" / "model.safetensors")
        )
        self.value_head.load_state_dict(
            torch.load(artifact_dir / "value_head.pt",
                       map_location="cpu", weights_only=True)
        )
```

**대칭 유지 필수**: `export()`만 오버라이드하고 `load_from_export()`를 빼먹으면 `mdp serve`/`reconstruct_model()`이 기본 safetensors 로더로 떨어져 실패한다.

---

## 커스텀 학습 콜백

`BaseCallback`을 상속하여 8개 hook 중 필요한 것만 구현한다:

```python
# my_callbacks/custom.py
from mdp.callbacks.base import BaseCallback

class GradientLogger(BaseCallback):
    """매 배치 gradient norm을 로깅하는 콜백."""

    critical = False  # True면 예외 발생 시 학습 중단

    def on_batch_end(self, step, metrics=None, **kwargs):
        model = kwargs.get("model")
        if model:
            total_norm = sum(
                p.grad.norm().item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            print(f"Step {step}: gradient norm = {total_norm:.4f}")
```

```yaml
callbacks:
  - _component_: my_callbacks.custom.GradientLogger
```

### Hook 호출 순서

```
on_train_start
  on_epoch_start
    on_batch_start → on_batch_end (반복)
  on_epoch_end
  on_validation_start → on_validation_end
on_train_end
```

### should_stop

콜백에서 `self.should_stop = True`를 설정하면 다음 epoch에서 학습이 종료된다.
`EarlyStopping`이 이 메커니즘을 사용한다.

### critical 플래그

- `critical = False` (기본): 예외 발생 시 경고만 출력
- `critical = True`: 예외가 전파되어 학습 중단 (`ModelCheckpoint`, `EarlyStopping`에 사용)

---

## 커스텀 추론 콜백

`BaseInferenceCallback`을 상속한다:

```python
# my_callbacks/analysis.py
from mdp.callbacks.base import BaseInferenceCallback

class AttentionExtractor(BaseInferenceCallback):
    def __init__(self, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self._attentions = []

    def setup(self, model, tokenizer=None, **kwargs):
        """모델에 forward hook 등록."""
        target = dict(model.named_modules())[f"model.layers.{self.layer_idx}.self_attn"]
        self._handle = target.register_forward_hook(self._capture)

    def _capture(self, module, input, output):
        # 모델 구조에 따라 attention weights 위치가 다름
        if isinstance(output, tuple) and len(output) > 1:
            self._attentions.append(output[1].detach().cpu())

    def on_batch(self, batch_idx, batch, outputs, **kwargs):
        """매 배치 후 호출. 캡처된 데이터 후처리."""
        pass

    def teardown(self, **kwargs):
        """hook 해제 + 결과 저장."""
        self._handle.remove()
        torch.save(self._attentions, f"attentions_layer{self.layer_idx}.pt")
```

```yaml
# analysis.yaml
- _component_: my_callbacks.analysis.AttentionExtractor
  layer_idx: 20
```

```bash
mdp inference --pretrained hf://model --data test.jsonl --callbacks analysis.yaml
```

---

## 커스텀 Dataset

```python
# my_data/custom.py
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, source: str, fields: dict, **kwargs):
        self.data = self._load(source)
        self.fields = fields

    def _load(self, source):
        # 커스텀 로딩 로직
        ...

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {role: item[col] for role, col in self.fields.items()}
```

```yaml
data:
  dataset:
    _component_: my_data.custom.CustomDataset
    source: /path/to/data
    fields: {text: content, label: category}
```

---

## 커스텀 Collator

```python
# my_data/collators.py
class CustomCollator:
    def __init__(self, tokenizer: str, max_length: int = 512):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        texts = [item["text"] for item in batch]
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor([item["label"] for item in batch]),
        }
```

```yaml
data:
  collator:
    _component_: my_data.collators.CustomCollator
    tokenizer: bert-base-uncased
    max_length: 256
```

---

## 커스텀 Loss 함수

```python
# my_losses/focal.py
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, labels):
        ce = nn.functional.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()
```

```yaml
loss:
  _component_: my_losses.focal.FocalLoss
  gamma: 2.0
  alpha: 0.25
```

---

## Semantic Module Routing 매핑

`target`, `slot`, `save` 필드의 semantic 이름은 모델 family별로 실제 모듈명으로 번역된다. Factory는 모델 로딩 후 `family_routing` 모듈로 번역을 수행하므로 동일한 Recipe를 서로 다른 family에 재사용할 수 있다. 개념·YAML 예시는 `AGENT.md`의 "Semantic Module Routing" 섹션에 있고, 여기는 **raw name 매핑과 family별 제약의 참조 자료**다.

### Semantic 네임스페이스

| Prefix | 이름 | 설명 |
|--------|------|------|
| `attn.` | `q`, `k`, `v`, `o`, `qkv` | Attention projections |
| `mlp.` | `gate`, `up`, `down`, `fc1`, `fc2`, `gate_up` | MLP/FFN layers |
| `head.` | `cls`, `lm`, `det`, `seg` | 출력 head |
| `embed.` | `token`, `pos` | Embedding layers |
| `conv.` | `dw` | Convolution layers |

### Family별 raw name 매핑

| Semantic | Llama | BERT | GPT-2 | ViT(HF) | ViT-timm | ConvNeXt |
|----------|-------|------|-------|---------|----------|----------|
| `attn.q` | q_proj | query | - | query | - | - |
| `attn.qkv` | - | - | c_attn | - | qkv | - |
| `attn.o` | o_proj | attention.output.dense | attn.c_proj | attention.output.dense | proj | - |
| `mlp.fc1` | - | intermediate.dense | c_fc | intermediate.dense | fc1 | fc1 |
| `mlp.fc2` | - | *미지원* | mlp.c_proj | *미지원* | fc2 | fc2 |
| `head.cls` | - | classifier | - | classifier | head | head |
| `head.lm` | lm_head | - | lm_head | - | - | - |
| `conv.dw` | - | - | - | - | - | conv_dw |

### Family 식별 제약

- **ViT(HF) vs ViT-timm**: HF ViT(`config.model_type="vit"`, family=`vit`)는 BERT-style 모듈명을 사용한다. timm ViT(family=`vit_timm`)는 timm 고유 모듈명을 사용한다. HF Swin(family=`swin`)도 BERT-style이며, timm Swin은 `swin_timm` family로 분리되어 있다.
- **BERT-style `mlp.fc2` 미지원**: BERT/ViT(HF)/Swin(HF) 및 이들의 alias(roberta, dinov2, segformer)에서 `mlp.fc2`는 semantic target으로 사용할 수 없다. PEFT의 suffix 매칭에서 `"output.dense"`가 `"attention.output.dense"`까지 매칭하여 의도하지 않은 attention 모듈에도 LoRA가 적용되기 때문이다. MLP output에 LoRA를 적용하려면 `target_modules: [output.dense]`로 raw name을 직접 지정하되, attention output도 함께 매칭됨을 인지해야 한다.

dot(`.`)이 없는 이름은 raw name으로 취급되어 번역 없이 PEFT에 직접 전달된다. `target`과 `target_modules`(또는 `save`와 `modules_to_save`)를 동시에 지정하면 ValueError로 차단된다.

---

## 커스텀 분산 전략

Config에서 `_component_` dict로 지정한다:

```yaml
compute:
  distributed:
    strategy:
      _component_: my_strategies.CustomStrategy
      custom_param: value
```

`BaseStrategy` 인터페이스는 **필수 3** + **선언적 브리지 2** + **선택 2** 로 구성된다.

### 필수 메서드 (abstract)

- `setup(model, device, optimizer=None)` — 모델을 분산 래퍼로 감싸 반환
- `save_checkpoint(model, path)` — 체크포인트 저장
- `load_checkpoint(model, path)` — 체크포인트 로드

### 선언적 계약 브리지 (default 제공, 필요 시 override)

trainer는 모델의 `training_step` / `validation_step` / 기타 custom 메서드를 분산 래퍼 위에서 호출할 때 아래 두 메서드를 통해 dispatch한다. 모델 구현자는 분산 래퍼를 의식하지 않고 plain 메서드를 선언하면 되고, 커스텀 전략은 필요에 따라 이 메서드들만 override하면 된다.

- `unwrap(wrapped_model) -> nn.Module` — 분산 래퍼를 벗긴 실제 model 반환. `hasattr` / `getattr` 같은 **read-only** 접근용. 기본 구현은 no-op(단일 GPU/래핑 없음). DDP/FSDP는 `.module`을 반환하도록 override.

- `invoke_custom(wrapped_model, method_name, *args, **kwargs)` — custom 메서드를 분산 의미(gradient sync, FSDP all-gather)를 보존한 채 호출한다. 기본 구현은 `unwrap` 후 `getattr(...)(...)`로 **DDP에는 충분하다** (backward autograd hook이 parameter 단위로 sync를 보장). FSDP는 이 경로로 호출하면 `embed_tokens.weight`가 1-D shard로 남아 `RuntimeError: 'weight' must be 2-D`가 발생하므로, wrapper의 `forward`를 custom 메서드로 일시 swap한 뒤 wrapper를 호출하는 패턴으로 override해야 한다 (`mdp/training/strategies/fsdp.py` 참조).

```python
# 예: 새 전략을 만들 때 두 브리지 중 필요한 것만 override
from mdp.training.strategies.base import BaseStrategy

class MyParallel(BaseStrategy):
    def setup(self, model, device, optimizer=None):
        return MyParallelWrapper(model.to(device))

    def unwrap(self, wrapped_model):
        # 래퍼가 .inner 속성으로 원본을 노출한다면
        return getattr(wrapped_model, "inner", wrapped_model)

    # invoke_custom은 기본 구현(unwrap + getattr)로 충분하면 생략
    # 래퍼 forward 경로를 타야 올바르면 override

    def save_checkpoint(self, model, path): ...
    def load_checkpoint(self, model, path): ...
```

### 선택 메서드

- `setup_models(models, trainable_names, device, optimizers=None)` — RL 멀티 모델용. 기본은 각 모델을 `setup`으로 감싸되, frozen 모델(`trainable_names`에 없음)은 device 이동만 수행
- `cleanup()` — `dist.destroy_process_group()` 등 정리 작업

---

## 커스텀 RL 알고리즘

RLTrainer는 알고리즘 클래스를 일반 Python 객체로 취급한다 (`nn.Module` 불필요). 다음 규약을 따르면 된다:

- **클래스 속성**
  - `needs_generation: bool` — `True`면 매 스텝마다 policy가 응답을 생성(`model.generate`)하고, 그 출력을 reward model에 넣어 reward를 계산 (GRPO/PPO 패턴). `False`면 offline 배치를 그대로 사용 (DPO, weighted-NTP 패턴)
  - `mini_epochs: int` — 생성된 배치당 optimizer step 횟수 (PPO는 4~8, DPO/weighted-NTP는 1)

- **필수 메서드**: `compute_loss(trainable_out, frozen_out, batch) -> dict[str, Tensor]`
  - `trainable_out[name]`: trainable 모델(예: `"policy"`)의 forward 출력
  - `frozen_out[name]`: frozen 모델(예: `"reference"`, `"value"`, `"reward"`)의 forward 출력
  - 반환: `{trainable_name: scalar_loss}`. 키는 `rl.models.*`의 이름과 일치해야 optimizer가 매칭된다

```python
# my_algorithms/custom.py
import torch
import torch.nn.functional as F

class CustomRLLoss:
    needs_generation = False
    mini_epochs = 1

    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def compute_loss(self, trainable_out, frozen_out, batch):
        policy_logits = trainable_out["policy"]["logits"]       # (B, S, V)
        reference_logits = frozen_out["reference"]["logits"]    # (B, S, V)
        labels = batch["labels"]                                 # (B, S)

        # 예: KL-regularized CE loss
        shift_policy = policy_logits[:, :-1].contiguous()
        shift_ref = reference_logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        ce = F.cross_entropy(
            shift_policy.view(-1, shift_policy.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        kl = F.kl_div(
            F.log_softmax(shift_policy, dim=-1),
            F.softmax(shift_ref, dim=-1),
            reduction="batchmean",
        )
        return {"policy": ce + self.beta * kl}
```

```yaml
rl:
  algorithm:
    _component_: my_algorithms.custom.CustomRLLoss
    beta: 0.2
  models:
    policy:
      _component_: AutoModelForCausalLM
      pretrained: hf://some-base-model
      optimizer: {_component_: AdamW, lr: 1.0e-5}
    reference:
      _component_: AutoModelForCausalLM
      pretrained: hf://some-base-model
      # optimizer 없음 → frozen
```

> **모델 출력 형태**: `trainable_out[name]`/`frozen_out[name]`의 값은 `model(input_ids=..., attention_mask=...)`의 출력이다. HF 모델은 `{"logits": ...}` 형태, 커스텀 BaseModel은 `forward()`가 반환한 dict 그대로. 단, `role="value"` 모델은 RLTrainer가 자동으로 `{"values": (B, S)}` 형태로 변환한다 (logits 마지막 차원이 1이면 squeeze).

---

## 로깅 규약

### mdp가 적용하는 서드파티 경고 억제

`mdp/__init__.py`가 import 시점에 다음을 수행한다:
- `FutureWarning` 전역 필터링 (의존성 라이브러리의 API deprecation 예고)
- `torch` beta feature `UserWarning` 필터링
- `transformers`, `datasets`, `accelerate`, `bitsandbytes` 로거를 WARNING 레벨로 설정

따라서 커스텀 코드에서 `import mdp`를 거친 뒤에는 이 의존성들의 INFO 로그가 보이지 않는다. 디버깅 시 필요하면 해당 로거의 레벨을 수동으로 낮춘다:

```python
import logging
logging.getLogger("transformers").setLevel(logging.INFO)
```

### 커스텀 코드의 로깅 권장

mdp 내부 규약은 다음과 같다. 커스텀 확장도 동일한 패턴을 따르는 것을 권장한다:

```python
import logging
logger = logging.getLogger(__name__)  # 모듈마다 __name__으로 생성

# 일회성 이벤트: warning
logger.warning("MLflow params 로깅 실패: %s", e)

# 반복 호출 이벤트 (매 step/batch): debug
# 실패해도 학습을 막지 않는 텔레메트리/로깅 호출은 debug로 낮춰
# 장애 시 경고 폭주를 방지한다
logger.debug("MLflow metric 로깅 실패 (key=%s): %s", key, e)

# 치명적 실패: exception (traceback 포함)
logger.exception("학습 실패 상세")
```

- **`print()` 지양**: mdp는 사용자 출력에만 `typer.echo`를 쓰고, 내부 메시지는 모두 logging 모듈로 통일. `print()`는 JSON 모드에서 stdout을 오염시켜 agent 파싱을 깨뜨린다
- **`critical = True` 콜백**: 예외가 학습 루프로 전파되어 중단을 일으킨다. 일회성 이벤트(체크포인트 저장 등)에만 사용
