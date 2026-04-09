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

### Loss 선택 분기

- `loss:` 지정 → `model.forward(batch)` 호출 → `loss_fn(outputs["logits"], batch["labels"])`
- `loss:` 생략 → `model.training_step(batch)` 호출 → 반환값을 loss로 사용

HuggingFace AutoModelFor\* 모델은 `forward()` 안에서 loss를 계산하므로 `loss:` 생략이 기본.
커스텀 모델은 둘 중 하나를 선택한다.

### Dict I/O 규칙

모든 입출력은 dict 기반이다:
- **입력**: Collator가 생성한 표준 키 (`input_ids`, `pixel_values`, `labels` 등)
- **출력**: `forward()`는 최소 `"logits"` 키 필요 (외부 loss 사용 시)

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

## 커스텀 분산 전략

Config에서 `_component_` dict로 지정한다:

```yaml
compute:
  distributed:
    strategy:
      _component_: my_strategies.CustomStrategy
      custom_param: value
```

`BaseStrategy` 인터페이스의 5개 메서드를 구현해야 한다:
- `setup(model, device)` — 모델 래핑
- `setup_models(models, trainable_names, device)` — RL 멀티 모델
- `save_checkpoint(model, path)` — 체크포인트 저장
- `load_checkpoint(model, path)` — 체크포인트 로드
- `cleanup()` — 분산 프로세스 정리

---

## 커스텀 RL 알고리즘

```python
# my_algorithms/custom.py
import torch.nn as nn

class CustomRLLoss(nn.Module):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, policy_logps, reference_logps, **kwargs):
        # 커스텀 RL loss 계산
        ...
        return loss
```

```yaml
rl:
  algorithm:
    _component_: my_algorithms.custom.CustomRLLoss
    beta: 0.2
```
