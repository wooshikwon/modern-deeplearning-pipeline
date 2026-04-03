# 감사 논의 항목 구현 계획

감사일 2026-04-03의 "논의 필요" 10건에 대한 구현 방향. 의존 관계와 변경 범위를 고려하여 구현 순서를 정한다.

---

## 구현 순서와 의존 관계

```
[1+2] compat_validator 방어 ─┐
[3] Seq2SeqLMHead alias 통합  │  독립 (병렬 가능)
[5] DataSpec val_split 필드   │
[10] BatchScheduler config    ┘

[7] 콜백 critical 플래그 ────→ [6] RLTrainer validation 통합 (콜백 동작에 의존)

[8] step-level resume offset  (독립)
[9] PPO mini-epoch accum      (독립)

[4] DualEncoderHead           (문서만, 코드 변경 없음)
```

---

## 1+2. compat_validator: FSDP 문자열 매칭 강화 + dict 입력 방어

**파일**: `settings/validation/compat_validator.py`
**변경 범위**: `_check_fsdp_qlora` 1개 메서드 (L86-105)

### 현재 코드

```python
# L99-100
strategy = distributed.get("strategy", "")
if strategy.lower() == "fsdp":
```

### 수정

```python
strategy = distributed.get("strategy", "")
if isinstance(strategy, str) and strategy.lower().startswith("fsdp"):
```

`isinstance` 가드로 dict 입력 시 `AttributeError` 방지, `startswith`로 `"fsdp_full_shard"` 등 변형 커버.

동일 패턴이 `_check_gpu_distributed`(L63)에도 있으나, 여기서는 `strategy`를 warning 메시지에 삽입할 뿐이라 dict가 들어와도 크래시하지 않고 `"{'_component_': ...}'"` 형태로 출력된다. 의미는 이상하지만 에러는 아니므로 수정 범위 밖.

**테스트**: `tests/e2e/test_settings.py`에 FSDP+QLoRA 검증 테스트가 있는지 확인 후, `strategy="fsdp_full_shard"` 케이스와 `strategy={"_component_": "FSDPStrategy"}` 케이스 추가.

---

## 3. Seq2SeqLMHead → CausalLMHead alias 통합

> 참고: audit 정정의 "SequenceClassificationHead"(분류용, aliases.yaml에만 존재)와 이 항목의 "Seq2SeqLMHead"(LM용, 실제 클래스 존재)는 다른 클래스다.

**파일**: 3개
- `models/heads/seq2seq_lm.py` → 삭제
- `aliases.yaml` → Seq2SeqLMHead 매핑 변경
- `models/heads/__init__.py` → import 정리

### 현재 상태

`seq2seq_lm.py`와 `causal_lm.py`가 완전히 동일한 구현 (`nn.Linear(hidden_dim, vocab_size, bias=False)`). 설계 문서에서 "향후 분기 예상"이라 했으나, 분기 시나리오가 구체적이지 않다.

### 수정

1. `aliases.yaml`에서 `Seq2SeqLMHead` 값을 `mdp.models.heads.causal_lm.CausalLMHead`로 변경
2. `models/heads/seq2seq_lm.py` 삭제
3. `models/heads/__init__.py`에서 `Seq2SeqLMHead` import 제거 (있다면)
4. `TASK_HEAD_COMPAT`(`business_validator.py`)에서 `"Seq2SeqLMHead"`는 유지 — alias 이름이므로 검증 시 필요

**호출 흐름 확인**: `Seq2SeqLMHead`를 직접 import하는 코드가 있는지 Grep. 없으면 alias 경유만이므로 안전.

**테스트**: `test_factory_e2e.py`에 seq2seq 태스크 테스트가 있다면, alias 경유로 CausalLMHead가 생성되는지 확인.

---

## 4. DualEncoderHead forward_pair() — 문서만 수정

**코드 변경 없음.** 현상 유지가 최선.

**문서 수정**:
- `AGENT.md`의 "커스텀 모델 구현" 섹션에 다음 가이드 추가:

> DualEncoderHead 사용 시, 모델의 `training_step()`에서 `self.head.forward_pair(image_features, text_features)`를 직접 호출한다. `forward()`는 추론 시 image projection만 수행한다.

---

## 5. DataSpec에 val_split 필드 추가

**파일**: 3개
- `settings/schema.py` → DataSpec에 필드 추가
- `data/dataloader.py` → val split 결정 로직 분기
- `AGENT.md` → 스키마 문서화

### 수정: schema.py

```python
class DataSpec(BaseModel):
    ...
    val_split: str | None = "auto"  # "auto" = 자동 추론, null = 비활성화, 문자열 = 직접 지정
```

### 수정: dataloader.py (L256-259)

현재:
```python
split_str = split if isinstance(split, str) else "train"
val_split = _infer_val_split(split_str)
```

변경:
```python
val_split_cfg = data_spec.val_split  # "auto", None, 또는 문자열
if val_split_cfg is None:
    val_ds = None  # 의도적 비활성화 → try 블록 건너뜀
elif val_split_cfg == "auto":
    split_str = split if isinstance(split, str) else "train"
    val_split = _infer_val_split(split_str)
else:
    val_split = val_split_cfg
```

**호출 흐름**: `create_dataloaders()`는 현재 `data_spec`의 개별 필드를 언팩해서 받는다. `val_split`을 추가 인자로 전달하거나, `data_spec` 객체 자체를 전달하도록 시그니처 조정. Factory(`factory.py:191-217`)에서 호출하므로 거기도 수정.

**기존 사용자 영향 없음**: 기본값 `"auto"`가 현재 동작과 동일.

---

## 6. RLTrainer에 validation 통합

**파일**: `training/rl_trainer.py`
**변경 범위**: `train()` 루프 내 (L508-555)

### 현재 상태

`_run_rl_validation()`(L209-245)이 이미 구현되어 있고, `train()` 루프(L547-554)에서 이미 호출되고 있다:

```python
if (
    self.val_loader is not None
    and self.val_check_interval > 0
    and self.global_step > 0
    and self.global_step % self.val_check_interval == 0
):
    val_metrics = self._run_rl_validation()
    self._fire("on_validation_end", metrics=val_metrics)
    self.last_metrics.update(val_metrics)
```

코드를 다시 읽으니 **이미 통합되어 있다**. 감사 시점의 "RLTrainer validation 부재"는 부정확 — `_run_rl_validation()`이 `train()` 내에서 호출되고 있고, `on_validation_end` 콜백도 발화되며, `last_metrics`도 업데이트된다.

### 실제 잔여 문제

1. **`_generation_kwargs` 초기화 시점**: `_run_rl_validation()`이 `self._generation_kwargs`를 사용하는데 이 속성은 `train()` 내부(L500-502)에서 설정됨. `train()` 전에 validation을 호출하면 `AttributeError`. → `__init__`에서 `self._generation_kwargs = {}` 초기화 추가.

2. **DPO에는 generation 없는 validation 필요**: 현재 `_run_rl_validation()`은 `policy.generate()` → reward scoring 파이프라인인데, DPO는 generation 없이 preference accuracy를 계산해야 함. → DPO용 분기 추가:

```python
def _run_rl_validation(self) -> dict[str, float]:
    if not getattr(self.algorithm, "needs_generation", False):
        return self._run_dpo_validation()  # preference accuracy
    # 기존 generation + reward 로직
```

3. **`_run_dpo_validation()` 신규 구현**: val_loader에서 preference 배치를 받아 chosen/rejected의 log-prob 차이 계산.

### 수정 범위

- `rl_trainer.py:__init__` — `self._generation_kwargs = {}` 추가 (1줄)
- `rl_trainer.py:_run_rl_validation` — DPO 분기 추가 (5줄)
- `rl_trainer.py:_run_dpo_validation` — 신규 메서드 (~20줄)

---

## 7. 콜백 critical 플래그

**파일**: 3개
- `training/callbacks/base.py` → `critical` 속성 추가
- `training/callbacks/checkpoint.py` → `critical = True`
- `training/callbacks/early_stopping.py` → `critical = True`
- `training/trainer.py` → `_fire()`에서 critical 분기

### 수정: base.py

```python
class BaseCallback:
    should_stop: bool = False
    critical: bool = False  # True면 예외 전파
```

### 수정: trainer.py (L282-288)

현재:
```python
for cb in self.callbacks:
    method = getattr(cb, hook_name, None)
    if method:
        try:
            method(**kwargs)
        except Exception as e:
            logger.warning(f"콜백 {type(cb).__name__}.{hook_name} 실패: {e}")
```

변경:
```python
for cb in self.callbacks:
    method = getattr(cb, hook_name, None)
    if method:
        try:
            method(**kwargs)
        except Exception as e:
            if getattr(cb, "critical", False):
                raise
            logger.warning(f"콜백 {type(cb).__name__}.{hook_name} 실패: {e}")
```

**RLTrainer에도 동일 적용**: `rl_trainer.py`의 `_fire()`도 같은 패턴이면 동일 수정.

**기존 사용자 영향 없음**: `critical=False`가 기본이므로 커스텀 콜백은 기존 동작 유지.

---

## 8. step-level checkpoint resume 시 에폭 내 offset 복원

**파일**: 3개
- `training/callbacks/checkpoint.py` → `step_in_epoch` 저장
- `training/trainer.py` → resume 시 offset 복원 + DataLoader skip

### 수정: checkpoint.py

`trainer_state.json`에 `step_in_epoch` 추가:

```python
state = {
    "epoch": epoch,
    "global_step": kwargs.get("global_step", 0),
    "step_in_epoch": kwargs.get("step", 0),  # 에폭 내 raw step
}
```

### 수정: trainer.py

`_maybe_resume()`에서 `step_in_epoch` 복원:

```python
self.start_epoch = state.get("epoch", 0)
self.global_step = state.get("global_step", 0)
self._resume_step_in_epoch = state.get("step_in_epoch", 0)
```

`_train_one_epoch()`에서 offset skip:

```python
loader_iter = iter(self.train_loader)
if self._resume_step_in_epoch > 0:
    for _ in range(self._resume_step_in_epoch):
        next(loader_iter, None)
    self._resume_step_in_epoch = 0  # 한 번만 적용
```

**주의**: `DistributedSampler.set_epoch()`이 resume 에폭과 동일한 셔플 순서를 보장하므로 skip이 의미 있다. `_resume_step_in_epoch`은 gradient accumulation의 raw step(optimizer step 아님)이어야 한다.

**범위 제한**: SFT Trainer만 적용. RLTrainer는 step 기반 루프여서 에폭 내 offset 개념이 없다.

---

## 9. PPO mini-epoch에서 gradient accumulation 비활성화

**파일**: `training/rl_trainer.py`
**변경 범위**: `_train_step_generation()` 내 mini-epoch 루프 (L647-661)

### 현재 코드

```python
# L648-661
ppo_epochs = getattr(self.algorithm, "ppo_epochs", 1)
last_loss = 0.0
for _ in range(ppo_epochs):
    with autocast(...):
        trainable_out = { ... }
        losses = self.algorithm(trainable_out, frozen_out, gen_batch)
    self._backward_and_step(losses)  # ← grad_accum_steps 조건 적용됨
    last_loss = ...
self.global_step += 1
```

`_backward_and_step()`(L718)에서 `(self.global_step + 1) % self.grad_accum_steps == 0` 조건으로 실제 step 여부를 결정한다. mini-epoch 루프 동안 `global_step`은 변하지 않으므로, `ppo_epochs` 중 일부만 실제 step이 발생한다.

### 수정

`_backward_and_step`에 `force_step` 파라미터 추가:

```python
def _backward_and_step(self, losses: dict[str, torch.Tensor], force_step: bool = False) -> None:
    ...
    for name, loss in losses.items():
        scaled = loss / (1 if force_step else self.grad_accum_steps)
        self.scaler.scale(scaled).backward()

    if force_step or (self.global_step + 1) % self.grad_accum_steps == 0:
        # unscale, clip, step, update, zero_grad
        ...
```

mini-epoch 루프에서:
```python
for _ in range(ppo_epochs):
    ...
    self._backward_and_step(losses, force_step=True)
```

**DPO/GRPO 영향 없음**: `_train_step_offline`과 `_train_step_generation`의 non-PPO 경로에서는 `force_step` 기본값 `False`로 기존 동작 유지. GRPO는 mini-epoch 없이 한 번만 `_backward_and_step`을 호출하므로 영향 없다.

---

## 10. BatchScheduler에 ServingConfig 전달

**파일**: 2개
- `serving/handlers.py` → BatchHandler 생성자에 config 전달
- `serving/server.py` → BatchHandler 생성 시 config 전달
- `settings/schema.py` → ServingConfig에 `batch_window_ms` 추가 (선택)

### 수정: handlers.py

```python
class BatchHandler:
    def __init__(self, model, tokenizer, transform, recipe, serving_config=None):
        max_batch = 8
        window_ms = 50
        if serving_config is not None:
            max_batch = getattr(serving_config, "max_batch_size", 8)
            window_ms = getattr(serving_config, "batch_window_ms", 50)
        self.scheduler = _BatchScheduler(model, max_batch_size=max_batch, batch_window_ms=window_ms)
```

### 수정: server.py

`create_app()` 또는 핸들러 생성 지점에서 `serving_config`를 전달. `server.py`에서 이미 `settings.config.serving`에 접근 가능한지 확인 필요.

### ServingConfig 필드 추가 (선택)

`batch_window_ms`가 현재 ServingConfig에 없으면 추가:
```python
class ServingConfig(BaseModel):
    backend: str = "torchserve"
    max_batch_size: int = 1
    batch_window_ms: float = 50  # 추가
```

기본값을 `1`에서 `8`로 올리지 않는다 — 현재 `max_batch_size: 1`이 문서화된 기본값이므로, 핸들러가 이를 존중해야 한다.

---

## 검증 전략

| 항목 | 기존 테스트 | 추가 테스트 |
|------|-----------|-----------|
| 1+2 | test_settings.py | FSDP 문자열 변형 + dict 입력 케이스 |
| 3 | test_factory_e2e.py | seq2seq 태스크에서 alias 경유 head 생성 |
| 5 | test_data_integration.py | val_split="test", val_split=null 케이스 |
| 6 | test_rl_dpo.py | DPO validation 메트릭 반환 확인 |
| 7 | test_callbacks.py | critical 콜백 예외 전파 확인 |
| 8 | test_resume.py | step-level resume 후 step_in_epoch skip 확인 |
| 9 | test_preference.py | PPO ppo_epochs + grad_accum 조합 |
| 10 | test_serve_endpoint.py | config.max_batch_size 반영 확인 |
