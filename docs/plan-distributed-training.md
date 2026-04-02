# 분산 학습 버그 수정 및 최적화

## 배경

학습 코드의 GPU 분산 처리, 메모리 관리, 데이터 로딩을 감사한 결과, 치명 4건 + 중간 6건의 이슈를 발견했다. 공식 문서와 커뮤니티 best practice로 검증 완료.

---

## 치명 이슈 (4건)

### 1. DistributedSampler의 set_epoch() 미호출

**위치**: `training/trainer.py` — 에폭 루프

**문제**: `DistributedSampler`는 매 에폭마다 `sampler.set_epoch(epoch)`를 호출해야 다른 셔플 순서를 생성한다. 호출하지 않으면 모든 에폭에서 동일한 데이터 순서가 각 rank에 공급된다.

**근거**: PyTorch 공식 문서 — "calling `set_epoch()` at the beginning of each epoch before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs."

**수정**:
```python
# trainer.py 에폭 루프 시작부
for epoch in range(self.start_epoch, max_epochs):
    if hasattr(self.train_loader.sampler, "set_epoch"):
        self.train_loader.sampler.set_epoch(epoch)
    ...
```

### 2. DeepSpeed train_micro_batch_size_per_gpu: "auto"

**위치**: `training/strategies/deepspeed.py:15`

**문제**: `"auto"` 값은 HuggingFace Trainer 전용 확장이다. `deepspeed.initialize()`를 직접 호출하는 MDP에서는 인식되지 않아 런타임 에러가 발생한다.

**근거**: DeepSpeed 공식 문서 — `"auto"`는 DeepSpeed 네이티브 config에서 지원하지 않음. HF의 `HfDeepSpeedConfig.fill_match`가 치환하는 메커니즘.

**수정**: 기본 config에서 `"auto"` 제거. Trainer의 batch_size를 직접 주입.
```python
_DEFAULT_DS_CONFIG = {
    "zero_optimization": {"stage": 2},
    "bf16": {"enabled": True},
    "gradient_clipping": 1.0,
    # train_micro_batch_size_per_gpu 제거 — Trainer에서 주입
}

def setup(self, model, device):
    # Trainer의 batch_size를 DeepSpeed config에 주입
    self.ds_config["train_micro_batch_size_per_gpu"] = self._batch_size
    ...
```

`_batch_size`는 `DeepSpeedStrategy.__init__`에서 전달받거나, Trainer가 strategy 생성 시 주입.

### 3. DeepSpeed에 optimizer 미전달 — Trainer optimizer와 이중 존재

**위치**: `training/strategies/deepspeed.py:44`, `training/trainer.py:111-118`

**문제**: `deepspeed.initialize()`에 `optimizer` 인자를 전달하지 않으면 DeepSpeed가 config의 optimizer 설정으로 자체 optimizer를 생성한다. 이때 Trainer도 별도로 optimizer를 생성하므로 두 개가 공존한다. Trainer의 `scaler.step(optimizer)`, `scheduler.step()`, `clip_grad_norm_` 등이 실제 학습에 영향을 미치지 않게 된다.

**근거**: HuggingFace Issue #38753 — custom optimizer가 DeepSpeed에 의해 silent하게 덮어쓰이는 버그 확인.

**수정**: 두 가지 접근 중 택1.

**A. Trainer optimizer를 DeepSpeed에 전달** (권장):
```python
def setup(self, model, device, optimizer=None):
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,  # Trainer가 만든 optimizer 전달
        model_parameters=model.parameters(),
        config=self.ds_config,
    )
    return model_engine
```

Trainer가 strategy.setup 호출 시 optimizer를 함께 전달. DeepSpeed가 이 optimizer를 래핑하여 ZeRO 최적화 적용.

**B. DeepSpeed가 optimizer를 소유** (대안):
DeepSpeed config에 optimizer를 정의하고, Trainer는 optimizer를 생성하지 않음. 다만 이 경우 Trainer의 optimizer 관련 코드(scheduler, grad clipping 등)를 모두 DeepSpeed에 위임해야 하므로 변경 범위가 큼.

**선택**: A가 변경 범위가 작고 기존 Trainer 흐름과 호환됨. 단, DeepSpeed가 반환하는 optimizer를 Trainer가 사용하도록 해야 함 (DeepSpeed가 래핑한 optimizer가 진짜).

### 4. 체크포인트에서 모든 rank가 optimizer/scheduler/state 저장

**위치**: `training/callbacks/checkpoint.py:126-136`

**문제**: `strategy.save_checkpoint(model, path)`는 rank-0에서만 모델을 저장하지만, 그 다음의 `torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")` 등은 모든 rank에서 실행된다. 동일 경로에 여러 rank가 동시에 쓰면 파일 손상.

**근거**: PyTorch 공식 튜토리얼 — "Only the main process (usually rank 0) should save the checkpoint."

**수정**: checkpoint 저장 전체를 rank-0 가드로 감싸기.
```python
def save_checkpoint(self, model, optimizer, scheduler, epoch, global_step,
                    metrics=None, strategy=None, recipe_dict=None) -> Path:
    ckpt_dir = self.dirpath / f"checkpoint-{global_step}"

    # 분산 학습 시 rank-0에서만 저장
    is_main = True
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            is_main = dist.get_rank() == 0
    except Exception:
        pass

    if not is_main:
        return ckpt_dir  # rank-0이 아니면 저장 건너뜀

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # ... 이하 기존 저장 로직
```

---

## 중간 이슈 (6건)

### 5. MPS + fp16 시 GradScaler 런타임 에러

**위치**: `training/trainer.py:73-76`

**문제**: fp16에서 `GradScaler(scaler_device, enabled=True)` 호출. `scaler_device`가 `"mps"`이면 GradScaler가 CUDA 전용이라 에러 가능.

**수정**: MPS에서는 bf16만 지원하거나, GradScaler를 disabled로 강제.
```python
if precision == "fp16":
    if self.device.type == "mps":
        logger.warning("MPS에서 fp16은 GradScaler를 지원하지 않습니다. bf16을 권장합니다.")
        self.scaler = GradScaler(scaler_device, enabled=False)
    else:
        self.scaler = GradScaler(scaler_device, enabled=True)
```

### 6. FSDP MixedPrecision bf16 하드코딩

**위치**: `training/strategies/fsdp.py:64-70`

**문제**: Trainer의 precision 설정(fp16/fp32)과 무관하게 항상 bf16 MixedPrecision policy를 사용.

**수정**: precision을 strategy에 전달.
```python
def __init__(self, ..., precision: str = "bf16"):
    self.precision = precision

def setup(self, model, device):
    if self.mixed_precision:
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        fsdp_kwargs["mixed_precision"] = MixedPrecision(
            param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype,
        )
```

### 7. FSDP auto_wrap_policy 미설정

**위치**: `training/strategies/fsdp.py:76`

**문제**: 모델 전체가 단일 FSDP unit으로 래핑되어 메모리 절감 효과가 제한적.

**수정**: `size_based_auto_wrap_policy` 또는 `transformer_auto_wrap_policy` 적용.
```python
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

fsdp_kwargs["auto_wrap_policy"] = functools.partial(
    size_based_auto_wrap_policy, min_num_params=1_000_000,
)
```

### 8. FSDP load_checkpoint에서 StateDictType 미사용

**위치**: `training/strategies/fsdp.py:92-95`

**문제**: 일반 `load_state_dict()`를 호출하므로, sharded 모델에 full state dict를 로드하면 불일치.

**수정**: save와 동일한 `StateDictType` 컨텍스트에서 로드.
```python
def load_checkpoint(self, model, path):
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = load_file(path)
        model.load_state_dict(state_dict)
    return model
```

### 9. Resume 시 GradScaler 상태 미복원

**위치**: `training/trainer.py` — `_maybe_resume()`

**문제**: fp16 학습에서 GradScaler의 scale factor, growth tracker가 복원되지 않아 resume 직후 불안정.

**수정**: checkpoint에 scaler 상태 저장/복원 추가.
```python
# save
if self.scaler.is_enabled():
    torch.save(self.scaler.state_dict(), ckpt_dir / "scaler.pt")

# resume
scaler_path = ckpt_path / "scaler.pt"
if scaler_path.exists() and self.scaler.is_enabled():
    self.scaler.load_state_dict(torch.load(scaler_path, map_location="cpu"))
```

### 10. EMA shadow parameter GPU 메모리 2배

**위치**: `training/callbacks/ema.py:63-64`

**문제**: 모든 trainable parameter의 GPU 복사본을 유지. 7B 모델이면 ~14GB 추가.

**수정**: shadow parameter를 CPU에 저장하고, 업데이트 시에만 GPU로 이동.
```python
self._shadow_params = [p.data.clone().cpu() for p in model.parameters() if p.requires_grad]

def on_batch_end(self, ...):
    for shadow, param in zip(self._shadow_params, trainable):
        shadow.mul_(self.decay).add_(param.data.cpu(), alpha=1.0 - self.decay)
```

다만 CPU↔GPU 전송 오버헤드가 발생하므로, `update_every`를 크게 설정하여 빈도를 낮추는 것이 현실적.

---

## 경미 이슈 (참고)

| # | 이슈 | 수정 |
|---|------|------|
| 11 | `optimizer.zero_grad(set_to_none=True)` 미사용 | 파라미터에 `set_to_none=True` 추가 (피크 메모리 절감) |
| 12 | DDP `torch.cuda.set_device(local_rank)` 미호출 | NCCL 기본 디바이스 설정을 위해 추가 |
| 13 | 에폭 끝 잔여 gradient 미처리 | `drop_last=True`로 충분히 완화되지만, accumulation 잔여 flush 옵션 고려 |
| 14 | MPS/CPU에서 pin_memory=True 불필요 | device 조건부 설정 |
| 15 | gradient checkpointing 미지원 모델에 warning 없음 | `hasattr` 실패 시 logger.warning 추가 |

---

## 구현 순서

```
치명 (즉시):
1. DistributedSampler set_epoch — trainer.py 에폭 루프
2. DeepSpeed "auto" batch size — deepspeed.py 기본 config
3. DeepSpeed optimizer 전달 — deepspeed.py setup + trainer.py 연동
4. Checkpoint rank-0 가드 — checkpoint.py save_checkpoint

중간 (우선):
5. MPS + fp16 GradScaler — trainer.py AMP 설정
6. FSDP precision 전달 — fsdp.py + trainer.py
7. FSDP auto_wrap_policy — fsdp.py
8. FSDP load_checkpoint StateDictType — fsdp.py
9. GradScaler 상태 저장/복원 — checkpoint.py + trainer.py
10. EMA CPU offload — ema.py
```
