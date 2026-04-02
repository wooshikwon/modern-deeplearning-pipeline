# audit 잔여 항목 구현 계획

**기준일**: 2026-04-02
**대상**: `docs/audit-2026-04-02.md`의 논의 필요 4건 + /coding 위반 7건 = 총 11건
**원칙**: 설계 문서(AGENT.md)가 정의한 범위 안에서 최선의 유연성과 정합성을 갖춘다

---

## 변경 그룹 분류

11건을 의존성과 영향 범위로 3개 그룹으로 나눈다.

| 그룹 | 내용 | 파일 수 | 의존 관계 |
|------|------|--------|---------|
| **A. 즉시 수정** | 단일 파일 내 명확한 수정. 테스트 영향 없음 | 5 | 없음 |
| **B. inference drift pipeline** | inference에서 drift detection 활성화 + monitoring 스키마 정합 | 3 | A 완료 후 |
| **C. RL multi-model 인터페이스** | RLTrainer validation + strategy 추상화 | 3 | 독립 |

---

## 그룹 A — 즉시 수정 (6건)

### A-1. `_extract_logits` 중복 제거

**위치**: `training/rl_trainer.py`
**현황**: 같은 함수가 683행(static method)과 701행(지역 함수) 두 곳에 정의됨. static method는 어디서도 호출되지 않음.
**변경**: 683-689행의 static method 삭제.

```python
# 삭제 대상 (rl_trainer.py:683-689)
@staticmethod
def _extract_logits(out):
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict):
        return out.get("logits", out.get("output"))
    return out
```

**검증**: `grep -n "_extract_logits" rl_trainer.py`로 호출처 확인 → 내부 함수만 호출됨을 재확인.

---

### A-2. preference + tokenizer 미설정 시 명확한 에러

**위치**: `data/dataloader.py:_select_collator()`
**현황**: `tokenizer_config is None`이면 collator가 None 반환. preference 전략에서 collator=None이면 raw text를 torch tensor로 변환 시도 → 불명확한 에러.
**변경**: LABEL_PREFERENCE + tokenizer_config=None 조합을 사전 차단.

```python
# dataloader.py:_select_collator 상단, 기존 early return 직후
if label_strategy == LABEL_PREFERENCE and tokenizer_config is None:
    raise ValueError(
        "preference 학습(DPO/GRPO)에는 data.tokenizer 설정이 필수입니다. "
        "recipe의 data.tokenizer.pretrained을 지정하세요."
    )
```

**검증**: 기존 preference 테스트가 모두 tokenizer를 설정하고 있으므로 테스트 영향 없음.

---

### A-3. prefix_tuning 로깅 패턴 통일

**위치**: `models/adapters/__init__.py`, `models/adapters/prefix_tuning.py`
**현황**: lora/qlora는 `apply_adapter` 끝에서 `log_trainable_params(model)` 호출. prefix_tuning은 `apply_prefix_tuning` 내부에서 직접 로깅.
**변경**:
1. `apply_prefix_tuning` 내부의 직접 로깅 삭제
2. `apply_adapter` return 직전에 method-공통으로 `log_trainable_params` 호출

```python
# adapters/__init__.py — 기존 method별 return을 변수에 받고 공통 로깅
if method == "lora":
    ...
    result = apply_lora(model, **config)
elif method == "qlora":
    ...
    result = apply_qlora(model_name, **config)
elif method == "prefix_tuning":
    ...
    result = apply_prefix_tuning(model, **config)
else:
    raise ValueError(...)

log_trainable_params(result)
return result
```

**검증**: 로깅 출력이 한 번만 나오는지 확인.

---

### A-4. vision task + tokenizer 경고

**위치**: `settings/validation/business_validator.py:_check_task_fields()`
**현황**: vision task에서 tokenizer를 설정해도 경고 없음. tokenizer가 무시되는 사실을 사용자가 모를 수 있음.
**변경**: `_check_task_fields` 내부, validate_task_fields 호출 후 추가 검증.

```python
# business_validator.py:_check_task_fields 끝부분
VISION_TASKS = {"image_classification", "object_detection", "semantic_segmentation"}
if recipe.task in VISION_TASKS and data.tokenizer is not None:
    result.warnings.append(
        f"태스크 '{recipe.task}'에서는 tokenizer가 사용되지 않습니다. "
        "data.tokenizer 설정을 제거하면 불필요한 로딩을 방지할 수 있습니다."
    )
```

**검증**: 기존 vision 테스트 fixture에 tokenizer가 없으므로 테스트 영향 없음.

---

### A-5. tokenizer 로드 로직 통합

**위치**: `serving/server.py`
**현황**: `_load_tokenizer_from_recipe(recipe)`와 `_load_tokenizer(model_dir, recipe)` 두 함수가 recipe fallback 로직을 중복 구현.
**변경**: `_load_tokenizer(model_dir, recipe)` 하나로 통합. model_dir=None이면 recipe에서만 로드.

```python
def _load_tokenizer(model_dir: Path | None, recipe: Any) -> Any:
    """tokenizer를 로드한다. model_dir 우선, 없으면 recipe fallback."""
    # 1. model_dir에서 시도
    if model_dir is not None:
        tokenizer_json = model_dir / "tokenizer.json"
        tokenizer_config = model_dir / "tokenizer_config.json"
        if tokenizer_json.exists() or tokenizer_config.exists():
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(str(model_dir))

    # 2. recipe에서 fallback
    if recipe.data.tokenizer:
        pretrained = (
            recipe.data.tokenizer.get("pretrained")
            if isinstance(recipe.data.tokenizer, dict)
            else getattr(recipe.data.tokenizer, "pretrained", None)
        )
        if pretrained:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(pretrained)

    return None
```

`_load_tokenizer_from_recipe` 삭제. `create_handler`의 호출부 단순화:

```python
# 기존: tokenizer = _load_tokenizer(model_dir, recipe) if model_dir else _load_tokenizer_from_recipe(recipe)
# 변경:
tokenizer = _load_tokenizer(model_dir, recipe)
```

**검증**: serve 테스트에서 tokenizer 로드 경로 확인.

---

### A-6. adapter config auto-mapping 문서화

**위치**: `AGENT.md`
**현황**: recipe에서 `alpha`, `dropout`, `r`을 쓰면 내부적으로 PEFT 파라미터명으로 변환되지만 문서에 없음.
**변경**: AGENT.md의 adapter 섹션에 매핑 테이블 추가.

```markdown
#### Recipe → PEFT 파라미터 자동 매핑

Recipe에서는 간결한 이름을 사용하고, 내부적으로 PEFT가 요구하는 이름으로 변환한다:

| Recipe 필드 | PEFT 필드 | 적용 method |
|------------|----------|------------|
| `alpha` | `lora_alpha` | lora, qlora |
| `dropout` | `lora_dropout` | lora, qlora |
| `r` | `num_virtual_tokens` | prefix_tuning |

PEFT 원본 이름(`lora_alpha` 등)을 직접 사용해도 동작한다.
```

**검증**: 코드 변경 없음.

---

## 그룹 B — inference drift detection pipeline (3건 통합)

감사 항목: 위반 9 (inference monitoring 미구현), 위반 10 (embedding centroid 미사용), 논의 6 (estimator 개선)

### 배경

AGENT.md가 설계한 drift detection 흐름:
1. `mdp train`에서 baseline 계산 + 저장 (`monitoring.baseline_saved: true`)
2. `mdp inference`에서 현재 데이터 측정 + baseline 비교 → drift 보고
3. agent가 severity_level에 따라 의사결정 (watch/alert/retrain)

현재 1번은 구현 완료 (compute_baseline + severity_level 추가됨). 2번이 미구현. 3번은 1+2가 있으면 자동으로 가능.

### B-1. inference에서 drift detection 활성화

**위치**: `cli/inference.py`
**현황**: InferenceResult에 `monitoring` 필드가 있지만 실제 drift detection을 호출하지 않음.
**변경**: inference 실행 후, baseline이 존재하면 비교 수행.

```python
# cli/inference.py — 추론 완료 후, result 생성 전
monitoring_result = None
if settings.recipe.monitoring.enabled:
    from mdp.monitoring.baseline import compute_baseline, compare_baselines

    baseline_path = _resolve_baseline_path(model_path)
    if baseline_path and baseline_path.exists():
        import yaml
        stored_baseline = yaml.safe_load(baseline_path.read_text())
        current = compute_baseline(model, test_loader, settings)
        monitoring_result = compare_baselines(stored_baseline, current, settings)
```

baseline 경로 결정 함수:

```python
def _resolve_baseline_path(model_path: Path) -> Path | None:
    """체크포인트 또는 artifact 디렉토리에서 baseline.yaml을 찾는다."""
    candidates = [
        model_path / "baseline.yaml",
        model_path.parent / "baseline.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None
```

InferenceResult 생성 시 monitoring_result 전달:

```python
result = InferenceResult(
    output_path=str(result_path),
    task=settings.recipe.task,
    monitoring=monitoring_result,       # 추가
    evaluation_metrics=eval_results or None,
)
```

**의존**: `mdp train`에서 baseline을 체크포인트 디렉토리에 저장해야 함. 현재 Trainer._maybe_compute_baseline()이 dict를 반환하지만 파일로 저장하지 않음.

### B-2. baseline을 체크포인트에 저장

**위치**: `training/trainer.py`
**현황**: `_maybe_compute_baseline()`이 dict를 반환하지만 파일 저장 없음.
**변경**: baseline이 계산되면 체크포인트 디렉토리에 `baseline.yaml`로 저장.

```python
# trainer.py — _maybe_compute_baseline 호출 후
baseline_info = self._maybe_compute_baseline()
if baseline_info is not None and self._is_main_process:
    import yaml
    ckpt_dir = Path(self.settings.config.storage.checkpoint_dir)
    baseline_path = ckpt_dir / "baseline.yaml"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(yaml.dump(baseline_info, allow_unicode=True))
    logger.info("Baseline saved: %s", baseline_path)
```

### B-3. estimator에서 AutoConfig 활용

**위치**: `utils/estimator.py`
**현황**: class_path 문자열에서 정규식으로 파라미터 수 추정. 커스텀 모델명이면 기본값 100M.
**변경**: HuggingFace AutoConfig를 먼저 시도하고, 실패하면 기존 regex fallback.

```python
@staticmethod
def _estimate_param_count(class_path: str) -> int:
    # 1. AutoConfig로 정확한 추정 시도
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(class_path)
        # config에서 파라미터 수 계산
        if hasattr(config, "num_parameters"):
            return config.num_parameters
        # 일반적인 transformer 구조에서 추정
        h = getattr(config, "hidden_size", None)
        n = getattr(config, "num_hidden_layers", None)
        v = getattr(config, "vocab_size", None)
        if h and n:
            # transformer: ~12 * n * h^2 + v * h (rough estimate)
            param_est = 12 * n * h * h
            if v:
                param_est += v * h
            return param_est
    except Exception:
        pass

    # 2. 기존 regex fallback
    ...
```

**검증**: 기존 estimator 테스트 + `gpt2` class_path로 AutoConfig 추정값과 known_sizes 비교.

---

## 그룹 C — RL multi-model 인터페이스 (2건 통합)

감사 항목: 논의 4 (RLTrainer validation 부재), 위반 5 (multi-model strategy 추상화)

### 배경

RLTrainer는 SFT Trainer와 다른 multi-model 구조 (policy, reference, value, reward)를 갖는다. 두 가지 문제:

1. **validation 부재**: `val_loader`가 할당되지만 사용되지 않음. 학습 중 "best 모델" 선별 불가.
2. **strategy 추상화**: `setup_models`가 있는 strategy(FSDP)와 없는 strategy(DDP)에서 동작이 다름.

위반 5(strategy 추상화)는 이미 해소됨 — `BaseStrategy.setup_models()`에 기본 구현이 추가되어 모든 strategy가 multi-model 인터페이스를 지원한다. RLTrainer의 `hasattr` 분기는 이제 항상 True이므로, else 분기를 삭제하여 정리한다.

### C-1. RLTrainer strategy 분기 정리

**위치**: `training/rl_trainer.py:421-442`
**현황**: `hasattr(self.strategy, "setup_models")` 분기가 있으나, BaseStrategy에 기본 구현이 있으므로 항상 True.
**변경**: else 분기 삭제, setup_models 직접 호출로 단순화.

```python
# 기존
if hasattr(self.strategy, "setup_models"):
    wrapped = self.strategy.setup_models(all_models, self.device, trainable_names)
    ...
else:
    for name, model in self.trainable.items():
        self.trainable[name] = self.strategy.setup(model, self.device)
    ...

# 변경
wrapped = self.strategy.setup_models(all_models, self.device, trainable_names)
for name in self.trainable:
    self.trainable[name] = wrapped[name]
for name in self.frozen:
    self.frozen[name] = wrapped[name]
self.policy = self.trainable["policy"]
```

### C-2. RLTrainer validation 구현

**위치**: `training/rl_trainer.py`
**현황**: val_loader가 할당되지만 학습 루프에서 사용되지 않음.
**변경**: reward 기반 validation 추가. val_loader의 prompt에 대해 generation → reward scoring → 평균 reward를 메트릭으로 반환.

설계:
- val_check를 `every_n_steps` 단위로 실행 (step 기반 학습이므로 epoch 단위 불가)
- val_check_interval은 recipe.training.val_check_interval 재사용
- 메트릭: `val_mean_reward` — ModelCheckpoint가 monitor 가능

```python
def _run_rl_validation(self) -> dict[str, float]:
    """validation prompt에 대해 generation + reward scoring."""
    if self.val_loader is None or "reward" not in self.frozen:
        return {}

    self.policy.eval()
    total_reward = 0.0
    count = 0

    with torch.no_grad():
        for batch in self.val_loader:
            batch = self._move_to_device(batch)
            prompt_ids = batch["input_ids"]
            prompt_mask = batch.get("attention_mask")

            generated_ids = self.policy.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                **self._generation_kwargs,
            )

            reward_out = self._forward_model(
                self.frozen["reward"],
                {"input_ids": generated_ids},
                role="reward",
            )
            rewards = self._compute_rewards(
                reward_out, generated_ids,
                (generated_ids != self.policy.config.pad_token_id).long(),
            )
            total_reward += rewards.sum().item()
            count += rewards.shape[0]

    self.policy.train()
    mean_reward = total_reward / max(count, 1)
    return {"val_mean_reward": mean_reward}
```

학습 루프에서 호출:

```python
# rl_trainer.py — train() 내부, step 완료 후
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

**val_check_interval**: RLTrainer `__init__`에서 `training.val_check_interval`을 step 단위로 해석 (val_check_unit은 RL에서 항상 "step").

**검증**: 기존 RL 테스트에서 val_loader=None이면 validation 스킵됨. val_loader가 있는 새 테스트 추가.

---

## DualEncoderHead forward_pair — 현상 유지

BaseHead의 `forward(features) → Tensor` 계약 밖에 `forward_pair()`가 있지만, 이것은 CLIP/SigLIP contrastive learning의 구조적 요구에서 비롯된다. BaseHead 계약을 확장하면 다른 head에 불필요한 메서드를 강제하게 되고, forward_pair를 삭제하면 CLIP 학습이 불가능해진다.

AGENT.md가 `feature_extraction` task에 `DualEncoderHead`를 허용하고 있으므로, 이 head가 dual encoder 전용 메서드를 제공하는 것은 설계 범위 안이다. 별도 인터페이스(PairwiseHead ABC)로 분리하는 것은 현재 DualEncoderHead 하나만 해당하므로 과잉 추상화이다.

**판단**: 변경 없음.

---

## Seq2SeqLMHead vs CausalLMHead — 현상 유지

구조적으로 동일하지만, AGENT.md의 Task-Head 호환성 테이블에서 `text_generation → CausalLMHead`, `seq2seq → Seq2SeqLMHead`로 분리 정의되어 있다. BusinessValidator가 이 매핑을 기반으로 검증하므로, 두 클래스가 별도로 존재해야 검증이 작동한다.

향후 인코더-디코더 모델(T5, BART)에서 cross-attention output 처리 등 차별화가 필요해질 때 독립적으로 확장 가능하다.

**판단**: 변경 없음.

---

## 구현 순서

```
A-1 ~ A-6 (병렬 가능, 6건)
    ↓
B-2 (baseline 저장) → B-1 (inference drift) → B-3 (estimator)
    ↓
C-1 (strategy 분기 정리) → C-2 (RL validation)
```

A 그룹은 상호 독립이므로 병렬 진행 가능.
B 그룹은 B-2가 선행되어야 B-1이 baseline을 읽을 수 있음.
C 그룹은 C-1이 선행되어야 C-2의 strategy 동작이 일관됨.

---

## 파일별 변경 요약

| 파일 | 변경 | 그룹 |
|------|------|------|
| `training/rl_trainer.py` | `_extract_logits` static method 삭제, strategy 분기 정리, `_run_rl_validation` 추가 | A-1, C-1, C-2 |
| `data/dataloader.py` | preference + no tokenizer 에러 | A-2 |
| `models/adapters/__init__.py` | 공통 `log_trainable_params` 호출 | A-3 |
| `models/adapters/prefix_tuning.py` | 내부 로깅 삭제 | A-3 |
| `settings/validation/business_validator.py` | vision + tokenizer 경고 | A-4 |
| `serving/server.py` | tokenizer 로드 통합 | A-5 |
| `AGENT.md` | adapter mapping 테이블 | A-6 |
| `cli/inference.py` | drift detection 호출 | B-1 |
| `training/trainer.py` | baseline.yaml 저장 | B-2 |
| `utils/estimator.py` | AutoConfig 활용 | B-3 |
