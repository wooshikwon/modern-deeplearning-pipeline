# Preference Collator + Fields 검증 강화

## 배경

### 왜 필요한가

MDP는 현재 단일 시퀀스 학습만 지원한다 — `{text}` → causal, `{text, label}` → classification 등. 하지만 다음 시나리오는 **쌍(pair) 데이터**를 필요로 한다:

- **Reward Modeling** (RLHF): 좋은 응답 vs 나쁜 응답을 비교하여 보상 모델 학습
- **DPO** (Direct Preference Optimization): 선호 쌍으로 직접 policy 학습
- **Critic 학습** (weighted-NTP): 정답 코드 vs 오답 코드의 pairwise ranking
- **검색 모델**: query에 대해 관련 문서 vs 비관련 문서

이 모든 시나리오는 같은 데이터 구조를 공유한다: `{prompt, chosen, rejected}`. HuggingFace TRL이 이미 이 규약을 확립했고, HF Hub에 수백 개 데이터셋이 이 형태다.

### 현재 구조의 문제

1. `derive_label_strategy`가 `{chosen, rejected}` 역할을 인식하지 못한다
2. `_select_collator`에 preference 분기가 없다
3. `validate_task_fields`가 필수 fields 누락을 **경고만** 하고 학습을 시작시킨다 — 사용자가 GPU 비용을 날린 뒤에야 런타임 에러를 만남

---

## 데이터 규약

### 사용자 데이터 형태

HF TRL 표준을 따른다:

```jsonl
{"prompt": "Sort the array...", "chosen": "def solve(): return sorted(arr)", "rejected": "def solve(): return list(arr)"}
```

- `prompt`: 공통 맥락. loss masking 대상 (labels에서 -100). **생략 가능** — 없으면 전체 시퀀스에 loss 적용.
- `chosen`: 긍정 시퀀스. **필수**.
- `rejected`: 부정 시퀀스. **필수**.

### Recipe YAML

```yaml
data:
  source: ./pairs.jsonl
  fields:
    prompt: prompt
    chosen: chosen
    rejected: rejected
  tokenizer:
    pretrained: meta-llama/Meta-Llama-3-8B
    max_length: 2048
```

### 다양한 사용처와 매핑

| 사용처 | prompt | chosen | rejected |
|--------|--------|--------|----------|
| Critic (weighted-NTP) | 문제 설명 | 정답 코드 | 오답 코드 |
| RLHF Reward Modeling | 사용자 질문 | 좋은 응답 | 나쁜 응답 |
| DPO | 지시문 | 선호 응답 | 비선호 응답 |
| Sentence Embedding | (빈 문자열) | 유사 문장 | 비유사 문장 |
| 검색 (Retrieval) | 쿼리 | 관련 문서 | 비관련 문서 |

---

## 변경 범위

### 1. `LABEL_PREFERENCE` 상수 추가 + `derive_label_strategy` 확장

**파일**: `mdp/data/tokenizer.py`

`chosen`과 `rejected` 역할이 모두 있으면 `LABEL_PREFERENCE`를 반환한다. `chosen`/`rejected`는 기존 역할(`text`, `target`, `label` 등)보다 **우선순위가 높다** — pairwise 데이터에 `text` 역할이 함께 있어도 LABEL_PREFERENCE로 결정.

```
기존 분기 순서:         추가 후:
target → SEQ2SEQ       chosen+rejected → PREFERENCE (최우선)
token_labels → ALIGN   target → SEQ2SEQ
text+label → COPY      token_labels → ALIGN
text → CAUSAL           text+label → COPY
(else) → NONE           text → CAUSAL
                        (else) → NONE
```

### 2. `PreferenceCollator` 신규 구현

**파일**: `mdp/data/collators.py` (신규)

동작:

1. 각 샘플에 대해:
   - `prompt` + `chosen` → tokenize → `chosen_input_ids` + `chosen_labels` (prompt 길이만큼 -100)
   - `prompt` + `rejected` → tokenize → `rejected_input_ids` + `rejected_labels` (prompt 길이만큼 -100)
   - `prompt`가 없으면 chosen/rejected를 그대로 tokenize, labels 전체에 loss 적용
2. 배치 내 chosen끼리 padding, rejected끼리 padding (길이 독립)
3. 반환 dict:
   ```
   {
     chosen_input_ids: [batch, max_len_chosen],
     chosen_attention_mask: [batch, max_len_chosen],
     chosen_labels: [batch, max_len_chosen],
     rejected_input_ids: [batch, max_len_rejected],
     rejected_attention_mask: [batch, max_len_rejected],
     rejected_labels: [batch, max_len_rejected],
   }
   ```

### 3. `_select_collator`에 `LABEL_PREFERENCE` 분기 추가

**파일**: `mdp/data/dataloader.py`

```python
if label_strategy == LABEL_PREFERENCE:
    return PreferenceCollator(tokenizer=tokenizer, max_length=max_length)
```

LABEL_PREFERENCE일 때 `build_tokenizer`는 None을 반환한다 — tokenize를 collator가 직접 수행하므로 별도 tokenize_fn이 불필요.

### 4. `validate_task_fields` 경고 → 에러 전환

**파일**: `mdp/task_taxonomy.py`, `mdp/settings/validation/business_validator.py`

현재 `validate_task_fields`는 `list[str]` (경고 목록)을 반환하고, `BusinessValidator._check_task_fields`가 `result.warnings`에 추가한다. 필수 fields 누락은 학습이 불가능하므로 **에러**로 전환한다.

변경:
- `validate_task_fields`가 `(errors: list[str], warnings: list[str])` 튜플을 반환
- 필수 fields 누락 → `errors`
- 필수 config 누락 (tokenizer, augmentation) → `warnings` (없어도 학습은 가능, 품질 문제)
- `BusinessValidator._check_task_fields`가 errors는 `result.errors`에, warnings는 `result.warnings`에 추가

### 5. preference 전용 fields 검증

LABEL_PREFERENCE 감지 시 `chosen`과 `rejected` 모두 필수. 하나라도 없으면 에러.

이 검증은 `validate_task_fields`와 별개로 `derive_label_strategy` 시점이 아닌 **`_select_collator` 또는 `create_dataloaders`** 시점에서 수행. 이유: label_strategy 파생은 어떤 전략인지 결정하는 것이고, 필수 fields 검증은 그 전략이 동작 가능한지 확인하는 것.

---

## 수정 파일 요약

| 파일 | 변경 유형 | 내용 |
|------|----------|------|
| `mdp/data/tokenizer.py` | 수정 | `LABEL_PREFERENCE` 상수, `derive_label_strategy`에 chosen/rejected 분기 |
| `mdp/data/collators.py` | **신규** | `PreferenceCollator` 구현 |
| `mdp/data/dataloader.py` | 수정 | `_select_collator`에 PREFERENCE 분기, PREFERENCE일 때 tokenize_fn=None |
| `mdp/data/__init__.py` | 수정 | `PreferenceCollator` export |
| `mdp/task_taxonomy.py` | 수정 | `validate_task_fields` 반환 타입 변경 (warnings → (errors, warnings)) |
| `mdp/settings/validation/business_validator.py` | 수정 | `_check_task_fields`에서 errors/warnings 분리 |

---

## 구현 순서

### Step 1: validate_task_fields 경고 → 에러

`task_taxonomy.py`의 반환 타입을 `(errors, warnings)` 튜플로 변경. `business_validator.py`에서 errors는 `result.errors`에 추가. 기존 테스트(`test_fields_routing.py`)가 경고를 기대하므로 함께 수정.

### Step 2: LABEL_PREFERENCE + derive_label_strategy

`tokenizer.py`에 상수 추가, 분기 최상단에 `chosen + rejected → PREFERENCE`. `build_tokenizer`에서 PREFERENCE이면 None 반환 (collator가 tokenize 담당).

### Step 3: PreferenceCollator

`collators.py` 신규. tokenizer를 받아서 prompt+chosen, prompt+rejected를 각각 tokenize + padding + label masking.

### Step 4: _select_collator 연결

`dataloader.py`에서 PREFERENCE일 때 `PreferenceCollator` 반환. `create_dataloaders`에서 PREFERENCE일 때 tokenize_fn을 None으로 설정.

### Step 5: 테스트

- `test_fields_routing.py` 확장: chosen+rejected → LABEL_PREFERENCE 파생 테스트
- `test_data_integration.py` 확장: PreferenceCollator가 올바른 dict 구조 반환하는지
- `test_preference_e2e.py` (신규): pairwise 데이터 + 커스텀 모델로 Trainer 학습 1 epoch

---

## 영향 범위

### Trainer

변경 없음. `model.training_step(batch)`가 dict를 받는 구조이므로 dict 키만 달라진다 (`input_ids` 대신 `chosen_input_ids`/`rejected_input_ids`). 모델의 `training_step`이 이 키를 읽어서 pairwise loss를 계산.

### AGENT.md

- `data.fields` 문서에 `prompt`, `chosen`, `rejected` 역할 추가
- `data.fields 역할명과 전처리 매핑` 테이블에 preference 행 추가
- Common Pitfalls에 "preference 학습 시 prompt 분리 권장" 추가
