# Cloud Testing — Multi-GPU e2e 환경 검증

mdp는 vision · language · RL · serving · callbacks 등 다수 task family를 단일 framework로 다룬다. 이 중 단일 GPU 또는 다중 GPU에서만 검증 가능한 경로(SFT 실제 backward, DPO preference path, DDP/FSDP 초기화, NCCL collective, rank-0 checkpoint, memory ceiling 등)를 vast.ai · runpod에서 **GPU 인스턴스를 단기 임대**해 검증한다. 외부 데이터·모델 의존을 줄이기 위해 **작고 실제로 학습된 HuggingFace 모델**과 **작은 데이터셋 슬라이스**를 fixture로 사용한다.

표준 인터페이스(provision/sync/test/teardown)와 state 파일 형식은 vault `infra/{vastai,runpod}/cloud-runner-spec.md`에 정의되어 있다. 본 문서는 **mdp 측 구현체의 사용법**.

## 권장 GPU 클래스

분산 collective + 다양한 e2e만 검증하면 **고가 GPU 불필요**. NCCL 작동 + 메모리 24GB 정도면 충분.

후보 (싸고 분산 가능):

- **RTX A5000 × 2** — Ampere, 24GB. 가장 안정적
- **RTX 3090 × 2** — Ampere, 24GB. 가용성 높음
- **L4 × 2** — Ada, 24GB. 저전력
- **L40S × 2** — Ada, 48GB. 큰 fixture 필요 시
- **A100 SXM 80GB × 2** — bf16/FSDP 본격 검증
- **H100 PCIe × 2** — 최신 SM/cuDNN 경로 검증

요구사항:

- GPU 수 N ≥ 2
- 디스크 ≥ 100GB (HF 캐시 + tiny fixtures + 체크포인트)
- reliability ≥ 0.95
- CUDA 12.x (torch 2.5.1 + cu124)

## 무엇을 캐싱하고 무엇을 테스트하는가

### 캐시 fixture (`scripts/prepare_test_fixtures.py`)

`/workspace/test-fixtures/`에 다음을 캐싱한다 — 인스턴스 부팅 후 1회 (또는 host ARTIFACT_DIR에서 hydrate):

**Tiny HF 모델** (작지만 실제 weight/tokenizer를 가진 fixture):

| local 이름 | repo | 용도 |
|---|---|---|
| `smollm2` | `HuggingFaceTB/SmolLM2-135M` | SFT, DDP/FSDP SFT |
| `gpt2` | `gpt2` | DPO, generation/inference |
| `bert-tiny` | `google/bert_uncased_L-2_H-128_A-2` | seq-classification |
| `vit-tiny` | `WinKawaks/vit-tiny-patch16-224` | vision evaluation |

**작은 데이터셋 슬라이스**:

| 이름 | 출처 | 크기 |
|---|---|---|
| `wikitext-2-tiny` | `wikitext/wikitext-2-raw-v1` (train[:1000]) | ~1MB |
| `cifar10-tiny` | `cifar10` (train[:1000], tensor로 직렬화) | ~3MB |
| `preference-tiny` | synthetic JSONL | 16 rows |
| `prompt-tiny` | synthetic JSONL | 16 rows |
| `classification-text-tiny` | synthetic JSONL | 16 rows |

캐시 후 `manifest.json`에 모델별 파라미터 수·아키텍처와 dataset path를 기록한다.

### 실행되는 테스트

`cloud_test.sh`는 suite 단위로 실행면을 고른다. 기본값은 `quick`이다. 인자를 직접 넘기면 suite 선택을 건너뛰고 해당 pytest 인자를 그대로 실행한다. 이 suite들은 runtime contract smoke의 소유자가 아니다. CPU/local에서 잡아야 하는 `RunPlan`, `LaunchPlan`, `AssemblyPlan`, `ExecutionEngine`, CLI JSON/error 계약은 `tests/contract`와 일반 unit/e2e subset이 먼저 검증하고, cloud suite는 CUDA/NCCL/fixture cache/memory ceiling처럼 실제 hardware boundary에서만 의미가 있는 경로를 검증한다.

| Suite | 명령 | 검증 대상 |
|---|---|---|
| quick | `bash scripts/cloud_test.sh` | fixture contract + 빠른 single-GPU smoke |
| gpu | `bash scripts/cloud_test.sh --suite gpu` | `gpu and fixtures and not distributed and not memory` |
| distributed | `bash scripts/cloud_test.sh --suite distributed` | `distributed and fixtures`, 2GPU 미만이면 fail-fast |
| memory | `bash scripts/cloud_test.sh --suite memory --memory-profile cuda_24gb` | `memory`, explicit budget profile 필수 |
| acceptance | `bash scripts/cloud_test.sh --suite acceptance --memory-profile cuda_24gb` | 유료 임대 완료 판정. shell CLI `train -> resume -> export -> inference -> generate`, RL CLI, distributed CLI, memory ceiling |
| all | `bash scripts/cloud_test.sh --suite all --memory-profile cuda_24gb` | slow 제외 전체 sweep, 조건 없는 memory는 skip |

테스트 안에서 fixture 경로는 `MDP_TEST_FIXTURES` 환경변수(=`/workspace/test-fixtures`)로 노출되며, conftest에서 이를 활용해 모델/데이터 fixture를 구성할 수 있다.

로컬에서 cloud instance 없이 suite 선택만 확인할 때는 dry-run을 사용한다. 이 명령은 state 파일·SSH·fixture 준비를 건너뛰고 실제로 전달될 pytest 인자만 출력한다.

```bash
bash scripts/cloud_test.sh --suite quick --dry-run
bash scripts/cloud_test.sh --suite gpu --dry-run
bash scripts/cloud_test.sh --suite distributed --dry-run
bash scripts/cloud_test.sh --suite memory --memory-profile cuda_24gb --dry-run
bash scripts/cloud_test.sh --suite acceptance --memory-profile cuda_24gb --dry-run
```

local full sweep과 cloud command는 분리해서 본다.

```bash
uv run pytest --collect-only -q
uv run pytest tests/contract -q
uv run pytest tests/unit -q
uv run lint-imports

bash scripts/cloud_test.sh --suite quick
bash scripts/cloud_test.sh --suite gpu
bash scripts/cloud_test.sh --suite distributed --allow-skip
bash scripts/cloud_test.sh --suite memory --memory-profile cuda_24gb
bash scripts/cloud_test.sh --suite acceptance --memory-profile cuda_24gb
```

특정 marker만:

```bash
bash scripts/cloud_test.sh -m distributed
bash scripts/cloud_test.sh -m "gpu and not slow"
```

marker 정의 (`pyproject.toml`):

- `gpu` — 단일 CUDA GPU 필요
- `distributed` — 다중 GPU + NCCL 필요 (subprocess+torchrun 패턴)
- `fixtures` — `MDP_TEST_FIXTURES`를 사용하는 테스트
- `memory` — `MDP_MEMORY_BUDGET_PROFILE`이 필요한 peak memory ceiling 테스트
- `slow` — 실모델 로딩이 필요한 통합

## 사전 준비 (1회만)

### 1. 자격증명

[[infra/claude-svr/credentials|Bitwarden]]에 다음이 등록되어 있어야 한다:

- `Vast.ai API Key`
- `RunPod API Key`
- `HuggingFace Token` (공개 fixture 다운로드에는 보통 불필요하지만 prepare_test_fixtures가 일관성 위해 HF_HOME만 설정)
- `Personal SSH Key — cloud_runner (mac)` / `(svr)` — instance 로그인 키 백업

### 2. 환경 변수

```bash
export BW_SESSION=$(bw unlock --raw)
ssh-add --apple-use-keychain ~/.ssh/cloud_runner
ssh-add --apple-use-keychain ~/.ssh/id_ed25519   # GitHub clone (agent forwarding)
```

### 3. CLI

```bash
pip install vastai
vastai set api-key "$(bw get password 'Vast.ai API Key' --session $BW_SESSION)"
```

RunPod은 GraphQL을 cloud_provision.sh가 직접 호출.

## 표준 절차

### Step 1 — Offer 검색 (사람 승인)

agent에게 검색 요청. agent는 vault `infra/{vastai,runpod}/pricing-search.md` 절차로 두 provider 후보 표를 만든다:

```
| # | provider | gpu       | n | $/hr | reliability | id/spec       |
| 1 | vast     | RTX_A5000 | 2 | 0.30 | 0.98        | offer 12345…  |
| 2 | runpod   | RTX A5000 | 2 | 0.32 | (community) | spec direct   |
```

사용자가 한 줄 승인.

### Step 2 — Provision

```bash
# Vast.ai
bash scripts/cloud_provision.sh vastai \
  --offer-id 12345678 --gpu-name RTX_A5000 --n-gpus 2 --disk 100

# RunPod
bash scripts/cloud_provision.sh runpod \
  --gpu-name "RTX A5000" --n-gpus 2 --tier community --disk 100
```

기본 설치(torch + mdp core)로 충분. transformers/accelerate가 e2e 테스트에 필요하면 `--with-language` 추가 (flash-attn 빌드로 셋업이 ~15분 더 걸림).

state 파일 작성 위치: `~/.cache/cloud-runner/modern-deeplearning-pipeline/current.env`. 이후 sync/test/teardown은 instance_id 인자 불필요.

### Step 3 — (선택) Hydrate

이전 임대에서 받아둔 fixture를 push:

```bash
bash scripts/cloud_sync.sh push
```

호스트 ARTIFACT_DIR 레이아웃: `~/cloud-artifacts/modern-deeplearning-pipeline/`

```
hf_cache/         ← /root/.cache/huggingface (transformers 다운로드 캐시)
test_fixtures/    ← /workspace/test-fixtures (tiny 모델 + dataset slice)
checkpoints/      ← /workspace/<repo>/checkpoints
outputs/          ← /workspace/<repo>/outputs
logs/             ← /tmp/install.log, /tmp/cloud_test.log
test_artifacts/   ← /tmp/mdp-test-artifacts (pytest.xml, cloud_test.log, memory/*.json, tests/<test-name>/...)
```

빈 디렉토리는 자동 skip — 첫 실행이면 그냥 cloud_test.sh로 진행. cloud_test.sh가 fixture를 자체 생성한다.

### Step 4 — Test

```bash
# 기본 quick suite
bash scripts/cloud_test.sh

# 단일 GPU fixture e2e
bash scripts/cloud_test.sh --suite gpu

# 2GPU 이상 distributed fixture e2e
bash scripts/cloud_test.sh --suite distributed
bash scripts/cloud_test.sh --suite distributed --allow-skip

# memory budget smoke
bash scripts/cloud_test.sh --suite memory --memory-profile cuda_24gb

# paid instance acceptance gate
bash scripts/cloud_test.sh --suite acceptance --memory-profile cuda_24gb

# paid instance 최종 sweep
bash scripts/cloud_test.sh --suite all --memory-profile cuda_24gb

# 임의 pytest 인자
bash scripts/cloud_test.sh tests/e2e/test_distributed_cpu.py -x
bash scripts/cloud_test.sh -m "distributed and not slow"
bash scripts/cloud_test.sh -k allreduce --tb=long

# fixture 캐싱 건너뛰기 (sync push로 이미 채웠을 때)
bash scripts/cloud_test.sh --skip-fixtures
```

cloud_test.sh가 수행:

1. `prepare_test_fixtures.py` (idempotent — 이미 있으면 skip)
2. `nvidia-smi` + `torch.cuda.device_count()` 출력
3. `MDP_TEST_FIXTURES=/workspace/test-fixtures`, 선택한 `MDP_MEMORY_BUDGET_PROFILE`, `MDP_TEST_ARTIFACT_DIR=/tmp/mdp-test-artifacts` 노출
4. pytest 실행 결과를 `/tmp/cloud_test.log`로 tee하고, structured artifact를 `/tmp/mdp-test-artifacts/`에 저장
   - `pytest.xml` — pytest JUnit XML
   - `cloud_test.log` — cloud test stdout/stderr 사본
   - `memory/*.json` — memory suite peak summary
   - `tests/<test-name>/...` — GPU fixture test별 checkpoint/output smoke artifact

### Step 5 — 산출물 회수

```bash
bash scripts/cloud_sync.sh pull
```

`hf_cache`, `test_fixtures`, `checkpoints`, `outputs`, `logs`, `test_artifacts` 모두 호스트로 회수. 다음 임대 시 push로 cache/checkpoint/output을 재사용하고, 실패 분석은 `test_artifacts/pytest.xml`, `test_artifacts/cloud_test.log`, `test_artifacts/memory/`, `test_artifacts/tests/`를 우선 확인한다.

### Step 6 — Teardown

```bash
bash scripts/cloud_teardown.sh
```

5분 내 pull 흔적 없으면 확인 프롬프트 (`--force` 우회). 종료 시 누적 비용 추정.

## 예상 비용

기본 워크플로 1회 (RTX A5000 × 2 @ ~$0.30/hr 기준):

| 단계 | 시간 | 비용 |
|---|---|---|
| provision (의존성 설치) | ~12분 | $0.06 |
| fixture 준비 (cold) | ~2분 | $0.01 |
| fixture 준비 (push로 hydrate) | ~30초 | $0.003 |
| quick suite | ~5~15분 | $0.025~0.075 |
| pull + teardown | ~3분 | $0.015 |
| **합계 (cold)** | ~22분 | **~$0.11** |
| **합계 (hydrate)** | ~20분 | **~$0.10** |

`--with-language` 사용 시 +15분 (flash-attn 빌드) → +$0.075.

## conftest에서 fixture 경로 활용 예시

```python
# tests/conftest.py 또는 tests/e2e/conftest.py
import os
from pathlib import Path
import pytest

@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    p = os.environ.get("MDP_TEST_FIXTURES")
    if not p:
        pytest.skip("MDP_TEST_FIXTURES not set (run via cloud_test.sh)")
    return Path(p)

@pytest.fixture(scope="session")
def smollm2(fixtures_dir):
    return fixtures_dir / "models" / "smollm2"

@pytest.fixture(scope="session")
def wikitext_tiny(fixtures_dir):
    return fixtures_dir / "data" / "wikitext-2-tiny" / "train.jsonl"
```

테스트에서:

```python
@pytest.mark.gpu
@pytest.mark.fixtures
def test_causal_lm_forward(smollm2):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(smollm2)).cuda()
    ...
```

## 트러블슈팅

### 1. fixture 다운로드 실패

원인: HF가 해당 repo를 일시 삭제/이동했거나 safetensors snapshot 구성이 바뀜.

대응: `prepare_test_fixtures.py`의 `TINY_MODELS` 딕셔너리 갱신. 모델 producer를 바꾸면 `tests/unit/test_gpu_fixture_contract.py`도 함께 갱신한다.

### 2. 데이터셋 다운로드 시 `datasets` 미설치

원인: provision이 `--with-language` 없이 끝나 `datasets`/`huggingface_hub`가 venv에 없음.

대응: cloud_test.sh가 자동으로 `uv pip install`을 호출하므로 보통 자동 처리. 안 되면 SSH로 들어가 수동 설치.

### 3. NCCL 초기화 실패

원인: NVLink 미보장 host에서 `NCCL_P2P_LEVEL=NVL` 강제.

대응: 테스트 코드에서 `NCCL_P2P_LEVEL` 강제 제거. 디버깅 시 `NCCL_DEBUG=INFO`.

### 4. `MDP_TEST_FIXTURES` not set

원인: cloud_test.sh를 거치지 않고 직접 pytest 실행.

대응: `MDP_TEST_FIXTURES=/workspace/test-fixtures pytest ...`로 직접 export.

### 5. memory suite가 즉시 실패

원인: `--suite memory`에 `--memory-profile` 또는 `MDP_MEMORY_BUDGET_PROFILE`이 없음.

대응: `bash scripts/cloud_test.sh --suite memory --memory-profile cuda_24gb`처럼 profile을 명시한다. profile 정의는 `tests/fixtures/memory_budgets.yaml`에 있다.

### 6. teardown 후 fixture 손실

원인: pull 없이 teardown.

대응: 다음번엔 sync pull → teardown 순서. teardown 5분 가드가 이를 방지.

### 7. state 파일 stale

원인: provision 후 인스턴스가 외부에서 destroy됨.

대응: `rm ~/.cache/cloud-runner/modern-deeplearning-pipeline/current.env` 후 다시 provision.

## 참조

- vault: `infra/{vastai,runpod}/cloud-runner-spec.md` — 4-동사 인터페이스 표준
- vault: `infra/{vastai,runpod}/pricing-search.md` — offer 검색 절차
- vault: `infra/{vastai,runpod}/instance-lifecycle.md` — 생성/SSH/destroy CLI 상세
- repo: `tests/e2e/models.py` — 자체 정의된 TinyVisionModel 등 (HF 의존 없는 fixture)
- repo: `scripts/prepare_test_fixtures.py` — fixture 캐싱 스크립트
