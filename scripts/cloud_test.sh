#!/bin/bash
# scripts/cloud_test.sh — Run mdp e2e tests on the active cloud instance.
#
# Pipeline:
#   1. Cache tiny test fixtures on the instance (idempotent, ~30s if hydrated)
#   2. Run multi-GPU pytest with MDP_TEST_FIXTURES env exposed
#
# Usage:
#   bash scripts/cloud_test.sh                              # default selection
#   bash scripts/cloud_test.sh tests/e2e/test_distributed_cpu.py
#   bash scripts/cloud_test.sh -k allreduce -x --tb=long
#   bash scripts/cloud_test.sh --skip-fixtures              # assume already cached
#
# Default test selection (when no path given) picks GPU-meaningful e2e:
#   - tests/e2e/test_distributed_cpu.py     (gloo→nccl path verification)
#   - tests/e2e/test_distributed_rl.py      (RL distributed)
#   - tests/e2e/test_factory_e2e.py         (factory routing on GPU)
#   - tests/e2e/test_inference.py           (inference path)
#   - tests/e2e/test_inference_hooks_device_map.py
#   - tests/e2e/test_pipeline_e2e.py        (end-to-end recipe runs)
#   - tests/e2e/test_callbacks.py           (callback hooks)
# Plus any test marked @pytest.mark.distributed or @pytest.mark.gpu.
#
# Reads state from ~/.cache/cloud-runner/<repo>/current.env.

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
die() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || die "must run inside a git repo"
REPO_NAME=$(basename "$REPO_ROOT")
STATE_FILE="$HOME/.cache/cloud-runner/$REPO_NAME/current.env"
[ -f "$STATE_FILE" ] || die "no active instance (run cloud_provision.sh first)"
# shellcheck disable=SC1090
source "$STATE_FILE"

SKIP_FIXTURES=0
PYTEST_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-fixtures) SKIP_FIXTURES=1; shift ;;
    *) PYTEST_ARGS+=("$1"); shift ;;
  esac
done

# Default test selection if none given
DEFAULT_TARGETS=(
  "tests/e2e/test_distributed_cpu.py"
  "tests/e2e/test_distributed_rl.py"
  "tests/e2e/test_factory_e2e.py"
  "tests/e2e/test_inference.py"
  "tests/e2e/test_inference_hooks_device_map.py"
  "tests/e2e/test_pipeline_e2e.py"
  "tests/e2e/test_callbacks.py"
)
if [ "${#PYTEST_ARGS[@]}" -eq 0 ]; then
  PYTEST_ARGS=("${DEFAULT_TARGETS[@]}")
fi

WORKSPACE=/workspace/$REPO_NAME
SSH_BASE="ssh -i $SSH_KEY -p $SSH_PORT root@$SSH_HOST -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=15"

# ────────────────────────────────────────────────────────────
# Step 1 — Test fixtures (tiny HF models + small dataset slices)
# ────────────────────────────────────────────────────────────
if [ "$SKIP_FIXTURES" -eq 1 ]; then
  log "skipping fixture preparation (--skip-fixtures)"
else
  log "preparing test fixtures on instance (~30s if cached, ~3min cold)..."
  FIXTURE_CMD="set -e
export PATH=\"\$HOME/.local/bin:\$PATH\"
cd $WORKSPACE
source .venv/bin/activate
# Ensure deps for fixture cache + verify + downstream e2e are installed.
# pytest: not in mdp's optional-dependencies, but required to run any e2e.
# transformers/peft/tokenizers: needed by most mdp e2e (factory, inference, etc).
python -c 'import pytest, datasets, huggingface_hub, transformers, peft, tokenizers, safetensors' 2>/dev/null \\
  || uv pip install \\
       pytest 'datasets>=2.16' 'huggingface_hub>=0.20' \\
       'transformers>=4.36' 'tokenizers>=0.15' 'peft>=0.10,<0.19.0' \\
       safetensors >/dev/null
python scripts/prepare_test_fixtures.py --target-dir /workspace/test-fixtures
"
  $SSH_BASE "$FIXTURE_CMD" || die "fixture preparation failed"
fi

# ────────────────────────────────────────────────────────────
# Step 2 — pytest with fixture path exposed
# ────────────────────────────────────────────────────────────
log "running pytest on $SSH_HOST:$SSH_PORT ($N_GPUS x $GPU_NAME)"
log "  targets: ${PYTEST_ARGS[*]}"

REMOTE_CMD="
set -e
cd $WORKSPACE
source .venv/bin/activate
export MDP_TEST_FIXTURES=/workspace/test-fixtures
export HF_HOME=/root/.cache/huggingface
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -$N_GPUS
python -c 'import torch; print(\"CUDA devices:\", torch.cuda.device_count(), \"| bf16:\", torch.cuda.is_bf16_supported())'
pytest ${PYTEST_ARGS[*]} --tb=short -ra 2>&1 | tee /tmp/cloud_test.log
exit \${PIPESTATUS[0]}
"

set +e
$SSH_BASE "$REMOTE_CMD"
RC=$?
set -e

if [ "$RC" -eq 0 ]; then
  log "tests passed"
else
  log "tests failed (rc=$RC) — full log: $WORKSPACE/../  /tmp/cloud_test.log on instance"
  log "  pull with: bash scripts/cloud_sync.sh pull"
fi
exit "$RC"
