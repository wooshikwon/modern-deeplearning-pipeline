#!/bin/bash
# scripts/cloud_test.sh — Run mdp e2e tests on the active cloud instance.
#
# Pipeline:
#   1. Cache tiny test fixtures on the instance (idempotent, ~30s if hydrated)
#   2. Run multi-GPU pytest with MDP_TEST_FIXTURES env exposed
#
# Usage:
#   bash scripts/cloud_test.sh                              # --suite quick
#   bash scripts/cloud_test.sh --suite gpu
#   bash scripts/cloud_test.sh --suite distributed --allow-skip
#   bash scripts/cloud_test.sh --suite memory --memory-profile cuda_24gb
#   bash scripts/cloud_test.sh --suite acceptance --memory-profile cuda_24gb
#   bash scripts/cloud_test.sh --suite all --memory-profile cuda_24gb
#   bash scripts/cloud_test.sh tests/e2e/test_distributed_cpu.py
#   bash scripts/cloud_test.sh -k allreduce -x --tb=long
#   bash scripts/cloud_test.sh --skip-fixtures              # assume already cached
#   bash scripts/cloud_test.sh --suite gpu --dry-run        # print selected pytest args
#
# Default suite is quick. Explicit pytest args still run as-is.
#
# Reads state from ~/.cache/cloud-runner/<repo>/current.env.

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
die() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

SUITE=quick
MEMORY_PROFILE="${MDP_MEMORY_BUDGET_PROFILE:-}"
ALLOW_SKIP=0
SKIP_FIXTURES=0
DRY_RUN=0
PYTEST_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-fixtures) SKIP_FIXTURES=1; shift ;;
    --suite)
      [ "$#" -ge 2 ] || die "--suite requires a value"
      SUITE="$2"; shift 2 ;;
    --suite=*)
      SUITE="${1#--suite=}"; shift ;;
    --memory-profile)
      [ "$#" -ge 2 ] || die "--memory-profile requires a value"
      MEMORY_PROFILE="$2"; shift 2 ;;
    --memory-profile=*)
      MEMORY_PROFILE="${1#--memory-profile=}"; shift ;;
    --allow-skip) ALLOW_SKIP=1; shift ;;
    *) PYTEST_ARGS+=("$1"); shift ;;
  esac
done

case "$SUITE" in
  quick|gpu|distributed|memory|acceptance|all) ;;
  *) die "unknown suite: $SUITE (expected quick|gpu|distributed|memory|acceptance|all)" ;;
esac

if [ "$DRY_RUN" -eq 0 ]; then
  REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || die "must run inside a git repo"
  REPO_NAME=$(basename "$REPO_ROOT")
  STATE_FILE="$HOME/.cache/cloud-runner/$REPO_NAME/current.env"
  [ -f "$STATE_FILE" ] || die "no active instance (run cloud_provision.sh first)"
  # shellcheck disable=SC1090
  source "$STATE_FILE"

  WORKSPACE=/workspace/$REPO_NAME
  SSH_BASE="ssh -i $SSH_KEY -p $SSH_PORT root@$SSH_HOST -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=15"
fi

if [ "${#PYTEST_ARGS[@]}" -eq 0 ]; then
  case "$SUITE" in
    quick)
      PYTEST_ARGS=("tests/unit/test_gpu_fixture_contract.py" "tests/e2e/test_gpu_sft.py" "tests/e2e/test_gpu_inference.py" "-m" "not distributed and not memory")
      ;;
    gpu)
      PYTEST_ARGS=("-m" "gpu and fixtures and not distributed and not memory")
      ;;
    distributed)
      if [ "$DRY_RUN" -eq 0 ] && [ "${N_GPUS:-0}" -lt 2 ]; then
        if [ "$ALLOW_SKIP" -eq 1 ]; then
          log "skipping distributed suite: requires >=2 GPUs (have ${N_GPUS:-0})"
          exit 0
        fi
        die "distributed suite requires >=2 GPUs (have ${N_GPUS:-0}); pass --allow-skip to skip"
      fi
      PYTEST_ARGS=("-m" "distributed and fixtures")
      ;;
    memory)
      [ -n "$MEMORY_PROFILE" ] || die "memory suite requires --memory-profile or MDP_MEMORY_BUDGET_PROFILE"
      PYTEST_ARGS=("-m" "memory")
      ;;
    acceptance)
      [ -n "$MEMORY_PROFILE" ] || die "acceptance suite requires --memory-profile or MDP_MEMORY_BUDGET_PROFILE"
      if [ "$DRY_RUN" -eq 0 ] && [ "${N_GPUS:-0}" -lt 2 ]; then
        if [ "$ALLOW_SKIP" -eq 1 ]; then
          log "skipping acceptance suite: requires >=2 GPUs (have ${N_GPUS:-0})"
          exit 0
        fi
        die "acceptance suite requires >=2 GPUs (have ${N_GPUS:-0}); pass --allow-skip to skip"
      fi
      PYTEST_ARGS=(
        "tests/e2e/test_gpu_cli_matrix.py"
        "tests/e2e/test_gpu_distributed_cli_matrix.py"
        "tests/e2e/test_gpu_memory_budget.py"
      )
      ;;
    all)
      [ -n "$MEMORY_PROFILE" ] || log "memory tests will skip: no memory profile selected"
      PYTEST_ARGS=("-m" "not slow")
      ;;
  esac
fi
PYTEST_REMOTE_ARGS=$(printf " %q" "${PYTEST_ARGS[@]}")

if [ "$DRY_RUN" -eq 1 ]; then
  log "dry-run cloud pytest selection"
  log "  suite: $SUITE"
  log "  targets: ${PYTEST_ARGS[*]}"
  if [ -n "$MEMORY_PROFILE" ]; then
    log "  memory profile: $MEMORY_PROFILE"
  fi
  exit 0
fi

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
# transformers/peft/tokenizers: needed by most mdp e2e (materializer, inference, etc).
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
log "  suite: $SUITE"
log "  targets: ${PYTEST_ARGS[*]}"

REMOTE_CMD="
set -e
cd $WORKSPACE
source .venv/bin/activate
export MDP_TEST_FIXTURES=/workspace/test-fixtures
export HF_HOME=/root/.cache/huggingface
export MDP_MEMORY_BUDGET_PROFILE=$MEMORY_PROFILE
export MDP_TEST_ARTIFACT_DIR=/tmp/mdp-test-artifacts
mkdir -p \$MDP_TEST_ARTIFACT_DIR
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -$N_GPUS
python -c 'import torch; print(\"CUDA devices:\", torch.cuda.device_count(), \"| bf16:\", torch.cuda.is_bf16_supported())'
pytest$PYTEST_REMOTE_ARGS --tb=short -ra --junitxml=\$MDP_TEST_ARTIFACT_DIR/pytest.xml 2>&1 | tee /tmp/cloud_test.log
rc=\${PIPESTATUS[0]}
cp /tmp/cloud_test.log \$MDP_TEST_ARTIFACT_DIR/cloud_test.log
exit \$rc
"

set +e
$SSH_BASE "$REMOTE_CMD"
RC=$?
set -e

if [ "$RC" -eq 0 ]; then
  log "tests passed"
else
  log "tests failed (rc=$RC) — full log: /tmp/cloud_test.log on instance"
  log "  pull with: bash scripts/cloud_sync.sh pull"
fi
exit "$RC"
