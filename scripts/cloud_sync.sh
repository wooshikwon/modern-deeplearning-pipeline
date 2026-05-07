#!/bin/bash
# scripts/cloud_sync.sh — Sync artifacts between host ARTIFACT_DIR and the active cloud instance.
#
# Usage:
#   bash scripts/cloud_sync.sh pull
#   bash scripts/cloud_sync.sh push
#
# Reads state from ~/.cache/cloud-runner/<repo>/current.env (written by cloud_provision.sh).

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
die() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

[ "$#" -eq 1 ] || die "usage: $0 pull|push"
MODE="$1"
case "$MODE" in pull|push) ;; *) die "mode must be pull or push" ;; esac

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || die "must run inside a git repo"
REPO_NAME=$(basename "$REPO_ROOT")
STATE_FILE="$HOME/.cache/cloud-runner/$REPO_NAME/current.env"
[ -f "$STATE_FILE" ] || die "no active instance: $STATE_FILE not found (run cloud_provision.sh first)"
# shellcheck disable=SC1090
source "$STATE_FILE"

mkdir -p "$ARTIFACT_DIR"/{hf_cache,test_fixtures,checkpoints,outputs,logs,test_artifacts}

RSYNC_SSH="ssh -i $SSH_KEY -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=20"

# Remote layout (mdp-specific — keep in sync with docs/cloud-testing.md)
REMOTE_HF=/root/.cache/huggingface
REMOTE_FIXTURES=/workspace/test-fixtures
REMOTE_CKPT=/workspace/$REPO/checkpoints
REMOTE_OUT=/workspace/$REPO/outputs
REMOTE_TEST_ARTIFACTS=/tmp/mdp-test-artifacts
REMOTE_LOG_GLOB="/tmp/install.log /tmp/cloud_test.log"

if [ "$MODE" = "push" ]; then
  log "push: $ARTIFACT_DIR -> $SSH_HOST:$SSH_PORT"
  ssh -i "$SSH_KEY" -p "$SSH_PORT" root@"$SSH_HOST" \
    -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
    "mkdir -p $REMOTE_HF $REMOTE_FIXTURES $REMOTE_CKPT $REMOTE_OUT $REMOTE_TEST_ARTIFACTS"
  [ -d "$ARTIFACT_DIR/hf_cache" ]       && rsync -az --partial -e "$RSYNC_SSH" "$ARTIFACT_DIR/hf_cache/"       "root@$SSH_HOST:$REMOTE_HF/"       || true
  [ -d "$ARTIFACT_DIR/test_fixtures" ]  && rsync -az --partial -e "$RSYNC_SSH" "$ARTIFACT_DIR/test_fixtures/"  "root@$SSH_HOST:$REMOTE_FIXTURES/" || true
  [ -d "$ARTIFACT_DIR/checkpoints" ]    && rsync -az --partial -e "$RSYNC_SSH" "$ARTIFACT_DIR/checkpoints/"    "root@$SSH_HOST:$REMOTE_CKPT/"     || true
  [ -d "$ARTIFACT_DIR/outputs" ]        && rsync -az --partial -e "$RSYNC_SSH" "$ARTIFACT_DIR/outputs/"        "root@$SSH_HOST:$REMOTE_OUT/"      || true
fi

if [ "$MODE" = "pull" ]; then
  log "pull: $SSH_HOST:$SSH_PORT -> $ARTIFACT_DIR"
  rsync -az --partial -e "$RSYNC_SSH" "root@$SSH_HOST:$REMOTE_HF/"       "$ARTIFACT_DIR/hf_cache/"      || true
  rsync -az --partial -e "$RSYNC_SSH" "root@$SSH_HOST:$REMOTE_FIXTURES/" "$ARTIFACT_DIR/test_fixtures/" || true
  rsync -az --partial -e "$RSYNC_SSH" "root@$SSH_HOST:$REMOTE_CKPT/"     "$ARTIFACT_DIR/checkpoints/"   || true
  rsync -az --partial -e "$RSYNC_SSH" "root@$SSH_HOST:$REMOTE_OUT/"      "$ARTIFACT_DIR/outputs/"       || true
  rsync -az --partial -e "$RSYNC_SSH" "root@$SSH_HOST:$REMOTE_TEST_ARTIFACTS/" "$ARTIFACT_DIR/test_artifacts/" || true
  for f in $REMOTE_LOG_GLOB; do
    rsync -az -e "$RSYNC_SSH" "root@$SSH_HOST:$f" "$ARTIFACT_DIR/logs/" 2>/dev/null || true
  done
  # Touch a marker so teardown can detect recent pull
  touch "$ARTIFACT_DIR/.last_pull"
fi

log "$MODE complete"
