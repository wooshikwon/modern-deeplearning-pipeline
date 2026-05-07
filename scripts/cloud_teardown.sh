#!/bin/bash
# scripts/cloud_teardown.sh — Destroy the active cloud instance and report cost.
#
# Usage:
#   bash scripts/cloud_teardown.sh           # interactive: warns if no recent pull
#   bash scripts/cloud_teardown.sh --force   # skip the recent-pull warning
#
# Reads state from ~/.cache/cloud-runner/<repo>/current.env, runs provider-specific
# destroy, prints cost summary, and removes the state file.

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || die "must run inside a git repo"
REPO_NAME=$(basename "$REPO_ROOT")
STATE_FILE="$HOME/.cache/cloud-runner/$REPO_NAME/current.env"
[ -f "$STATE_FILE" ] || die "no active instance: $STATE_FILE not found"
# shellcheck disable=SC1090
source "$STATE_FILE"

# ────────────────────────────────────────────────────────────
# Recent-pull guard — destroy is irreversible
# ────────────────────────────────────────────────────────────
if [ "$FORCE" -ne 1 ]; then
  PULL_MARKER="$ARTIFACT_DIR/.last_pull"
  RECENT=0
  if [ -f "$PULL_MARKER" ]; then
    AGE=$(( $(date +%s) - $(stat -f %m "$PULL_MARKER" 2>/dev/null || stat -c %Y "$PULL_MARKER") ))
    [ "$AGE" -lt 300 ] && RECENT=1
  fi
  if [ "$RECENT" -ne 1 ]; then
    warn "no cloud_sync.sh pull within last 5min — outputs/checkpoints on instance will be LOST"
    read -r -p "destroy anyway? [y/N] " ans
    [ "$ans" = "y" ] || die "aborted (run cloud_sync.sh pull first, or --force to skip)"
  fi
fi

# ────────────────────────────────────────────────────────────
# Destroy + cost report
# ────────────────────────────────────────────────────────────
log "destroying $PROVIDER instance $INSTANCE_ID..."

if [ "$PROVIDER" = "vastai" ]; then
  VASTAI_FIRST='if type=="array" then .[0] else . end'
  COST_BEFORE=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
    | jq -r "$VASTAI_FIRST | .total_cost // 0")
  vastai destroy instance "$INSTANCE_ID" -y || warn "destroy command failed (may already be gone)"
  log "vastai instance destroyed"
  log "  estimated total cost: \$${COST_BEFORE}"
fi

if [ "$PROVIDER" = "runpod" ]; then
  RUNPOD_KEY=$(bw get password "RunPod API Key" --session "${BW_SESSION:-}" 2>/dev/null) \
    || warn "could not fetch RunPod API key (skipping cost query)"
  if [ -n "${RUNPOD_KEY:-}" ]; then
    POD=$(curl -sf -X POST https://api.runpod.io/graphql \
      -H "Authorization: Bearer $RUNPOD_KEY" -H "Content-Type: application/json" \
      -d "{\"query\":\"{ pod(input:{podId:\\\"$INSTANCE_ID\\\"}) { costPerHr lastStatusChange } }\"}" 2>/dev/null || echo "{}")
    RATE=$(echo "$POD" | jq -r '.data.pod.costPerHr // 0')
    HOURS=$(python3 -c "
from datetime import datetime, timezone
t = '$CREATED_AT'
try:
    start = datetime.fromisoformat(t.replace('Z','+00:00'))
    dur = (datetime.now(timezone.utc) - start).total_seconds() / 3600
    print(f'{dur:.2f}')
except Exception:
    print('0')")
    EST=$(python3 -c "print(f'{float(\"$RATE\") * float(\"$HOURS\"):.2f}')" 2>/dev/null || echo "?")
    curl -sf -X POST https://api.runpod.io/graphql \
      -H "Authorization: Bearer $RUNPOD_KEY" -H "Content-Type: application/json" \
      -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$INSTANCE_ID\\\"}) }\"}" >/dev/null \
      || warn "terminate command failed (may already be gone)"
    log "runpod pod terminated"
    log "  rate: \$$RATE/hr × ${HOURS}h ≈ \$$EST"
  else
    warn "skipping cost calc (no API key)"
  fi
fi

# ────────────────────────────────────────────────────────────
# Cleanup state
# ────────────────────────────────────────────────────────────
rm -f "$STATE_FILE"
log "state file removed: $STATE_FILE"
log "teardown complete"
