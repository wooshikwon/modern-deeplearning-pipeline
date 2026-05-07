#!/bin/bash
# scripts/cloud_provision.sh — Provision a cloud GPU instance for mdp distributed testing.
#
# Usage:
#   bash scripts/cloud_provision.sh vastai --offer-id <ID> [--gpu-name <MODEL>] [--n-gpus <N>] [--disk <GB>] [--with-language]
#   bash scripts/cloud_provision.sh runpod --gpu-name "<NAME>" --n-gpus <N> --tier <secure|community> [--disk <GB>] [--volume <GB>] [--with-language]
#
# Provider-agnostic spec: vault infra/{vastai,runpod}/cloud-runner-spec.md

set -euo pipefail

# ────────────────────────────────────────────────────────────
# Common
# ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || die "must run inside a git repo"
REPO_NAME=$(basename "$REPO_ROOT")
STATE_DIR="$HOME/.cache/cloud-runner/$REPO_NAME"
STATE_FILE="$STATE_DIR/current.env"
ARTIFACT_DIR="${ARTIFACT_DIR:-$HOME/cloud-artifacts/$REPO_NAME}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/cloud_runner}"
DEFAULT_IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"  # base image; venv installs torch 2.6 on top
TORCH_WHEEL_URL="https://download.pytorch.org/whl/cu124"

# ────────────────────────────────────────────────────────────
# Provider selection
# ────────────────────────────────────────────────────────────
[ "$#" -ge 1 ] || die "usage: $0 vastai|runpod [args]"
PROVIDER="$1"; shift
case "$PROVIDER" in
  vastai|runpod) ;;
  *) die "PROVIDER must be vastai or runpod (got: $PROVIDER)" ;;
esac

# Provider-specific args
OFFER_ID=""; GPU_NAME=""; N_GPUS=""; TIER="secure"; DISK_GB=200; VOLUME_GB=200
IMAGE="$DEFAULT_IMAGE"; WITH_LANGUAGE=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --offer-id) OFFER_ID="$2"; shift 2 ;;
    --gpu-name) GPU_NAME="$2"; shift 2 ;;
    --n-gpus)   N_GPUS="$2"; shift 2 ;;
    --tier)     TIER="$2"; shift 2 ;;
    --disk)     DISK_GB="$2"; shift 2 ;;
    --volume)   VOLUME_GB="$2"; shift 2 ;;
    --image)    IMAGE="$2"; shift 2 ;;
    --with-language) WITH_LANGUAGE=1; shift ;;
    *) die "unknown flag: $1" ;;
  esac
done

if [ "$PROVIDER" = "vastai" ]; then
  [ -n "$OFFER_ID" ] || die "--offer-id required for vastai (use vault infra/vastai/pricing-search.md to find one)"
fi
if [ "$PROVIDER" = "runpod" ]; then
  [ -n "$GPU_NAME" ] || die "--gpu-name required for runpod"
  [ -n "$N_GPUS" ] || die "--n-gpus required for runpod"
fi

# ────────────────────────────────────────────────────────────
# Pre-flight checks (BEFORE any paid action)
# ────────────────────────────────────────────────────────────
log "pre-flight checks..."
[ -f "$SSH_KEY" ] || die "SSH key not found: $SSH_KEY"
ssh-add -l >/dev/null 2>&1 || die "ssh-agent not running or empty"
ssh-add -l | grep -q "$(ssh-keygen -lf "$SSH_KEY" | awk '{print $2}')" \
  || die "ssh-agent missing $SSH_KEY (run: ssh-add $SSH_KEY)"
[ -n "${BW_SESSION:-}" ] || die "BW_SESSION unset (run: export BW_SESSION=\$(bw unlock --raw))"

if [ "$PROVIDER" = "vastai" ]; then
  command -v vastai >/dev/null || die "vastai CLI not found (pip install vastai)"
  vastai show user >/dev/null 2>&1 || die "vastai auth failed (vastai set api-key ...)"
fi
if [ "$PROVIDER" = "runpod" ]; then
  command -v jq >/dev/null || die "jq required"
  RUNPOD_KEY=$(bw get password "RunPod API Key" --session "$BW_SESSION" 2>/dev/null) \
    || die "RunPod API Key fetch from Bitwarden failed"
  curl -sf -X POST https://api.runpod.io/graphql \
    -H "Authorization: Bearer $RUNPOD_KEY" \
    -H "Content-Type: application/json" \
    -d '{"query":"{ myself { id } }"}' \
    | jq -e '.data.myself.id' >/dev/null \
    || die "RunPod API auth failed"
fi

mkdir -p "$STATE_DIR" "$ARTIFACT_DIR"
[ -f "$STATE_FILE" ] && {
  warn "existing state file: $STATE_FILE"
  warn "  if a previous instance is still alive, run cloud_teardown.sh first"
  read -r -p "  overwrite and continue? [y/N] " ans
  [ "$ans" = "y" ] || die "aborted"
}
log "pre-flight OK"

# ────────────────────────────────────────────────────────────
# Provision — provider branch
# ────────────────────────────────────────────────────────────
INSTANCE_ID=""; SSH_HOST=""; SSH_PORT=""

if [ "$PROVIDER" = "vastai" ]; then
  log "creating vastai instance from offer $OFFER_ID..."
  RESP=$(echo "y" | vastai create instance "$OFFER_ID" \
    --image "$IMAGE" --disk "$DISK_GB" --ssh --direct --raw 2>/dev/null)
  INSTANCE_ID=$(echo "$RESP" | jq -r '.new_contract')
  [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "null" ] || die "create failed: $RESP"
  log "instance created: $INSTANCE_ID"

  log "waiting for running status..."
  # vastai show instance --raw can return either an array (older) or a single object (newer).
  VASTAI_FIRST='if type=="array" then .[0] else . end'
  for i in $(seq 1 60); do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
      | jq -r "$VASTAI_FIRST | .actual_status // \"\"")
    echo "  [$i/60] status=$STATUS"
    [ "$STATUS" = "running" ] && break
    [ "$i" -eq 60 ] && {
      vastai destroy instance "$INSTANCE_ID" -y || true
      die "instance did not reach running in 10min (destroyed)"
    }
    sleep 10
  done

  SSH_URL=$(vastai ssh-url "$INSTANCE_ID" 2>/dev/null)
  SSH_HOST=$(echo "$SSH_URL" | sed 's|ssh://root@\(.*\):\(.*\)|\1|')
  SSH_PORT=$(echo "$SSH_URL" | sed 's|ssh://root@\(.*\):\(.*\)|\2|')
fi

if [ "$PROVIDER" = "runpod" ]; then
  log "creating runpod pod ($N_GPUS x $GPU_NAME, $TIER)..."
  CLOUD_TYPE=$(echo "$TIER" | tr a-z A-Z)
  POD_NAME="$REPO_NAME-$(date +%s)"
  QUERY=$(cat <<EOF
mutation { podFindAndDeployOnDemand(input: {
  cloudType: $CLOUD_TYPE,
  gpuCount: $N_GPUS,
  gpuTypeId: "$GPU_NAME",
  name: "$POD_NAME",
  imageName: "$IMAGE",
  containerDiskInGb: $DISK_GB,
  volumeInGb: $VOLUME_GB,
  volumeMountPath: "/workspace",
  ports: "22/tcp"
}) { id } }
EOF
)
  CREATE_PAYLOAD=$(jq -n --arg q "$QUERY" '{query: $q}')
  RESP=$(curl -sf -X POST https://api.runpod.io/graphql \
    -H "Authorization: Bearer $RUNPOD_KEY" \
    -H "Content-Type: application/json" \
    -d "$CREATE_PAYLOAD") || die "runpod create failed"
  INSTANCE_ID=$(echo "$RESP" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
  [ -n "$INSTANCE_ID" ] || die "no podId in response: $RESP"
  log "pod created: $INSTANCE_ID"

  log "waiting for RUNNING status..."
  for i in $(seq 1 24); do
    POD=$(curl -sf -X POST https://api.runpod.io/graphql \
      -H "Authorization: Bearer $RUNPOD_KEY" -H "Content-Type: application/json" \
      -d "{\"query\":\"{ pod(input:{podId:\\\"$INSTANCE_ID\\\"}) { desiredStatus runtime { ports { ip publicPort privatePort } } } }\"}")
    STATUS=$(echo "$POD" | jq -r '.data.pod.desiredStatus // ""')
    echo "  [$i/24] status=$STATUS"
    if [ "$STATUS" = "RUNNING" ]; then
      SSH_HOST=$(echo "$POD" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22) | .ip')
      SSH_PORT=$(echo "$POD" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22) | .publicPort')
      [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] && break
    fi
    [ "$i" -eq 24 ] && {
      curl -sf -X POST https://api.runpod.io/graphql \
        -H "Authorization: Bearer $RUNPOD_KEY" -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$INSTANCE_ID\\\"}) }\"}" >/dev/null || true
      die "pod did not reach RUNNING in 4min (terminated)"
    }
    sleep 10
  done
fi

# ────────────────────────────────────────────────────────────
# SSH reachability + repo clone + venv + base deps
# ────────────────────────────────────────────────────────────
SSH_BASE="ssh -i $SSH_KEY -p $SSH_PORT root@$SSH_HOST -o StrictHostKeyChecking=no -o ConnectTimeout=20"

log "SSH reachability check..."
for i in $(seq 1 12); do
  $SSH_BASE 'echo READY' 2>/dev/null | grep -q READY && break
  [ "$i" -eq 12 ] && die "SSH not reachable after 2min"
  sleep 10
done
log "SSH reachable: $SSH_HOST:$SSH_PORT"

REPO_URL=$(git -C "$REPO_ROOT" config --get remote.origin.url)
log "cloning $REPO_URL on instance + installing base deps..."

ssh -A -i "$SSH_KEY" -p "$SSH_PORT" root@"$SSH_HOST" \
  -o StrictHostKeyChecking=no -o ConnectTimeout=30 \
  "set -e
   mkdir -p ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null || true
   mkdir -p /workspace && cd /workspace
   if [ ! -d $REPO_NAME/.git ]; then git clone $REPO_URL $REPO_NAME; fi
   echo CLONED: \$(cd /workspace/$REPO_NAME && git log --oneline -1)"

EXTRA_INSTALL=""
if [ "$WITH_LANGUAGE" = "1" ]; then
  EXTRA_INSTALL=$(cat <<'EXTRA'
uv pip install packaging setuptools wheel
uv pip install flash-attn==2.8.3 --no-build-isolation
uv pip install -e ".[language]"
EXTRA
)
fi

INSTALL_SCRIPT=$(cat <<EOF
#!/bin/bash
set -e
export PATH="\$HOME/.local/bin:\$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="\$HOME/.local/bin:\$PATH"
cd /workspace/$REPO_NAME
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url $TORCH_WHEEL_URL
uv pip install -e .
$EXTRA_INSTALL
python -c "import torch; assert torch.cuda.is_available(); print('CUDA_GPUS=', torch.cuda.device_count())"
echo INSTALL_DONE
EOF
)

# Ship install script to instance and run in background
echo "$INSTALL_SCRIPT" | $SSH_BASE 'cat > /tmp/install.sh && chmod +x /tmp/install.sh'
$SSH_BASE 'nohup bash /tmp/install.sh > /tmp/install.log 2>&1 & echo PID:$!'

log "polling install (target: INSTALL_DONE)..."
INSTALL_TIMEOUT=$([ "$WITH_LANGUAGE" = "1" ] && echo 60 || echo 24)
for i in $(seq 1 "$INSTALL_TIMEOUT"); do
  TAIL=$($SSH_BASE 'tail -1 /tmp/install.log 2>/dev/null' 2>/dev/null || echo "")
  echo "  [$i/$INSTALL_TIMEOUT] $TAIL"
  echo "$TAIL" | grep -q INSTALL_DONE && break
  echo "$TAIL" | grep -qiE "ModuleNotFoundError|Traceback|build failed|No matching distribution" && {
    $SSH_BASE 'cat /tmp/install.log' || true
    die "install failed (see log above)"
  }
  [ "$i" -eq "$INSTALL_TIMEOUT" ] && die "install timeout"
  sleep 30
done
log "install OK"

# ────────────────────────────────────────────────────────────
# Write state file
# ────────────────────────────────────────────────────────────
cat > "$STATE_FILE" <<EOF
PROVIDER=$PROVIDER
INSTANCE_ID=$INSTANCE_ID
SSH_HOST=$SSH_HOST
SSH_PORT=$SSH_PORT
SSH_KEY=$SSH_KEY
GPU_NAME=$GPU_NAME
N_GPUS=$N_GPUS
TIER=$TIER
DISK_GB=$DISK_GB
VOLUME_GB=$VOLUME_GB
IMAGE=$IMAGE
CREATED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
REPO=$REPO_NAME
ARTIFACT_DIR=$ARTIFACT_DIR
WITH_LANGUAGE=$WITH_LANGUAGE
EOF
chmod 600 "$STATE_FILE"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Instance ready${NC}"
echo -e "${GREEN}========================================${NC}"
echo "  Provider     : $PROVIDER"
echo "  Instance ID  : $INSTANCE_ID"
echo "  SSH          : ssh -i $SSH_KEY -p $SSH_PORT root@$SSH_HOST"
echo "  Workspace    : /workspace/$REPO_NAME"
echo "  Venv         : /workspace/$REPO_NAME/.venv"
echo "  State file   : $STATE_FILE"
echo ""
echo "  Next:"
echo "    bash scripts/cloud_sync.sh push        # hydrate (optional)"
echo "    bash scripts/cloud_test.sh             # run distributed tests"
echo "    bash scripts/cloud_sync.sh pull        # collect outputs"
echo "    bash scripts/cloud_teardown.sh         # destroy + bill"
echo ""
