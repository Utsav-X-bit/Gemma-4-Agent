#!/usr/bin/env bash
set -euo pipefail

WORK="${WORK:-$HOME/gemma4}"
DATA="${DATA:-$WORK/data}"
RUNS="${RUNS:-$WORK/runs}"
AGENT="${AGENT:-$WORK/submission-primary}"
VENV="${VENV:-$WORK/harness-venv}"
TASK_ID="${TASK_ID:-fastapi_15661}"
SANDBOX="${SANDBOX:-auto}"
export WORK DATA RUNS AGENT VENV TASK_ID SANDBOX

if [ ! -f "$AGENT/agent.yaml" ]; then
  echo "Missing $AGENT/agent.yaml. Copy submission-primary/ to $AGENT first."
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"
mkdir -p "$RUNS"

if [ "$SANDBOX" = "auto" ]; then
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    SANDBOX=docker
  else
    SANDBOX=subprocess
  fi
fi

if [ "$SANDBOX" = "subprocess" ]; then
  echo
  echo "WARNING: using --sandbox subprocess."
  echo "This is NOT scorer-identical. Use it for the Phase 1 smoke test only;"
  echo "do not treat its timings or gold-patch behavior as the scorer's."
  echo
fi

echo "== One-task smoke test: $TASK_ID =="
echo "Agent: $AGENT"
echo "Sandbox: $SANDBOX"
echo "Budget: 24 calls / 5 min (from eval_config.yaml; CLI budget flags omitted)"

ARGS=(
  eval
  --tasks "$DATA/tasks.jsonl"
  --snapshots-dir "$DATA/snapshots"
  --submission-dir "$AGENT"
  --results-dir "$RUNS/smoke-$TASK_ID"
  --sandbox "$SANDBOX"
  --task-id "$TASK_ID"
  --concurrency 1
  --display auto
)

if [ -n "${MODELS_YAML:-}" ]; then
  ARGS+=(--models-yaml "$MODELS_YAML")
fi

swegemma "${ARGS[@]}"

echo "$SANDBOX" > "$RUNS/smoke-$TASK_ID/sandbox.txt"

echo
echo "== Artifacts =="
find "$RUNS/smoke-$TASK_ID" -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
echo
echo "Look for: summary.json, patches/$TASK_ID.patch, test_outputs/$TASK_ID.log, traces/trace_$TASK_ID.json"
echo "Pass condition: a non-empty patch OR a clear harness/test failure you can explain."
