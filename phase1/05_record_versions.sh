#!/usr/bin/env bash
set -euo pipefail

WORK="${WORK:-$HOME/gemma4}"
VENV="${VENV:-$WORK/harness-venv}"
ZIP="${ZIP:-$WORK/submission-primary.zip}"
export WORK VENV ZIP

mkdir -p "$WORK/env"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

{
  echo "== UTC =="
  date -u
  echo
  echo "== GPU =="
  nvidia-smi
  echo
  echo "== Python =="
  python --version
  echo
  echo "== Harness versions =="
  python - <<'PY'
import swegemma, adk_submission
print("swegemma", getattr(swegemma, "__version__", "unknown"))
print("adk_submission", getattr(adk_submission, "__version__", "unknown"))
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as e:
    print("vllm import failed:", e)
PY
  echo
  echo "== Sandbox backend =="
  cat "$WORK/env/sandbox-backend.txt" 2>/dev/null || echo "sandbox-backend.txt missing"
  echo
  echo "== Docker image =="
  docker image inspect swebench-sandbox:latest --format '{{.Id}}' 2>/dev/null || echo "docker unavailable or image missing"
  echo
  echo "== Wheelhouse =="
  find "$WORK/wheelhouse" -maxdepth 1 -name '*.whl' -printf '%f\n' | sort | sha256sum
  echo
  echo "== Submission zip =="
  if [ -f "$ZIP" ]; then sha256sum "$ZIP"; else echo "missing $ZIP"; fi
} | tee "$WORK/env/phase1-versions.txt"

echo "Recorded $WORK/env/phase1-versions.txt"
