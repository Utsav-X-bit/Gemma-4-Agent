#!/usr/bin/env bash
set -euo pipefail

WORK="${WORK:-$HOME/gemma4}"
WHEELS="${WHEELS:-$WORK/wheelhouse}"
VENV="${VENV:-$WORK/harness-venv}"
export WORK WHEELS VENV

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it in user space first:"
  echo '  curl -LsSf https://astral.sh/uv/install.sh | sh'
  echo '  source ~/.local/bin/env'
  exit 1
fi

echo "== uv =="
uv --version

if ! ls "$WHEELS"/*.whl >/dev/null 2>&1; then
  echo "No wheels in $WHEELS. Run 01_get_data.sh first."
  exit 1
fi

echo "== Provisioning Python 3.12 with uv =="
uv python install 3.12

echo "== Creating venv =="
rm -rf "$VENV"
uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

echo "== Staging wheels =="
rm -rf /tmp/gemma4-wheelhouse
mkdir -p /tmp/gemma4-wheelhouse
for w in "$WHEELS"/*.whl; do
  case "$(basename "$w")" in *cutlass*) continue;; esac
  name="$(basename "$w")"
  case "$name" in
    *cu128*+*) ;;
    *cu128*) name="${name/cu128/+cu128}" ;;
  esac
  ln -sf "$w" "/tmp/gemma4-wheelhouse/$name" 2>/dev/null || cp "$w" "/tmp/gemma4-wheelhouse/$name"
done

echo "== Installing harness wheels =="
uv pip install --no-deps --force-reinstall /tmp/gemma4-wheelhouse/*.whl || true

if ! python -c "import swegemma, adk_submission" >/dev/null 2>&1; then
  echo "Bare install did not import; retrying with dependency resolution"
  uv pip install --force-reinstall /tmp/gemma4-wheelhouse/*.whl
fi

echo "== Import check =="
python - <<'PY'
import sys
print("python:", sys.version)
import swegemma, adk_submission
print("swegemma", getattr(swegemma, "__version__", "unknown"))
print("adk_submission", getattr(adk_submission, "__version__", "unknown"))
PY

echo "== Recording versions =="
{
  date -u
  python --version
  uv pip freeze
  nvidia-smi || true
} > "$WORK/env/harness-versions.txt"

echo "Harness ready in $VENV"
