#!/usr/bin/env bash
set -euo pipefail

WORK="${WORK:-$HOME/gemma4}"
WHEELS="${WHEELS:-$WORK/wheelhouse}"
VENV="${VENV:-$WORK/harness-venv}"
export WORK WHEELS VENV

if ! ls "$WHEELS"/*.whl >/dev/null 2>&1; then
  echo "No wheels found in $WHEELS. Run 01_get_data.sh or point WHEELS at the wheelhouse."
  exit 1
fi

echo "== Creating harness venv =="
python3 -m venv --system-site-packages "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

echo "== Removing broken cutlass .pth hooks from the venv =="
find "$VIRTUAL_ENV/lib" -maxdepth 3 -name '*cutlass*.pth' -delete 2>/dev/null || true

echo "== Staging wheels (skip cutlass, restore +cu128) =="
rm -rf /tmp/gemma4-wheelhouse
mkdir -p /tmp/gemma4-wheelhouse
for w in "$WHEELS"/*.whl; do
  case "$(basename "$w")" in
    *cutlass*) continue ;;
  esac
  name="$(basename "$w")"
  case "$name" in
    *cu128*+*) ;;
    *cu128*) name="${name/cu128/+cu128}" ;;
  esac
  ln -sf "$w" "/tmp/gemma4-wheelhouse/$name" 2>/dev/null || cp "$w" "/tmp/gemma4-wheelhouse/$name"
done

echo "== Installing harness =="
pip install -U pip
# The notebook uses --no-deps on top of the Kaggle image. We use
# --system-site-packages; if imports still fail, fall back to dependency resolution.
if ! pip install --no-deps --force-reinstall /tmp/gemma4-wheelhouse/*.whl; then
  echo "Bare --no-deps install failed; retrying with dependency resolution"
  pip install --force-reinstall /tmp/gemma4-wheelhouse/*.whl
fi

echo "== Import check =="
python - <<'PY'
import swegemma, adk_submission
print("swegemma", getattr(swegemma, "__version__", "unknown"))
print("adk_submission", getattr(adk_submission, "__version__", "unknown"))
PY

echo "== Recording versions =="
{
  date -u
  python --version
  pip freeze
  nvidia-smi
} > "$WORK/env/harness-versions.txt"

echo "Harness ready in $VENV"
