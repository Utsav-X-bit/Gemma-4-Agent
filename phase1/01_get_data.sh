#!/usr/bin/env bash
set -euo pipefail

WORK="${WORK:-$HOME/gemma4}"
DATA="${DATA:-$WORK/data}"
WHEELS="${WHEELS:-$WORK/wheelhouse}"
export WORK DATA WHEELS
mkdir -p "$DATA" "$WHEELS"

# New CLI: KAGGLE_API_TOKEN or ~/.kaggle/access_token. Old CLI: ~/.kaggle/kaggle.json.
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ -f "$HOME/.kaggle/access_token" ]; then
  export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token")"
fi

echo "== Checking Kaggle auth =="
if ! kaggle competitions list -s gemma-4 >/dev/null 2>&1; then
  echo
  echo "Kaggle auth failed."
  echo "The installed CLI may be the old one that only reads ~/.kaggle/kaggle.json."
  echo "Fastest fix: Kaggle Settings -> API -> Legacy API Credentials -> Create Legacy API Key,"
  echo "copy the downloaded kaggle.json to ~/.kaggle/kaggle.json, chmod 600, and rerun this script."
  echo "Alternative: install the current CLI in a Python 3.11+ venv and use KAGGLE_API_TOKEN."
  exit 1
fi

echo "== Downloading competition data =="
# The old CLI does not support --unzip; download, then extract any zips with Python.
kaggle competitions download -c gemma-4-developer-agent -p "$DATA"

python3 - "$DATA" <<'PY'
import sys, zipfile
from pathlib import Path
d = Path(sys.argv[1])
for z in sorted(d.glob("*.zip")):
    print("extracting", z.name)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(d)
PY

echo "== Checking task count =="
python3 - <<'PY'
import json, os
p = os.path.join(os.environ.get("DATA", os.path.expanduser("~/gemma4/data")), "tasks.jsonl")
rows = [json.loads(line) for line in open(p) if line.strip()]
print("tasks:", len(rows))
print("repos:", sorted({r["repo"] for r in rows}))
assert len(rows) == 129, f"expected 129 tasks, got {len(rows)}"
PY

echo "== Downloading harness wheelhouse =="
# The slug is inferred from the starter notebook's input path; if this 403s,
# discover the current slug with: kaggle datasets list -s gemma-4
if ! kaggle datasets download -d metric/gemma-4-developer-agent-wheelhouse -p "$WHEELS"; then
  echo
  echo "Wheelhouse download failed."
  echo "Open the competition Data tab or the starter notebook, download the wheelhouse dataset,"
  echo "and place the .whl files in $WHEELS. Then re-run 02_setup_harness.sh."
  exit 1
fi

python3 - "$WHEELS" <<'PY'
import sys, zipfile
from pathlib import Path
d = Path(sys.argv[1])
for z in sorted(d.glob("*.zip")):
    print("extracting", z.name)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(d)
# Flatten nested wheel directories so $WHEELS/*.whl works.
for w in list(d.rglob("*.whl")):
    target = d / w.name
    if w != target and not target.exists():
        target.write_bytes(w.read_bytes())
PY

echo "== Wheel count =="
find "$WHEELS" -maxdepth 1 -name '*.whl' | wc -l
echo "Expected: a wheelhouse with the harness dependencies (the starter notebook staged 41 wheels)."
