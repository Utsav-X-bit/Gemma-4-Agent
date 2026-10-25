#!/usr/bin/env bash
set -euo pipefail

WORK="${WORK:-$HOME/gemma4}"
DATA="${DATA:-$WORK/data}"
WHEELS="${WHEELS:-$WORK/wheelhouse}"
export WORK DATA WHEELS
mkdir -p "$DATA" "$WHEELS"

echo "== Downloading competition data =="
kaggle competitions download -c gemma-4-developer-agent -p "$DATA" --unzip

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
if ! kaggle datasets download -d metric/gemma-4-developer-agent-wheelhouse -p "$WHEELS" --unzip; then
  echo
  echo "Wheelhouse download failed."
  echo "Open the competition Data tab or the starter notebook, download the wheelhouse dataset,"
  echo "and place the .whl files in $WHEELS. Then re-run 02_setup_harness.sh."
  exit 1
fi

echo "== Wheel count =="
ls "$WHEELS"/*.whl | wc -l
echo "Expected: a wheelhouse with the harness dependencies (the starter notebook staged 41 wheels)."
