#!/usr/bin/env bash
set -euo pipefail

WORK="${WORK:-$HOME/gemma4}"
mkdir -p "$WORK/env"

echo "== Docker availability =="
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "docker: usable"
  echo "backend: docker"
  docker --version
else
  echo "docker: not usable (missing, no daemon, or no permission)"
  echo "backend: subprocess"
  echo
  echo "No sudo/Docker is fine. The harness supports --sandbox subprocess."
  echo "It creates an isolated temp workspace/venv and rewrites /workspace paths,"
  echo "but it is NOT scorer-identical: Python version, isolation, and wheel/test"
  echo "behavior may differ. Use it for the Phase 1 smoke test only."
fi | tee "$WORK/env/sandbox-backend.txt"

echo
echo "Recorded $WORK/env/sandbox-backend.txt"
