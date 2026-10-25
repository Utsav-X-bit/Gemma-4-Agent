#!/usr/bin/env bash
set -euo pipefail

WORK="${WORK:-$HOME/gemma4}"
mkdir -p "$WORK/env"

{
  echo "== UTC =="
  date -u
  echo
  echo "== GPU =="
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; this is not the A100 box or the driver is missing"
    exit 1
  fi
  nvidia-smi
  echo
  echo "== MEMORY =="
  free -h
  echo
  echo "== DISK =="
  df -h "$WORK"
  echo
  echo "== PYTHON =="
  python3 --version
  pip --version || true
  echo
  echo "== DOCKER =="
  docker --version || echo "docker not available"
  echo
  echo "== VENV / ENSUREPIP =="
  python3 -m venv --help >/dev/null 2>&1 && echo "venv: ok" || echo "venv: missing"
  python3 -m ensurepip --version 2>/dev/null || echo "ensurepip: missing (subprocess sandbox may create a venv without pip)" 
  echo
  echo "== GIT =="
  git --version
} | tee "$WORK/env/preflight.txt"

echo
echo "Preflight written to $WORK/env/preflight.txt"
echo "Expected: 4x A100 40GB visible; >=64GB RAM preferred; >=250GB free disk."
echo "Docker is optional; if unavailable, Phase 1 uses --sandbox subprocess (not scorer-identical)."
