#!/bin/bash
# Start Gemma 4 31B W4A16 + gold LoRA on the single-A100 training node.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/dev/shm/gemma4-w4a16}"
ADAPTER_DIR="${ADAPTER_DIR:-$HOME/Gemma-4-Agent/phase5/submission-gold/adapters/gold}"
LOG="${LOG:-$HOME/gemma4/vllm-gold.log}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-gemma-4-31b-it-qat-w4a16-ct}"
VLLM_PY="${VLLM_PY:-$HOME/gemma4/vllm-venv/bin/python}"

export PYTHONUNBUFFERED=1
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export LITELLM_TELEMETRY=False

test -f "$MODEL_DIR/config.json" || { echo "missing model at $MODEL_DIR"; exit 1; }
test -f "$ADAPTER_DIR/adapter_config.json" || { echo "missing adapter at $ADAPTER_DIR"; exit 1; }

echo "starting vLLM: model=$MODEL_DIR adapter=$ADAPTER_DIR port=$PORT log=$LOG"
nohup "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --dtype auto \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --reasoning-config '{"reasoning_start_str":"<|channel>","reasoning_end_str":"<channel|>"}' \
  --no-scheduler-reserve-full-isl \
  --disable-custom-all-reduce \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank 16 \
  --lora-modules "gold=$ADAPTER_DIR" \
  --trust-remote-code \
  > "$LOG" 2>&1 &
echo "PID=$!"
