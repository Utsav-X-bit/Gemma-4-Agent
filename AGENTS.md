# Gemma-4 — Project Rules

## Remote GPU server (4×A100)

The GPU server is restricted. Follow these rules on every remote task:

- **Never run compute or GPU work on the login node.** The login node is only for SSH, file transfer, package downloads, and job submission/management.
- **All compute and GPU work must run on a compute node** through Slurm (`sbatch`, `srun`, `salloc`) or the documented compute-node path.
- **Compute nodes have no internet access.** Do all downloads, `pip install`, `git clone`, Hugging Face/Kaggle downloads, and API calls on the login node first, then expose them to the compute node via shared storage.
- **No `sudo` on the remote server.** Do not attempt privileged commands; use user-level installs (`uv`, venvs, `$HOME`, project storage).
- **Login-node network work is allowed; login-node compute is not.** Do not start vLLM, training, evaluation, or long CPU jobs on the login node.
- Prefer `uv`/virtualenvs in user storage; never install into system Python.

If a command needs a GPU or more than trivial CPU, it belongs in a compute-node job, not an interactive login shell.

### Connection and paths

- VPN config: `~/isea13` (contains credentials; never print or commit it).
- Login alias: `ssh isea31` → `isea31@172.16.74.11` (login node).
- Slurm binaries: `/opt/slurm-20.11.7/bin/` (`sinfo`, `squeue`, `scontrol`, `sbatch`, `srun`).
- A100 partition: `airawatp`; GPU request: `--gres=gpu:A100-SXM4:4` (or `:1` for smoke tests).
- Shared home: `/nlsasfs/home/isea/isea31`.
- Project code: `~/Gemma-4-Agent`; data/models/work: `~/gemma4`.
- Competition data: `~/gemma4/data` (tasks, graphs, embeddings, sandbox, wheels).
- Harness wheelhouse: `~/gemma4/wheelhouse` (42 wheels; designed for the Kaggle image).
- Harness venv: `~/gemma4/harness-venv` (currently incomplete; see below).
- Compute-node local scratch: `/raid/user-$(id -u)`; stage large models/data there from shared storage.
- Existing long-running Qwen inference job `541396` uses 4×A100 on `scn2-10g` until **2026-10-28 06:24**. Do not kill it without explicit permission; submit a separate job.

### Persistent compute shell (use this)

A 7-day interactive allocation is held in tmux session **`gemma4`** on the login node:

```bash
tmux attach -t gemma4
```

It runs:

```bash
srun --job-name=gemma4-shell --partition=airawatp --nodelist=scn19-10g \
  --gres=gpu:A100-SXM4:4 --cpus-per-task=4 --time=7-00:00:00 --pty bash -l
```

- Node: `scn19-mn` (Slurm node `scn19-10g`), 4×A100-SXM4-40GB.
- Use `tmux send-keys -t gemma4 '<command>' Enter` for non-interactive commands, then `tmux capture-pane -pt gemma4` to read output.
- Do not run GPU commands in the login shell; run them inside `gemma4`.
- If the session dies, recreate it with the same `srun` command; `--cpus-per-task=4` is what scheduled immediately (32 CPUs stayed pending).

### Verified remote state (2026-10-27)

- Login node: internet works; HF token present (`Utsav-X-Bit`), Kaggle credentials present.
- Compute nodes: A100-SXM4-40GB, driver `550.90.07`; `srun` works.
- The HF repo `google/gemma-4-31B-it-qat-w4a16-ct` is accessible; model downloaded to `~/gemma4/models/gemma-4-31B-it-qat-w4a16-ct` (23 GB `model.safetensors`).
- `~/Gemma-4-Agent/.venv` is the project venv (uv, Python 3.12). It has `swegemma`, `adk_submission`, `adk_eval_core`, `google-adk`, `litellm==1.105.0`, and the harness deps. Use it for the agent/harness.
  - `litellm` must stay at **1.105.0**: 1.106.4 imports `mcp.types.CacheableResult`, which mcp 1.30.0 does not provide, breaking every model call.
  - Register the served alias as `aliases={declared_model: declared_model}`; the default mapping sends the model filesystem path, which vLLM 404s.
- `~/gemma4/vllm-venv` is the separate serving venv: `vllm 0.19.1`, `torch 2.10.0+cu128`, `transformers 5.13.1` (pinned to the wheelhouse), `tokenizers 0.22.2`. Torch CUDA works on the A100s with driver 550.90.07.
- `flashinfer-python`/`flashinfer-cubin` were removed: their `nvidia_cutlass_dsl` import aborts under this stack. vLLM falls back to NCCL.
- `~/gemma4/harness-venv` is incomplete (no torch/vllm); do not use it.
- The staged model at `/dev/shm/gemma4-model` has `allow_global_per_layer_attribute_access: true` added to `config.json`; without it, transformers 5.13 raises `AmbiguousGlobalPerLayerAttributeError` on Gemma 4 heterogeneous `head_dim`. Reapply after any restage.
- Another user (`isea52`, PID `2242364`, ollama/llama-server) is using GPUs 0–2 on `scn19-10g`; do not kill it. vLLM therefore runs with `--gpu-memory-utilization 0.40`, not 0.90.
- vLLM startup takes ~4 min (torch.compile + CUDA graph capture); generation is ~72 tok/s; KV cache is 25,056 tokens. The server listens on `127.0.0.1:8000`.
- Remote parity smoke passed: `rich_4077` resolved with the exact 395-char patch (25 calls, 122.6 s); `fastapi_15661` no-patch (28 calls, 398.2 s).
- Kaggle GPU quota is exhausted; remote is now the primary inference/evaluation/training target.

### Remote serving/eval commands

```bash
# inside tmux session gemma4
bash ~/Gemma-4-Agent/remote_vllm_gemma4.sh
tail -f ~/gemma4/vllm-gemma4.log

# harness venv (separate shell)
cd ~/Gemma-4-Agent
source .venv/bin/activate
python remote_eval.py --agent-dir ~/Gemma-4-Agent/submission-probe --tag remote-probe --limit 2
```

`remote_eval.py` registers the already-running vLLM endpoint; it does not start vLLM. Do not run `VllmServer.start()` in the harness venv (it has no torch/vLLM).

## Git sync (local ↔ remote)

- Local repo: `/home/utsav/Github/Gemma-4`.
- Remote clone: `/nlsasfs/home/isea/isea31/Gemma-4-Agent`.
- Origin: `git@github.com:Utsav-X-bit/Gemma-4-Agent.git` (remote uses HTTPS; either works on the login node).
- Sync direction:
  1. Commit and push from the local workstation.
  2. On the **login node** (internet), `cd ~/Gemma-4-Agent && git pull --ff-only`.
  3. Compute nodes read the same shared files; never run `git pull` on a compute node.
- Do **not** commit `.kaggle/`, `.venv/`, `__pycache__/`, model weights, `/dev/shm` staging, logs, or `remote-results/`.
- `.kaggle/kaggle.json` was removed from the index; if it was ever pushed, rotate the Kaggle token. The local file stays on disk for CLI use.

## Competition state

- **Phase 0 hidden score: 0.18** (probe bundle: 31B, 8 min, 28 calls, 4096 thinking, 8192 output).
- Local 13-task screen: probe 28 calls = 2/13 resolved, 12/13 non-empty patches; calls32 = 3/13 then 1/13 (unstable); calls40 = 1/13; prompt v2 = 2/13.
- Known-good config: **8 min / 28 calls / 4096 thinking / 8192 output / 240 s command timeout**.
- Prompt is the public 0.22 prompt plus one parallel-implementations line; prompt tuning is effectively exhausted.
- The 120-task projection at 8/28 is ~10.8 h, close to the 12 h global cap.
- Keep `submission-probe.zip` as the scored fallback.
- Do not claim a local improvement is real from a 13-task screen; require 20–30 tasks or paired seeds.

## Remote workflow

1. Download/install on the **login node** (internet).
2. Put models/data/venvs on shared storage (`~/gemma4`, `/nlsasfs/...`).
3. Submit a Slurm job; stage models from shared storage to `/raid/user-$(id -u)` on the compute node.
4. Run vLLM, agent evaluation, or training only inside the Slurm allocation.
5. Copy results back to shared storage and then to the local repo; compute-node local disk is ephemeral.
