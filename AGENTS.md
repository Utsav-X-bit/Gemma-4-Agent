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

### Training shell (1×A100)

Training runs in a separate tmux session **`train`** on `scn6-mn`:

```bash
tmux attach -t train
```

Allocation:

```bash
srun --job-name=gemma4-train --partition=airawatp --nodelist=scn6-10g \
  --gres=gpu:A100-SXM4:1 --cpus-per-task=4 --time=7-00:00:00 --pty bash -l
```

- Training venv: `~/gemma4/train-venv` (`torch 2.10.0`, `transformers 5.13.1`, `peft`, `trl`, `bitsandbytes`).
- Trainable base: `~/gemma4/models/gemma-4-31B-qat-unquantized` (59 GB).
- Data: `~/gemma4/sft/gold_sft.jsonl`; split `train_ids.json` / `holdout_ids.json` (116/13).
- Adapter output: `~/gemma4/adapters/gold-sft-r16`.
- The `scn19` allocation shares GPUs with another user's Flux training; do not train there. Use `train` for training, `gemma4` for serving/eval.

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

### Verified remote state (2026-10-29)

- Login node: internet works; Kaggle CLI and credentials present. Compute nodes have no internet.
- `scn6-mn` has 4×A100-SXM4-40GB; tmux `train4` holds 4 idle GPUs for training/serving.
- Exact scorer model: `~/gemma4/models/gemma-4-31B-it-qat-w4a16-ct` (22 GB), staged at `/dev/shm/gemma4-w4a16`.
- Scorer vLLM wheel: `~/gemma4/wheelhouse/vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl`, sha256 `73c69bafca24fbdad44690c44d6c6446514351e6f3cc9a6057cba05e66051310`. It is byte-identical to the Kaggle wheelhouse dataset.
- `~/gemma4/vllm-venv`: vLLM 0.19.1, torch 2.10.0+cu128, transformers 5.13.1. The custom patch resolves the W4A16 model as `Gemma4ForCausalLM`.
- Adapter format (verified): train separate `q/k/v/o/gate/up/down` projections, then run `phase5/remap_adapter_prefix.py` to rewrite `model.language_model.layers.*` -> `model.self_decoder.decoder_layers.*`. Serve with `--max-lora-rank 16`. Do **not** fuse qkv/gate_up; vLLM rejects fused target names.
- Active adapter: `~/gemma4/adapters/native-clean-prefixed` (234 MB). Activation is proven by logprobs, not by load logs: base `-1.788e-06` vs adapter `-2.861e-06` for the first token.
- Harness venv: `~/Gemma-4-Agent/.venv` (Python 3.12) with `swegemma`, `adk_submission`, `adk_eval_core`, `google-adk`, `litellm==1.105.0`.
  - `litellm` must stay at **1.105.0**; 1.106.4 breaks model calls via `mcp.types.CacheableResult`.
  - Register the served alias as `aliases={declared_model: declared_model}`.
- Local Docker evaluator: `.venv-local` (Python 3.13), image `swebench-sandbox:latest`, local `data/` with 13 snapshots. Gold audit: 8/13; the five clean tasks are `fastapi_15588`, `fastapi_15589`, `requests_7427`, `rich_4077`, `rich_4076`.
- vLLM startup ~3–4 min; adapter KV cache ~15k tokens at 32k max len. Servers currently run on ports 8007/8008 on `scn6-mn`.
- Kaggle GPU quota is exhausted; remote is the primary inference/evaluation/training target.

### Remote serving/eval commands

```bash
# prefixed adapter server (inside tmux train4, GPU 0)
cd ~/Gemma-4-Agent
CUDA_VISIBLE_DEVICES=0 MAX_LORA_RANK=16 ADAPTER_NAME=native-clean-prefixed \
  ADAPTER_DIR=$HOME/gemma4/adapters/native-clean-prefixed PORT=8007 \
  LOG=$HOME/gemma4/vllm-prefixed-8007.log bash phase5/remote_vllm_gold.sh

# local tunnel
ssh -N -L 8007:scn6-mn:8007 isea31

# local Docker evaluator (host)
.venv-local/bin/python phase5/local_eval.py \
  --agent-dir phase5/submission-v4-32-prefixed --tag v4-32-prefixed \
  --task-id fastapi_15588 --task-id fastapi_15589 --task-id requests_7427 \
  --task-id rich_4077 --task-id rich_4076 \
  --per-repo 4 --host 127.0.0.1 --port 8007

# remote subprocess evaluator (smoke only; 8/13 gold tasks are invalid)
source .venv/bin/activate
python phase5/remote_eval.py --agent-dir phase5/submission-v4-32-prefixed --tag smoke --limit 1
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
- Canonical remote scripts are `phase5/remote_eval.py` and `phase5/remote_vllm_gold.sh`; the local Docker evaluator is `phase5/local_eval.py` + `phase5/local_verify.py`. Legacy copies at `~/Gemma-4-Agent/remote_eval.py` and `remote_vllm_gemma4.sh` are stale; do not use them.
- `.kaggle/kaggle.json` was removed from the index; if it was ever pushed, rotate the Kaggle token. The local file stays on disk for CLI use.

## Training split

- Use all 129 public tasks as the data pool, but never train and evaluate on the same task.
- First adapter: **116 train / 13 holdout** (the current local screen).
- Confirmation: **103 train / 26 holdout**.
- Final adapter: **129 train**; the old holdout is invalid after retraining, so use the hidden score.
- Split by `instance_id`; `patch` is the target, `test_patch` is verification only.
- Current active adapter: `~/gemma4/adapters/native-clean-prefixed`, trained on `~/gemma4/sft/native_clean_sft.jsonl` (778 rows, rank 16, alpha 8), then remapped with `phase5/remap_adapter_prefix.py`.
- Gold-patch SFT (`gold_sft.jsonl`) is not the current adapter and is a known wash; use successful agent trajectories for future training.

## Competition state (2026-10-29)

- Latest public notebook score: **0.17** (not our adapter; no adapter submission has been scored yet).
- Current best local bundle: `phase5/submission-v4-32-prefixed` — 5/5 reported, **4/5 certain** on the five clean Docker tasks; all five have non-empty patches.
- Adapter: `native-clean-prefixed`, rank 16, alpha 8, 234 MB; activation confirmed by logprob A/B.
- Keep the original tool list (`get_status` + graph tools). Removing them regressed `requests_7427` (4/5 reported, 3/5 real).
- `v6-32-prefixed`: 4/5 reported but two `patch_chars=0` false positives; `requests_7427` fails.
- `v7-32-finisher-prefixed`: 5/5 reported but two `patch_chars=0` false positives.
- Config: **32 calls / 8 min / 4096 thinking / 8192 output / 240 s command timeout**; projected ~10.7 h for 120 tasks.
- Do not raise to 48 calls; the global cap is 12 h.
- The old 13-task subprocess screen is invalid for 8/13 (gold fails); use the five clean Docker tasks.
- Keep `submission-probe.zip` only as a historical fallback; the current candidate is the prefixed adapter bundle.
- Do not claim a local improvement is real from a single 5-task screen; the hidden score is the only final signal.

## Remote workflow

1. Download/install on the **login node** (internet).
2. Put models/data/venvs on shared storage (`~/gemma4`, `/nlsasfs/...`).
3. Submit a Slurm job; stage models from shared storage to `/raid/user-$(id -u)` on the compute node.
4. Run vLLM, agent evaluation, or training only inside the Slurm allocation.
5. Copy results back to shared storage and then to the local repo; compute-node local disk is ephemeral.
