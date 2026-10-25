# Phase 1 — Remote Preflight + One-Task Smoke Test

**Goal:** on the 4×A100 40 GB box, install the harness, select the sandbox backend, and run **one public task end-to-end** with the primary 5-min/24-call bundle. This proves the remote stack before calibration or training.

**Phase gate:** one task produces a non-empty patch or a clearly explained harness/test failure, and `phase1-versions.txt` exists. Do not start calibration or training until this passes.

---

## 0. What you need

- SSH access to the remote 4×A100 box.
- Kaggle API token at `~/.kaggle/kaggle.json` on the remote box (mode 600).
- **No Docker/sudo required.** The harness supports `--sandbox subprocess`; Phase 1 uses it when Docker is unavailable.
- At least ~250 GB free disk (competition data + models + results).
- The `submission-primary/` directory from this repo (the 5-min/24-call bundle).

Copy the phase files to the remote box:

```bash
scp -r phase1 submission-primary user@HOST:~/gemma4/
```

Then SSH in:

```bash
ssh user@HOST
cd ~/gemma4
```

All scripts use `WORK=$HOME/gemma4` by default. Override `WORK`, `DATA`, `RUNS`, `AGENT`, `VENV` if your paths differ.

---

## 1. Preflight

```bash
bash phase1/00_preflight.sh
```

Expected:

- 4× A100 40 GB visible.
- ≥64 GB RAM preferred.
- ≥250 GB free disk.
- `nvidia-smi`, Python, git present.
- `venv` available; `ensurepip` ideally available. If `ensurepip` is missing, the subprocess sandbox may create a venv without pip; record it and see whether the task still runs.
- Docker is optional; the sandbox check will choose Docker if usable, otherwise subprocess.

If `nvidia-smi` fails, stop and fix the driver/container. Do not continue.

---

## 2. Download competition data + wheelhouse

```bash
bash phase1/01_get_data.sh
```

Expected:

- `129 tasks` and four repos.
- Wheelhouse `.whl` files under `~/gemma4/wheelhouse`.

If the wheelhouse dataset slug 403s, discover the current slug:

```bash
kaggle datasets list -s gemma-4
```

or download it from the competition Data tab / starter notebook and place the wheels in `~/gemma4/wheelhouse`.

---

## 3. Install the harness

```bash
bash phase1/02_setup_harness.sh
```

Expected:

- `swegemma` and `adk_submission` import.
- `~/gemma4/env/harness-versions.txt` exists.

The script creates a venv with `--system-site-packages`, removes broken cutlass `.pth` hooks, stages wheels with the `+cu128` fix, and falls back to dependency resolution if the bare `--no-deps` install does not import.

If import fails, read the error before changing anything:

```bash
source ~/gemma4/harness-venv/bin/activate
python -c "import swegemma, adk_submission"
pip list | grep -E 'swegemma|adk|vllm|torch|transformers'
```

---

## 4. Check the sandbox backend

```bash
bash phase1/03_sandbox_check.sh
```

Expected: `~/gemma4/env/sandbox-backend.txt` says either `backend: docker` or `backend: subprocess`.

**No Docker/no sudo:** use the subprocess backend. The harness creates an isolated temp workspace/venv, rewrites `/workspace` paths, and kills process groups on timeout, but it is **not scorer-identical**:

- Python version may differ (the scorer's Docker image is Python 3.13; the remote venv may be 3.12).
- Isolation is weaker than Docker.
- Wheel/test behavior can differ.

Use subprocess for the Phase 1 smoke test. For Phase 2 gold-patch calibration, use a clean venv or a Kaggle CPU notebook, not the subprocess backend alone. Do not install Docker or ask for sudo; it is not required.

---

## 5. One-task smoke test

The default task is `fastapi_15661`. The script auto-selects Docker if usable, otherwise subprocess.

```bash
bash phase1/04_smoke_one_task.sh
```

To force the backend:

```bash
SANDBOX=subprocess bash phase1/04_smoke_one_task.sh
# or
SANDBOX=docker bash phase1/04_smoke_one_task.sh
```

Expected artifacts:

```text
~/gemma4/runs/smoke-fastapi_15661/
├── summary.json
├── patches/fastapi_15661.patch
├── test_outputs/fastapi_15661.log
├── traces/trace_fastapi_15661.json
└── logs/fastapi_15661.log
```

Pass condition:

- a non-empty `patches/fastapi_15661.patch`, **or**
- a clear failure you can classify: no model server, Docker error, missing wheel, collection error, timeout, or agent no-patch.

If the CLI cannot start/find the model server, provide a models YAML:

```bash
MODELS_YAML=/path/to/models.yaml bash phase1/04_smoke_one_task.sh
```

or use the starter notebook’s `VllmServer` path. The goal is one task, not a perfect stack.

Inspect:

```bash
cat ~/gemma4/runs/smoke-fastapi_15661/summary.json
ls -l ~/gemma4/runs/smoke-fastapi_15661/patches
head -80 ~/gemma4/runs/smoke-fastapi_15661/traces/trace_fastapi_15661.json
```

What you are looking for:

- the agent made at least one tool call;
- it edited a source file before the budget ended;
- the patch is non-empty;
- Phase 2 either passed or failed for a reason you can explain.

---

## 6. Record versions

```bash
bash phase1/05_record_versions.sh
```

Expected: `~/gemma4/env/phase1-versions.txt` with GPU, Python, harness, vLLM, Docker image, wheelhouse hash, and submission zip hash.

---

## 7. Stop and report

Send back:

1. `phase1/00_preflight.sh` output (or the key lines).
2. `summary.json`.
3. `phase1-versions.txt`.
4. The first 20 lines of the trace if there is no patch.
5. Any error text.

Do **not** start calibration, trajectory generation, or training yet. Phase 2 will use this stack to:

- validate gold patches on 10–15 tasks;
- create a 20–40 task holdout;
- compare the 5-min/24-call primary bundle against the 8-min/28-call reference;
- measure `agent_p50`, `setup_p50`, patch rate, and first-edit call.

Only after Phase 2 passes do we touch LoRA/training.
