# 08 — Run Log and Deviations from the Main Runbook

This is the chronological record of what we actually did, what changed, and why. It is not a polished guide; it is the audit trail for future reference. The main plan is `README.md`; deviations are recorded here.

Last updated: **2026-10-27**.

---

## 1. Status summary

| Item | Status |
|---|---|
| Phase 0 — hidden submission probe | **Scored 0.18** (ref **57642443**) |
| Phase 5 — exact public 0.24 prompt | Submitted **57709729**, **PENDING** |
| Phase 1 — harness smoke test | Passed on Kaggle L4x4; one non-empty patch |
| Phase 2a — gold-patch calibration | **13/13** selected public tasks resolved |
| Phase 2b — budget comparison | Complete; **8 min / 28 calls** is best of three |
| Phase 3 — prompt v2 | Run; **2/13**, no improvement over probe |
| Remote 4×A100 harness | **Operational**; vLLM 0.19.1 + harness venv; smoke tasks pass |
| Docker | Not used; no sudo on remote; Kaggle uses subprocess sandbox |

Current best configuration: **31B, 8 min, 28 calls, 240 s command timeout, 8192 output, 4096 thinking, subprocess sandbox**. It is not yet final because the projected 120-task time is ~10.8 h, too close to the 12 h limit.

---

## 2. Phase 0 — submission probe

- Built `submission-probe.zip` from the public 0.22-style prompt plus 0.24 edit-recovery rules.
- Config: 31B, 8 min, 28 calls, 240 s, 8192 output, 4096 thinking.
- Submitted on 2026-10-25 20:46:54 UTC.
- Ref: **57642443**.
- **Scored 0.18 public/private** after the queue cleared on 2026-10-27.
- Purpose: validate that Kaggle accepts and scores the bundle. It does not test the agent quality.

Artifact: `submission-probe.zip`.

### 2.1 Exact public 0.24 prompt submission

- Bundle: `phase5/submission-024`, prompt SHA-256 `37dbbed1a80113851abae5166b48b9205e6929c05d46682d06c64ad607190345`.
- Source: `honghanhhh/gemma-4-baseline-lb-0-24` (public notebook; author-reported 0.24).
- Submitted 2026-10-27 18:06 UTC, ref **57709729**.
- Status: **PENDING**.
- The daily submission quota is now exhausted; the next slot is tomorrow.
- Purpose: test whether the four missing prompt rules close the 0.18 → 0.24 gap.

---

## 3. Phase 1 — harness smoke test

### 3.1 What was attempted on the remote 4×A100 box

- Kaggle CLI auth initially failed because the old CLI only reads `~/.kaggle/kaggle.json`; the user had a newer `~/.kaggle/access_token`.
- Fixed by using a legacy `kaggle.json` (or a newer CLI).
- Competition data downloaded: **129 tasks**, 20.4 GB zip; wheelhouse 827 MB, 41 wheels.
- Remote Python was **3.10**; the wheelhouse contains **cp312** wheels, so install failed.
- `uv` provisioned **Python 3.12.14**; 40 wheels installed.
- Import still failed because the wheelhouse is built for the Kaggle image and is not self-contained. Resolver conflicts:
  - `prometheus-fastapi-instrumentator 8.1.0` requires `starlette>=1.0`, but `google-adk 1.36.1` requires `starlette<1.0`.
  - `xgrammar 0.2.5.post1` requires `transformers<5`, but the wheelhouse pins `transformers 5.13.1` and `vllm 0.19.1`.
- Pinning `prometheus-fastapi-instrumentator==7.1.0` fixed one conflict, but the xgrammar/transformers conflict remains metadata-level. The Kaggle image uses `--no-deps` and already has the base packages.

**Decision:** stop the remote harness install. The wheelhouse is designed for the Kaggle image; chasing its dependency graph on a bare HPC venv is not worth the time. Use Kaggle for harness/evaluation, remote for training.

### 3.2 Kaggle GPU quota exhausted — remote becomes primary

- Kaggle GPU quota is exhausted as of 2026-10-27. All further inference/evaluation must run on the remote 4×A100.
- Remote harness is now operational:
  - login `isea31` → Slurm partition `airawatp`; persistent 7-day tmux session `gemma4` on `scn19-mn` (4×A100-SXM4-40GB).
  - `~/Gemma-4-Agent/.venv`: `swegemma`, `adk_submission`, `adk_eval_core`, `google-adk`, `litellm==1.105.0`.
  - `~/gemma4/vllm-venv`: `vllm 0.19.1`, `torch 2.10.0+cu128`, `transformers 5.13.1`, `tokenizers 0.22.2`; flashinfer removed.
  - Model: `google/gemma-4-31B-it-qat-w4a16-ct`, 23 GB safetensors, staged to `/dev/shm/gemma4-model`.
  - Workarounds: `allow_global_per_layer_attribute_access: true` in staged `config.json`; `--disable-custom-all-reduce`; `--gpu-memory-utilization 0.40` because another user's process occupies GPUs 0–2; `--gpu-memory-utilization` 0.90/0.45 OOMs.
  - vLLM startup ~4 min (torch.compile + CUDA graphs); generation ~72 tok/s; KV cache 25,056 tokens.
- Remote parity smoke:
  - `rich_4077`: **resolved**, exact 395-char patch, 25 calls, 122.6 s (Kaggle: resolved, 395 chars, 18 calls, 149.9 s).
  - `fastapi_15661`: no patch, 28 calls, 398.2 s (matches Kaggle's known no-patch).
- Full 13-task remote baseline `remote-probe13` is running.

### 3.3 Kaggle smoke test

- Official Getting Started notebook, L4x4, internet off.
- vLLM started; 2 tasks ran; sample submission 1 min/10 calls returned `NO_PATCH` on both. That is expected from the sample budget, not a harness failure.
- Later, with `submission-primary` (5 min/24 calls), one of two tasks produced a non-empty patch (`fastapi_15588`); `fastapi_15661` returned `NO_PATCH`.
- Versions:
  - Python **3.12.13**
  - vLLM **0.19.1**
  - transformers **5.13.1**
  - torch **2.10.0+cu128**
  - 4× L4, 23 GB per GPU
- A stale vLLM server caused a startup OOM (`1.56/22.03 GiB` free); fixed by killing the old workers or restarting the session.

**Phase 1 gate passed:** vLLM starts, tools execute, Phase 2 runs, and a non-empty patch is produced.

---

## 4. Phase 2a — gold-patch calibration

- Used `zbigniewkirchner/gemma-4-agent-do-gold-patches-pass-locally`.
- Selected 13 tasks: 4 fastapi, 4 rich, 4 requests, 1 httpx.
- Scorer-like sandbox (`clean_venv=True`).
- Result: **13/13 gold patches resolved**.
- The full public set is 127/129; the two known exceptions are internally inconsistent tests.

**Phase 2a gate passed.**

---

## 5. Phase 2b — budget comparison

Same 13 tasks, same 31B model, same prompt/tools, same vLLM server. Only budget/thinking changed.

| Arm | Time | Calls | Thinking | Resolved | Patches | Mean | p95 | Projected 120 tasks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | 5 min | 24 | 2048 | **0/13** | 7/13 | 290 s | 337 s | **9.7 h** |
| probe | 8 min | 28 | 4096 | **2/13** | 12/13 | 325 s | 475 s | **10.8 h** |
| calls40 | 8 min | 40 | 4096 | **1/13** | 10/13 | 381 s | 485 s | **12.7 h** |

### 5.1 Resolved tasks

- Probe resolved `fastapi_15589` and `rich_4077`.
- Calls40 resolved only `rich_4077`; it lost `fastapi_15589`.
- Primary resolved none.

### 5.2 Why calls40 was rejected

- More calls did not improve resolved count.
- It increased mean duration to 381 s and projected 120-task time to **12.7 h**.
- It produced more scratch files (`repro.py`, `reproduce_bug.py`, `reproduce_issue.py`) and extra edits.
- The 28-call cap acts as a finalization pressure; 40 calls encouraged more wandering.

**Decision:** do not increase to 40 calls. The best arm is **8 min / 28 calls / 4096 thinking**.

### 5.3 Failure taxonomy from patch file overlap

| Failure | Tasks | Evidence |
|---|---|---|
| Multi-file change missed | `fastapi_15030`, `rich_4070`, `httpx_3672` | Reference touches 9/9/7 files; agent touched 1/2/4 |
| Wrong file | `rich_4076`, `rich_4079` | Edited `rich/markdown.py`/`repro.py`; reference is `rich/ansi.py`/`rich/markdown.py` |
| Wrong new filename | `fastapi_15661` | Created `scripts/release.py`; reference is `scripts/prepare_release.py` |
| Scratch files in patch | several | `repro.py`, `reproduce_bug.py`, `reproduce_issue.py` |
| Tests fail despite correct files | `requests_7505`, `requests_7502`, `requests_7433`, `requests_7427`, `fastapi_15588` | File overlap is correct; patch is incomplete or edge cases missing |

Artifacts: `phase2/task-results-primary.csv`, `task-results-probe.csv`, `task-results-calls40.csv`, `task-results-all.csv`.

---

## 6. Phase 3 — prompt v2

Not yet run. Same 13 tasks, same 8 min / 28 calls / 4096 thinking as the probe. Only the prompt changes.

### 6.1 Prompt additions

Appended to the probe prompt:

```markdown
# MULTI-FILE AND PATCH HYGIENE
- Before editing, run one `git grep -n` for the exact symbol or behavior and list every file that implements, re-exports, or documents it. If the behavior is shared, edit every implementation: sync/async, Client/AsyncClient, backends, `docs_src` examples, `__all__`.
- If the repo has `docs_src/` and the issue mentions docs/examples, update every matching `docs_src` example, not just the library code.
- If the issue names an exact file or script, create or modify that exact path. Do not invent a new filename.
- Do not create any file under `/workspace` except the source edit. Put reproduction scripts in `/tmp` only. Any scratch file under `/workspace` becomes part of the graded patch.
- If pytest fails at collection or import, inspect the traceback and fix the source/import; never edit the test or runner config.
- Once the targeted test passes, stop editing. Do not add refactors or speculative changes. Submit.
```

### 6.2 Why each rule was added

| Rule | Failure it addresses |
|---|---|
| `git grep` inventory before editing | multi-file tasks (`fastapi_15030`, `rich_4070`, `httpx_3672`) |
| Update `docs_src` examples | `fastapi_15030` reference has 9 files including docs examples |
| Exact file/script name | `fastapi_15661` created the wrong script name |
| No scratch under `/workspace` | `repro.py`/`reproduce_bug.py` appeared in patches |
| Fix source on collection/import error | exit code 2 tasks; do not edit tests |
| Stop after targeted test passes | calls40 over-editing lost a resolve |

Notebook: `phase3/phase3-prompt-v2.ipynb` and `/home/utsav/Downloads/phase3-prompt-v2.ipynb`.

### 6.3 Phase 3 gate

Promote v2 only if:

- resolved improves by ≥2 tasks on the same 13, and
- patch count does not drop, and
- p95 duration does not rise materially.

If v2 improves, test **7 min / 28 calls** to reduce the 10.8 h projection. If v2 does not improve, inspect test failure logs instead of adding more budget.

### 6.4 Phase 4 result — calls32 is provisional

| Arm | Resolved | Patches | p50 | p95 |
|---|---:|---:|---:|---:|
| probe 8/28/4096 | 2/13 | 12/13 | 339.2 s | 474.6 s |
| calls32 8/32/4096 | **3/13** | 12/13 | 319.2 s | **493.7 s** |
| calls40 8/40/4096 | 1/13 | 10/13 | 373.1 s | 484.9 s |
| v2 8/28/4096 | 2/13 | 11/13 | 352.6 s | 464.8 s |

Calls32 resolved 3/13: rich 2/4, fastapi 1/4, requests 0/4, httpx 0/1. It produced 12/13 patches; `rich_4076` remained `NO_PATCH`. Mean tool calls 27.1, max 32. Mean duration 320.6 s; projected 120-task agent time 10.69 h.

It is a **clean +1** over probe: it keeps `fastapi_15589` and `rich_4077` and adds `rich_4079`; no resolved task was lost. Patches are 12/13 in both. But +1 is inside the noise band and p95 is 484.9 s.

**Decision:** provisional candidate. Keep probe as the known-good fallback. Run a second seed on the same 13 tasks before promoting. If the second run reproduces the +1, test 7 min / 32 calls to create global-time margin.

### 6.5 Public 0.24 prompt recovered

The public 0.24 notebook (`honghanhhh/gemma-4-baseline-lb-0-24`) was downloaded. Its bundle is the 0.22 bundle with four prompt changes:

- hard limits 28 calls / 8 minutes (same as probe);
- test-file ban also covers `pytest.ini` and `conftest.py`;
- scratch files must go in `/tmp`, never `/workspace`;
- final check is `git status --short && git diff`, and untracked scratch/test files must be removed before `submit_patch`.

Our probe prompt lacked those four changes. The exact bundle is now in `phase5/submission-024` (prompt SHA-256 `37dbbed1...`). Test it on the same 13 remote tasks after `remote-probe13` finishes.

**Decision:** exact 0.24 is the prompt-only baseline to match. A remote A/B must show ≥2 tasks gained with no losses before adoption.

### 6.6 Remote 13-task baseline (A100)

| Arm | Resolved | Patches | Mean | p95 |
|---|---:|---:|---:|---:|
| Kaggle L4 probe 28 | 2/13 | 12/13 | 324.6 s | 474.6 s |
| Remote A100 probe 28 | **3/13** | 11/13 | **162.8 s** | 227.5 s |

Remote gained `requests_7427`; no Kaggle-resolved task was lost. `rich_4079` and `rich_4076` produced no patch on remote. The A100 is ~2× faster than L4, so remote timing is **not** a valid global-time projection; use it for correctness/ranking only.

The exact 0.24 prompt is now running as `remote-024` on the same 13 tasks.

### 6.7 Exact 0.24 prompt remote A/B — rejected

| Arm | Resolved | Patches | Mean | p95 |
|---|---:|---:|---:|---:|
| remote probe 28 | **3/13** | 11/13 | 162.8 s | 227.5 s |
| remote exact 0.24 | 2/13 | 9/13 | 211.7 s | 381.9 s |

The exact 0.24 prompt lost `fastapi_15589`, gained nothing, and produced fewer patches. The public 0.24 claim does not reproduce on this 13-task subset. The hidden submission `57709729` remains the authoritative test, but the local screen does not support adopting the 0.24 prompt.

**Decision:** keep the probe prompt as the base. Next change is structural: a read-only explorer sub-agent for multi-file localization.

### 6.8 Explorer sub-agent experiment

The next structural arm is `phase5/submission-explorer`: the probe bundle plus a read-only `explorer_agent` sub-agent.

- Sub-agent tools: `run_command`, `read_file`, `get_status`.
- Its prompt requires exact files/lines, parallel implementations, root cause, edit plan, and a targeted test command, with at most 3 tool calls.
- Main prompt calls `explorer_agent` once after the first search and uses its FILES/PARALLEL lists as the edit checklist.
- Smoke task: `fastapi_15030`; the sub-agent was invoked at 9.2 s and read `fastapi/__init__.py`, confirming the `agent_tool` wiring works.

If the smoke task shows no regression, run the full 13-task A/B against `remote-probe13`.

### 6.9 Explorer sub-agent — rejected

| Task | Probe | Explorer |
|---|---|---|
| `fastapi_15030` | 0, 1127-char patch | 0, 1109-char patch |
| `rich_4079` | 0, NO_PATCH | 0, NO_PATCH |
| `httpx_3672` | 0, 3-file patch | 0, 1-file patch |

The explorer was invoked correctly (`explorer_agent` at 9.2 s), but it consumed main-agent budget and did not improve localization. On `httpx_3672` it produced a smaller, less complete patch than the probe. Reject.

**Decision:** prompt and sub-agent tuning on the 13-task screen is exhausted. The remaining lever is model adaptation.

### 6.10 Training pivot

- Downloading the trainable checkpoint `google/gemma-4-31B-it-qat-q4_0-unquantized` on the login node to `~/gemma4/models/gemma-4-31B-qat-unquantized`.
- First experiment: gold-patch SFT (rank-16 QLoRA, text backbone, response-only loss) on the 129 public tasks.
- Split policy (adopted 2026-10-28):
  - First adapter: **116 train / 13 holdout** (the current screen).
  - Confirmation: **103 train / 26 holdout**.
  - Final adapter: **129 train**, evidence from the hidden score; the old holdout is no longer valid after retraining.
  - Split by `instance_id`; `test_patch` is never a training target.
- If the adapter is active and wins a paired 13-task A/B, test trajectory SFT next.
- Do not submit an adapter until it beats the prompt-only baseline and passes the vLLM W4A16 activation check.

### 6.10 Gold-patch SFT smoke — passed

- Trainable base: `google/gemma-4-31B-it-qat-q4_0-unquantized` (59 GB), downloaded.
- Data: 128 examples (129 tasks minus `fastapi_15661`), 116 train / 13 holdout.
- 12k-char context version: p50 2,306 tokens; 108/128 ≤4,096 tokens.
- PEFT target: text `q/k/v/o/gate/up/down` only; vision/audio excluded. Trainable 122.4M (0.39%); adapter 234 MB.
- 2k smoke: 10 steps, loss 0.91→0.19, `grad_norm` non-zero, `lora_B` non-zero. Passed.
- 4k smoke: one step, 20k tokens, 51 s; fits on 1×A100 40 GB. 8k OOMs.
- Full 116-task, 2-epoch run started: `~/gemma4/adapters/gold-sft-r16`.

Next: serve the adapter on the W4A16 base, prove it changes outputs, then run the 13-task holdout A/B. Do not submit unless it wins.

### 6.11 Gold adapter served — A/B running

- Full SFT completed: 30 steps, 2 epochs, train loss 0.713, 23m22s, 234 MB adapter.
- Adapter keys: 820 text-only keys, `lora_B` non-zero (max 0.025).
- vLLM `Gemma4ForConditionalGeneration` rejects LoRA. Workaround: patch the staged W4A16 `config.json` to `architectures: ["Gemma4ForCausalLM"]`; vLLM then loads the text-only model with LoRA and maps adapter keys.
- Server: `scn6-10g:8000`, TP=1, 32k max len, 13,552 KV tokens (1×A100).
- `gold-holdout` is running on the 13 holdout tasks.
- Activation check at temperature 0 with logprobs was **inconclusive**: base and `gold` produced identical top-5 logprobs for one prompt. The 13-task A/B is the deciding test.

### 6.12 Gold adapter A/B — active but a wash

| Arm | Resolved | Patches | Mean | p95 |
|---|---:|---:|---:|---:|
| base probe 28 | **3/13** | 11/13 | 162.8 s | 227.5 s |
| gold adapter, alpha 16 | **3/13** | 12/13 | 297.7 s | 435.1 s |

Same resolved set (`fastapi_15589`, `requests_7427`, `rich_4077`). The gold patches are materially different from the base patches on every task, so the adapter **is applied**; the one-prompt identical-logprob check was a false negative caused by an easy first token.

The adapter is slower and does not improve resolved count. This matches the public LoRA evidence: alpha 16 causes longer thinking/time; alpha 8 is safer but an aggregate wash.

**Decision:** do not submit the gold adapter. Keep the prompt-only bundle. A useful adapter needs trajectory/thinking data, not gold-patch SFT alone.

### 7. Public trajectory SFT pivot (2026-10-28)

The public `dmitriigluzdov/gemma-4-serving-and-lora-verification-kit` dataset was found on Kaggle.

- 1,670 resolved agent trajectories; 56,702 prompt/target training steps.
- Native tier: 103 Gemma 4 31B Budget-Fit trajectories on 55 tasks.
- External A: 14 public trajectories within 28 calls.
- External B/B2: 1,553 Qwen3-Coder-480B trajectories, converted to the nine-tool format.
- Native unresolved: 56 failures for preference-pair analysis.
- License: CC BY 4.0 for data, Apache 2.0 for Qwen3 teacher text.

Downloaded to `~/gemma4/data/lora`:

```text
lora_native_resolved.jsonl  (198 MiB)
lora_external_A.jsonl       (18 MiB)
lora_data.py                (loader)
lora_index.csv              (per-trajectory index)
```

Built `~/gemma4/sft/public_sft.jsonl`:

- Tiers: native + external_A.
- Context cap: 8,192 tokens.
- Steps: 1,051.
- Format: prompt/completion, already in the competition chat template.

Training:

- Model: `/dev/shm/gemma4-31B-qat-unquantized` (staged on the compute node).
- 4×A100 40 GB DDP.
- QLoRA rank 16, alpha 16, text backbone only.
- `max_length=8192`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`.
- One optimizer step on 4 GPUs: ~214 s, loss 2.52, `grad_norm` 25.12.
- Full run started: 2 epochs on 1,051 steps → `~/gemma4/adapters/public-sft-r16`.
- Expected duration: ~4 hours.

This is the highest-signal training data found so far because it teaches the actual agent behavior (thoughts + tool calls + edits) rather than only the final diff.

### 6.13 Calls32 second seed — unstable

| Run | Resolved | Patches | p50 | p95 | NO_PATCH |
|---|---:|---:|---:|---:|---|
| calls32 (seed 1) | 3/13 | 12/13 | 319.2 s | 493.7 s | `rich_4076` |
| calls32b (seed 2) | **1/13** | 10/13 | 371.8 s | 483.0 s | `fastapi_15661`, `rich_4070`, `rich_4076` |

The +1 was not stable. Calls32b lost `fastapi_15589` and `rich_4079`, keeping only `rich_4077`. Average across the two 32-call runs is 2/13, the same as probe 28. The 13-task screen is too noisy to distinguish a one-task difference.

**Decision:** reject 32 calls. Keep **8 min / 28 calls** as the known-good prompt-only arm. Do not run more call-count arms; the next change must be structural (inventory sub-agent) or a larger evaluation set (20–30 tasks) before another promotion.

The exact public 0.22 bundle was also inspected: its prompt differs from our primary prompt by only one line (the parallel-implementations rule). So the prompt is not the main gap; the local 13-task subset is hard and noisy.

### 6.5 Phase 3 result — v2 did not improve

| Arm | Resolved | Patches | Mean | p95 | Projected 120 tasks |
|---|---:|---:|---:|---:|---:|
| probe 8/28/4096 | 2/13 | 12/13 | 324.6 s | 474.6 s | 10.8 h |
| v2 8/28/4096 + prompt additions | **2/13** | **11/13** | **344.8 s** | **464.8 s** | **11.5 h** |

V2 did not improve resolved count, lost one patch, and increased mean/projection. It also did not fix the multi-file failures:

- `fastapi_15030`: still 1/9 reference files.
- `rich_4070`: still 2/9.
- `httpx_3672`: 2/7 (probe had 4/7).
- `rich_4076` and `rich_4079`: still `NO_PATCH`.
- `fastapi_15661`: still creates `scripts/release.py` instead of `scripts/prepare_release.py`.

**Decision:** keep the probe prompt/budget as the current best; do not promote v2. The prompt addition alone is insufficient. Next step is trace/failure analysis (did the agent actually run the multi-file inventory? did it stop after the first fix?) and, if needed, a known-good public 0.22 prompt control.

Artifacts: `phase3/task-results-v2.csv`, `phase3/task-results-all.csv`.

---

## 7. Deviations from the main runbook

| Main runbook | Actual | Reason |
|---|---|---|
| Build harness on remote 4×A100 | Harness on Kaggle L4x4 | Remote Python 3.10/no Docker; wheelhouse is Kaggle-image-specific and dependency graph is unsatisfiable on a bare venv |
| Use Docker sandbox | `subprocess` sandbox | No Docker/sudo on remote; Kaggle notebook uses subprocess |
| Run 20–40 task holdout | 13 tasks | Calibrated 13/13; fast iteration; 13 is a screen, not statistical proof |
| Primary budget 5 min / 24 calls | Probe 8 min / 28 calls is current best | Primary resolved 0/13; probe resolved 2/13 and produced 12/13 patches |
| Consider more calls | 40 calls tested and rejected | Worse resolves, worse projection (12.7 h) |
| Use full 129 public tasks | 13-task subset for budget comparison | Cost/time; gold-calibrated subset |
| Start LoRA after baseline | Not started | Agent behavior/prompt is still the bottleneck; no adapter yet |

---

## 8. Open questions

1. Does prompt v2 improve resolved count on the same 13 tasks?
2. Can the budget drop to 7 min / 28 calls without losing resolves, to create global-time margin?
3. Why do `requests_*` patches touch the correct files but fail tests? Need test failure logs.
4. Is the hidden test set similar enough that a 13-task local screen is predictive? The public evidence says local-to-hidden transfer is weak.
5. What is the Phase 0 probe score/status? Still pending at last check.

---

## 9. Artifact index

| Artifact | Path |
|---|---|
| Phase 2 all-three notebook | `phase2/phase2-all-three.ipynb` |
| Phase 2 results | `phase2/task-results-{primary,probe,calls40,all}.csv` |
| Phase 3 prompt v2 notebook | `phase3/phase3-prompt-v2.ipynb` |
| Submission probe | `submission-probe.zip` |
| Primary bundle | `submission-primary.zip` |
| Main playbook | `README.md`, `docs/00`–`docs/07` |
| Source ledger | `docs/SOURCES.md` |

---

## 10. How to update this log

After each phase, append:

- date and phase;
- what was run (notebook, config, task count);
- result table;
- decision and reason;
- deviation from the runbook, if any;
- artifact path.

Do not rewrite history to make the plan look cleaner. The value of this log is that it records what actually happened.
