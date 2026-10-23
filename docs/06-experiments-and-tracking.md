# 06 — Experiments, Metrics, and Tracking

The competition is noisy. Identical files have scored 0.12–0.22 public (S34); the same byte-identical zip scored 0.10 and 0.15 (S39); a local 28/74 screen produced hidden 0.17 and 0.10 for tied configs (S35). A single score is not a measurement. This document defines the metrics, splits, and decision rules that make experiments interpretable.

The goal is not “run more experiments.” It is **make fewer, better experiments and know when a difference is real enough to spend a submission slot.**

---

## 1. What to measure

### 1.1 Outcome metrics

| Metric | Definition | Why |
|---|---|---|
| `patch_rate` | fraction of tasks with a non-empty patch | An unedited task cannot score (S5:536-537) |
| `resolve_rate` | fraction of tasks with `resolved=True` | The competition metric (S1) |
| `apply_fail_rate` | fraction of patches that fail to apply in Phase 2 | Patch hygiene/format (S5:548) |
| `collection_fail_rate` | fraction of test runs failing at collection | Environment vs agent (S31) |
| `timeout_rate` | fraction of tasks hitting per-task time | Budget fit (S5:499-509) |
| `global_projection` | projected total time for N≈120 | 12h risk (S1, S29) |

### 1.2 Process metrics

| Metric | Definition | Why |
|---|---|---|
| `first_edit_call` | tool-call index of first source edit | Public evidence: no edit by call 12 fails 87% (S38) |
| `tool_calls` | calls used per task | Budget fit (S5:499-509) |
| `turns` | model turns per task | Loop/termination behavior |
| `malformed_calls` | tool calls rejected for missing/invalid args | Parser/model issue (S30, S39) |
| `repeat_streak` | max consecutive identical calls | Loop risk (S39) |
| `compaction_events` | number of compactions | Context pressure (S5:518-521) |
| `prompt_tokens` / `completion_tokens` | per call | Context/time cost |
| `adapter_effect` | output difference base vs adapter at temp 0 | Adapter active (S23) |
| `startup_seconds` | vLLM server ready time | Startup risk (S26, S30) |
| `kv_cache_tokens` | `GPU KV cache size` | Long-prompt risk (S27) |

### 1.3 Quality metrics for trajectories

| Metric | Definition | Prefer |
|---|---|---|
| `resolved` | harness outcome | True |
| `efficiency` | calls/time vs budget | Lower |
| `style` | first edit by call 12, source-only, no test edits | Yes |
| `hygiene` | no scratch files, no test/config edits | Yes |
| `loopiness` | identical-call streak | Low |
| `leakage` | reads outside `/workspace`/wheels/site-packages | None |
| `thinking_match` | thought tokens match inference mode | Yes |
| `repo_balance` | distribution across repos | Balanced |

These are not all equally important. Start with outcome, patch rate, first-edit call, time, and repeat streaks.

---

## 2. Splits

### 2.1 Public data is not a random sample

The 129 public tasks come from four popular repos; the hidden tasks come from private repos (S2, S31). A random public split overestimates transfer. Use **repository-grouped** holdouts where possible, and keep a set of tasks you never tune on.

### 2.2 Recommended splits

| Split | Size | Use |
|---|---|---|
| Dev | 40–60 tasks | Prompt/budget iteration |
| Holdout A | 20–30 tasks | Paired comparisons; promotion decisions |
| Holdout B | 20–30 tasks | Final confirmation; untouched until the end |
| Training | remaining resolved trajectories | LoRA/QLoRA |
| Calibration | 10–15 tasks | Gold-patch/empty-patch checks |

These are subsets of the 129 public training tasks, not additive. Pick disjoint task IDs for Dev/Holdout A/Holdout B; training uses trajectories from tasks outside the holdouts. If you need more tasks, reuse Dev for exploration but never for promotion.

If you have limited compute, use 20–30 holdout tasks and accept wider error bars. Do not use 2 tasks (the starter) for decisions.

### 2.3 Task selection

Stratify by repo and difficulty:

- Include fastapi (many tasks), rich, requests, httpx.
- Include tasks with/without hints.
- Include short and long problem statements.
- Exclude known-broken tasks from comparison, but record why.

The public 0.18 analysis found statements ≤120 chars had 7 resolves in 32 attempts vs 24 in 56 for longer statements (S39); length is a proxy, not a cause. Balance it.

### 2.4 Gold/empty controls

For every evaluation environment:

- Gold patch resolves on the calibration tasks.
- Empty patch fails (or the task is marked broken).
- If gold fails, fix the environment before comparing agents (S31, S32).

---

## 3. Paired comparisons

### 3.1 Why paired

Run-to-run variance is large. Comparing config A on tasks 1–20 and config B on tasks 21–40 confounds config with task difficulty. Run both on the **same tasks**, same seeds, alternating order.

### 3.2 Design

- N = 20–40 tasks.
- Two seeds if time permits (two attempts per task).
- Alternate A/B order per task to balance drift.
- Same model revision, wheelhouse, prompt hash, eval config, GPU topology.
- Record everything in the manifest.

### 3.3 Analysis

For binary outcomes, use discordant pairs:

| | B resolved | B failed |
|---|---|---|
| A resolved | both | A only |
| A failed | B only | neither |

The sign test uses only A-only and B-only. With `d` discordant pairs and `g` wins for the candidate:

$$
p = \min\left(1,\;2\sum_{j=\max(g,d-g)}^{d} \binom{d}{j}2^{-d}\right)
$$

A 6–1 win gives p≈0.125; 8–2 gives p≈0.109; 9–1 gives p≈0.021. Small samples cannot establish small effects. Report the raw counts, not just a p-value.

For continuous metrics (time, calls), report median and p95, and the paired difference per task. A 10% mean improvement driven by one task is not a win.

### 3.4 Practical decision rule

Promote a candidate only if:

1. It has **≥8 discordant pairs and ≥75% wins** (or two seeds) on the paired holdout, **and**
2. It does not increase p95 time, setup time, or timeout rate materially, **and**
3. It does not reduce patch rate, **and**
4. The change is mechanistically plausible, **and**
5. It has a hidden score before it is eligible for a final slot.

At 6 discordant pairs, 2 wins gives p≈0.69; at 8, 2 wins gives p≈0.29; at 10, 2 wins gives p≈0.11. Report the raw discordant counts, not just the net. A 3+ task win with better p95/patch rate and a hidden score is a real candidate.

---

## 4. Run manifest

Every run must be reproducible. Save a JSON file next to the results:

```json
{
  "run_id": "2026-10-24-baseline-01",
  "date_utc": "2026-10-24T12:00:00Z",
  "competition": "gemma-4-developer-agent",
  "purpose": "baseline prompt-only on holdout A",
  "agent_dir": "submission/",
  "agent_zip_sha256": "...",
  "model": "gemma-4-31b-it-qat-w4a16-ct",
  "model_revision": "...",
  "vllm_version": "0.19.1",
  "vllm_wheel_sha256": "...",
  "wheelhouse_hash": "...",
  "python_version": "3.12.x",
  "cuda_version": "...",
  "gpu": "4x A100 40GB",
  "sandbox": "docker",
  "eval_config": {
    "timeout_seconds": 240,
    "max_tool_calls": 28,
    "max_time_minutes": 8,
    "max_turns": 80
  },
  "sampling": {
    "temperature": 0.2,
    "top_p": 0.95,
    "max_output_tokens": 8192,
    "thinking_budget": 4096,
    "include_thoughts": true
  },
  "prompt_sha256": "...",
  "tool_schema_sha256": "...",
  "adapter": null,
  "adapter_sha256": null,
  "adapter_rank": null,
  "adapter_alpha": null,
  "task_ids": ["..."],
  "seeds": [42],
  "git_commit": "...",
  "notes": "first baseline; no adapter"
}
```

For training runs, add:

```json
{
  "training_base": "google/gemma-4-31B",
  "training_base_revision": "...",
  "training_base_note": "use the verified BF16 ID or a verified QAT-unquantized ID/revision",
  "dataset_sha256": "...",
  "num_trajectories": 0,
  "num_samples": 0,
  "sequence_len": 8192,
  "rank": 16,
  "alpha": 16,
  "target_modules": "...",
  "learning_rate": 5e-5,
  "epochs": 1,
  "batch_size": 1,
  "grad_accum": 8,
  "optimizer": "adamw_8bit",
  "train_loss": 0.0,
  "val_loss": 0.0,
  "tokens_per_second": 0.0,
  "peak_vram_gb": 0.0,
  "gpu_hours": 0.0
}
```

Hash the prompt/config/zip. If you can’t reproduce the hash, you can’t reproduce the run.

---

## 5. Results schema

Use JSONL for per-task results:

```json
{"run_id":"...","task_id":"fastapi_15661","repo":"fastapi/fastapi","resolved":false,"patch_size":0,"files_changed":0,"tool_calls":10,"turns":8,"duration_s":70.2,"setup_s":55.0,"first_edit_call":null,"malformed_calls":0,"max_repeat_streak":3,"compaction_events":0,"error":"BudgetExceeded"}
```

Aggregate:

```json
{"run_id":"...","n":33,"resolved":9,"patch_rate":0.82,"apply_fail":0,"collection_fail":1,"timeout":2,"p50_s":210,"p95_s":480,"mean_calls":22.1,"mean_turns":18.4,"malformed_rate":0.04,"global_projection_h":9.8}
```

Keep the raw traces and patches; the aggregate is not enough to debug.

---

## 6. Experiment backlog

Run these in order. Do not skip the baseline.

### 6.1 Scaffold experiments

| ID | Change | Hypothesis | Metric |
|---|---|---|---|
| E1 | Reproduce public prompt-only config (author-reported 0.24) | Baseline works | resolve, patch rate, time |
| E2 | 5 vs 6 vs 8 min cap | Lower cap improves global completion without losing solves | resolve, timeout, projection |
| E3 | 24 vs 28 vs 32 calls | Fewer calls reduces loops; more calls helps hard tasks | resolve, patch rate, calls |
| E4 | thinking 0/512/2048/4096 | Bounded thinking reduces loops; too much costs time | resolve, malformed, time |
| E5 | output 8192 vs 12288/16384 | Higher cap reduces truncation but hurts prompt room | truncation, resolve |
| E6 | graph tools on/off | Graph tools help localization or waste calls | first edit, resolve |
| E7 | edit-by-call 10/12/14 | Earlier edits increase patches but may increase wrong edits | patch rate, resolve |
| E8 | read-only explorer sub-agent | Context isolation improves hard tasks | resolve, time, calls |
| E9 | prompt length 2k/4k/6k | Shorter prompt survives compaction better | resolve, compaction |
| E10 | temp 0.2 vs 0.7 | Lower temp reduces loops; higher explores | repeat streak, resolve |

### 6.2 Data/training experiments

| ID | Change | Hypothesis | Metric |
|---|---|---|---|
| T1 | 10-step smoke test | Fits GPU, loss finite | VRAM, tokens/s |
| T2 | 31B QLoRA rank 8 vs 16 | Lower rank more stable | loss, adapter effect, resolve |
| T3 | alpha 8/16 at inference (rank 16); 32 diagnostic | Scale 1.0 best; 2.0 degenerate | loops, time, resolve |
| T4 | all-layer vs last-N coverage | Partial coverage more stable on W4A16 | reproducibility, resolve |
| T5 | thinking vs no-thinking data | Match inference | thought length, time, resolve |
| T6 | 300 vs 1,000 trajectories | More data helps until quality drops | val loss, resolve |
| T7 | quality-filtered vs all resolved | Quality beats quantity | resolve |
| T8 | SRFT step masking | Unresolved data helps | resolve |
| T9 | 31B vs 12B adapter | 31B better despite speed | resolve, time |
| T10 | E4B LoRA | Small-model bonus | small-model score |

Do not run T1–T10 in parallel with E1–E10 on the same GPU without a queue; they compete for memory. Use separate GPUs or sequential runs.

---

## 7. Local-to-hidden calibration

You cannot fully calibrate local to hidden, but you can track the relationship:

- For each hidden submission, record the local holdout resolve rate and the hidden score.
- Plot local vs hidden; compute the ratio and residual.
- Do not fit a conversion factor on 2–3 points. Use it as a warning, not a prediction.
- If local improves but hidden does not across 2–3 submissions, your local set is not representative; change the holdout/repos.

Public evidence: local 28/74 tied configs scored 0.17 and 0.10 hidden (S35); local wins did not always transfer (S39). Treat local as a screening tool, not an oracle.

---

## 8. Decision log

Keep a short decision log:

```text
2026-10-24 — E1 baseline: 9/33, patch 82%, p95 480s. Projected 9.8h. Submit.
2026-10-26 — E2 5min: 8/33, patch 80%, p95 360s. Projected 8.2h. Not promoted (1-task loss, but time better; keep as fallback).
2026-10-28 — E4 thinking512: 11/33, patch 85%, p95 420s. Promoted for next submission.
2026-10-30 — T2 rank16 adapter: 12/33 vs base 9/33, but p95 560s. Alpha sweep needed.
```

The log prevents re-litigating the same experiment and preserves the reasoning when context is lost.

---

## 9. Statistical cautions

- **Multiple comparisons:** if you test 10 prompt variants on 20 tasks, one will look good by chance. Pre-register the main metric and limit variants.
- **Survivorship:** tasks that fail early may be dropped by the harness; don’t compare only the tasks that produced patches.
- **Correlated tasks:** tasks from the same repo share code/conventions; treat repo as a cluster, not independent samples.
- **Seed variance:** one seed is not a distribution. Two seeds on 20 tasks is better than one seed on 40; alternate order.
- **Display truncation:** public scores are two decimals; one public-LB task is ~1/58 ≈ 0.017, one final-split task is ~1/60 ≈ 0.017, and the 0.01 display step is smaller than one task (S4, S39). Do not overinterpret a 0.01 difference.
- **Global time coupling:** a config that solves more tasks but times out globally has zero utility. Always report completion risk with quality (S29, S39).

---

## 10. Experiment checklist

- [ ] Run ID and manifest saved.
- [ ] Prompt/config/zip hashes recorded.
- [ ] Model/adapter revisions recorded.
- [ ] Holdout task IDs fixed; no training overlap.
- [ ] Gold/empty controls pass.
- [ ] Same tasks/seeds/order for A/B.
- [ ] Raw traces/patches preserved.
- [ ] Per-task JSONL + aggregate.
- [ ] Decision and rationale logged.
- [ ] Known-good bundle updated only after a win.
- [ ] Submission slot not wasted on a 1-task local difference.
