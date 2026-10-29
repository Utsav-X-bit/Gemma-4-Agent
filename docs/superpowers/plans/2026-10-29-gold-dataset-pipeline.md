# Gold-Standard Gemma 4 Agent Dataset — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, execution-validated, process-filtered, human-reviewed dataset for Gemma 4 coding-agent SFT, with gold/silver/bronze tiers, repo-level holdouts, and a dataset card.

**Architecture:** A `phase6/` pipeline: task pool → trajectory generation → execution validation → process scoring → human review → per-call SFT formatting → dedupe/decontamination → dataset card → training/evaluation → hard-example iteration. Raw data lives under `~/gemma4/dataset-v1`; code, manifests, and the dataset card live in git.

**Tech Stack:** Python 3.12, `uv`, vLLM 0.19.1 (scorer wheel), Gemma 4 31B QAT W4A16, Docker sandbox, `safetensors`, `pytest`, `pandas`/`pyarrow`.

## Global Constraints

- Exact inference prompt, tool schema, and Gemma 4 chat template. Record `prompt_sha256` and `tool_schema_sha256` in every manifest.
- Execution validation before process scoring. A trajectory that does not pass F2P/P2P tests is never silver or gold.
- Never train on test/config edits, environment hacks, or scratch files.
- Cap **3 trajectories per task**; hold out entire repos/tasks for evaluation.
- No evaluation contamination: no training task/repo may appear in the holdout; scan for benchmark-specific strings.
- Deterministic generation seeds and deterministic validation reruns.
- QLoRA rank 16, alpha 8, text-backbone targets; 1–2 epochs; max length 8192.
- Large artifacts (snapshots, traces, adapters, datasets) stay outside git and are gitignored.
- Every phase writes a machine-readable report with counts, hashes, and rejection reasons.
- Gold requires human review; silver requires execution + process filtering; bronze is raw successful.

---

## File Structure

```text
phase6/
  dataset/
    schema.py              # dataclasses + tier invariants
    config.yaml            # paths, thresholds, seeds, budgets
    task_pool.py           # real + synthetic + PR-mirror task pool
    generate.py            # multi-temperature trajectory generation
    validate.py            # execution validation + determinism
    process_score.py       # TQS-like structural scoring
    human_review.py        # sampling, rubric, adjudication, kappa
    format_sft.py          # per-call prompt/completion + tokenization
    dedupe.py              # near-dedupe, decontamination, diversity
    card.py                # dataset card + manifest hashes
  train/
    train_adapter.py       # thin wrapper around phase5/train_gold_sft.py
  eval/
    evaluate.py            # base vs adapter, held-out metrics
  tests/
    test_schema.py
    test_task_pool.py
    test_validate.py
    test_process_score.py
    test_human_review.py
    test_format_sft.py
    test_dedupe.py
    test_card.py
```

Reuse existing code:

- `phase5/build_agent_sft.py` — per-call prompt/completion format.
- `phase5/train_gold_sft.py` — QLoRA training.
- `phase5/remap_adapter_prefix.py` — vLLM adapter export.
- `phase5/local_eval.py`, `phase5/local_verify.py` — Docker evaluator.
- `phase5/submission-v4-32-prefixed` — current inference prompt/tool schema.

---

## Phase 0 — Schema, config, and directories

### Task 0.1: Dataset schema and tiers

**Files:**
- Create: `phase6/dataset/schema.py`
- Test: `phase6/tests/test_schema.py`

**Interfaces:**
- `TaskRecord`, `TrajectoryRecord`, `StepRecord`, `ValidationRecord`, `ProcessScore`, `ReviewRecord`
- `Tier = Literal["gold", "silver", "bronze"]`

- [ ] **Step 1: Write the failing test**

```python
from phase6.dataset.schema import TaskRecord, TrajectoryRecord, Tier

def test_task_roundtrip():
    t = TaskRecord(instance_id="x", repo="a/b", base_commit="abc", source="public", split="train")
    assert TaskRecord.from_dict(t.to_dict()) == t

def test_invalid_tier_rejected():
    import pytest
    with pytest.raises(ValueError):
        TrajectoryRecord(task_id="x", trajectory_id="t", tier="platinum")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest phase6/tests/test_schema.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Tier = Literal["gold", "silver", "bronze"]

@dataclass
class TaskRecord:
    instance_id: str
    repo: str
    base_commit: str
    source: Literal["public", "synthetic", "pr_mirror"]
    split: Literal["train", "val", "test"]
    difficulty: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskRecord":
        return cls(**d)

@dataclass
class StepRecord:
    index: int
    action: str
    thought: str
    observation: str
    files: list[str] = field(default_factory=list)

@dataclass
class TrajectoryRecord:
    task_id: str
    trajectory_id: str
    tier: Tier
    steps: list[StepRecord]
    patch: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tier not in ("gold", "silver", "bronze"):
            raise ValueError(f"invalid tier: {self.tier}")

@dataclass
class ValidationRecord:
    trajectory_id: str
    resolved: bool
    f2p_pass: bool
    p2p_pass: bool
    patch_applies: bool
    test_files_touched: bool
    deterministic: bool
    exit_code: int
    test_output_path: str

@dataclass
class ProcessScore:
    trajectory_id: str
    first_edit_fraction: float
    repeated_call_rate: float
    repeated_error_rate: float
    scope_narrowing: float
    intent_action_consistency: float
    no_edit: bool
    early_stop: bool
    scratch_file: bool
    malformed_call_rate: float
    composite: float

@dataclass
class ReviewRecord:
    trajectory_id: str
    reviewer: str
    correctness: int
    localization: int
    minimality: int
    process_efficiency: int
    tool_discipline: int
    reasoning_action_alignment: int
    recovery: int
    format_compliance: int
    notes: str = ""
    gold: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest phase6/tests/test_schema.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add phase6/dataset/schema.py phase6/tests/test_schema.py
git commit -m "feat(dataset): add schema and tier invariants"
```

### Task 0.2: Config and directory layout

**Files:**
- Create: `phase6/dataset/config.yaml`
- Create: `phase6/dataset/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write config**

```yaml
paths:
  data_root: ~/gemma4/dataset-v1
  snapshots: ~/gemma4/data/snapshots
  public_tasks: ~/gemma4/data/tasks.jsonl
  synthetic_tasks: ~/gemma4/dataset-v1/synthetic/tasks.jsonl
  pr_mirror_tasks: ~/gemma4/dataset-v1/pr_mirror/tasks.jsonl
  traces: ~/gemma4/dataset-v1/traces
  validated: ~/gemma4/dataset-v1/validated
  scored: ~/gemma4/dataset-v1/scored
  reviews: ~/gemma4/dataset-v1/reviews
  sft: ~/gemma4/dataset-v1/sft
  final: ~/gemma4/dataset-v1/final

generation:
  temperatures: [0.2, 0.4, 0.6, 0.8, 1.0]
  seeds: [0, 1, 2]
  max_tool_calls: 32
  max_time_minutes: 8
  max_output_tokens: 8192
  thinking_budget: 4096

validation:
  rerun_count: 2
  max_test_seconds: 300

process:
  keep_fraction: 0.40
  min_steps: 3
  max_repeated_call_rate: 0.25
  max_repeated_error_rate: 0.30

diversity:
  max_trajectories_per_task: 3
  holdout_repo_fraction: 0.20

training:
  rank: 16
  alpha: 8
  epochs: 2
  max_length: 8192
  lr: 5e-5
```

- [ ] **Step 2: Ignore large artifacts**

Append to `.gitignore`:

```gitignore
# Phase 6 dataset artifacts
~/gemma4/dataset-v1/
phase6/dataset/data/
```

Use repo-relative ignores if the data root is inside the repo:

```gitignore
phase6/dataset/data/
```

- [ ] **Step 3: Commit**

```bash
git add phase6/dataset/config.yaml phase6/dataset/__init__.py .gitignore
git commit -m "chore(dataset): add pipeline config and ignores"
```

---

## Phase 1 — Task pool

### Task 1.1: Build the public task pool and repo-level split

**Files:**
- Create: `phase6/dataset/task_pool.py`
- Test: `phase6/tests/test_task_pool.py`

**Interfaces:**
- `build_task_pool(config) -> list[TaskRecord]`
- `split_by_repo(tasks, holdout_fraction, seed) -> list[TaskRecord]`

- [ ] **Step 1: Write the failing test**

```python
from phase6.dataset.task_pool import split_by_repo
from phase6.dataset.schema import TaskRecord

def test_no_repo_overlap():
    tasks = [TaskRecord(f"t{i}", f"repo{i%4}", "abc", "public", "train") for i in range(20)]
    split = split_by_repo(tasks, 0.25, seed=0)
    train = {t.repo for t in split if t.split == "train"}
    test = {t.repo for t in split if t.split == "test"}
    assert not (train & test)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest phase6/tests/test_task_pool.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
import random
from pathlib import Path
import json
from phase6.dataset.schema import TaskRecord

def split_by_repo(tasks: list[TaskRecord], holdout_fraction: float, seed: int) -> list[TaskRecord]:
    repos = sorted({t.repo for t in tasks})
    rng = random.Random(seed)
    rng.shuffle(repos)
    n_test = max(1, int(len(repos) * holdout_fraction))
    test_repos = set(repos[:n_test])
    for t in tasks:
        t.split = "test" if t.repo in test_repos else "train"
    return tasks

def load_public_tasks(path: Path) -> list[TaskRecord]:
    out = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        out.append(TaskRecord(
            instance_id=d["instance_id"],
            repo=d["repo"],
            base_commit=d["base_commit"],
            source="public",
            split="train",
        ))
    return out
```

- [ ] **Step 4: Run test**

Run: `pytest phase6/tests/test_task_pool.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add phase6/dataset/task_pool.py phase6/tests/test_task_pool.py
git commit -m "feat(dataset): build public task pool with repo-level split"
```

### Task 1.2: Add synthetic and PR-mirror tasks

**Files:**
- Modify: `phase6/dataset/task_pool.py`
- Test: `phase6/tests/test_task_pool.py`

- [ ] **Step 1: Add loader**

```python
def load_extra_tasks(path: Path, source: str) -> list[TaskRecord]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        out.append(TaskRecord(
            instance_id=d["instance_id"],
            repo=d["repo"],
            base_commit=d["base_commit"],
            source=source,
            split="train",
            difficulty=d.get("difficulty", "unknown"),
        ))
    return out
```

- [ ] **Step 2: Deduplicate by `instance_id` and `repo+base_commit+patch_hash`**

```python
def dedupe_tasks(tasks: list[TaskRecord]) -> list[TaskRecord]:
    seen = set()
    out = []
    for t in tasks:
        key = (t.instance_id, t.repo, t.base_commit)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out
```

- [ ] **Step 3: Write manifest**

```python
def write_task_pool(tasks: list[TaskRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(t.to_dict()) for t in tasks) + "\n")
```

- [ ] **Step 4: Test counts and source mix**

```python
def test_sources_and_counts():
    tasks = (
        [TaskRecord(f"p{i}", f"repo{i}", "abc", "public", "train") for i in range(4)]
        + [TaskRecord(f"s{i}", f"srepo{i}", "abc", "synthetic", "train") for i in range(2)]
        + [TaskRecord(f"m{i}", f"mrepo{i}", "abc", "pr_mirror", "train") for i in range(2)]
    )
    assert len(tasks) == 8
    assert {t.source for t in tasks} == {"public", "synthetic", "pr_mirror"}
```

- [ ] **Step 5: Commit**

```bash
git add phase6/dataset/task_pool.py phase6/tests/test_task_pool.py
git commit -m "feat(dataset): add synthetic and PR-mirror task loaders"
```

---

## Phase 2 — Multi-temperature trajectory generation

### Task 2.1: Generation wrapper

**Files:**
- Create: `phase6/dataset/generate.py`
- Test: `phase6/tests/test_generate.py`

**Interfaces:**
- `generate_for_task(task, temperature, seed, agent_dir, out_dir) -> Path`
- `run_generation(tasks, config) -> list[Path]`

- [ ] **Step 1: Write the failing test**

```python
def test_generation_idempotent(tmp_path, monkeypatch):
    task = TaskRecord("t1", "a/b", "abc", "public", "train")
    p = result_path(tmp_path, task, 0.2, 0)
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("reran")))
    assert generate_for_task(task, 0.2, 0, tmp_path / "agent", tmp_path) == p
```

- [ ] **Step 2: Implement using the existing local evaluator**

```python
import json, subprocess
from pathlib import Path

def result_path(out_dir: Path, task, temperature: float, seed: int) -> Path:
    return out_dir / task.instance_id / f"t{temperature}_s{seed}" / "result.json"

def generate_for_task(task, temperature, seed, agent_dir, out_dir):
    result = result_path(out_dir, task, temperature, seed)
    task_dir = result.parent
    if result.exists():
        return result
    task_dir.mkdir(parents=True, exist_ok=True)
    agent = task_dir / "agent"
    subprocess.run(["cp", "-r", str(agent_dir), str(agent)], check=True)
    # edit configs/sampling.yaml temperature/seed
    cfg = agent / "configs/sampling.yaml"
    text = cfg.read_text()
    text = text.replace("temperature: 0.2", f"temperature: {temperature}")
    cfg.write_text(text)
    cmd = [
        ".venv-local/bin/python", "phase5/local_eval.py",
        "--agent-dir", str(agent),
        "--tag", f"{task.instance_id}-t{temperature}-s{seed}",
        "--task-id", task.instance_id,
        "--per-repo", "4",
        "--host", "127.0.0.1", "--port", "8007",
    ]
    subprocess.run(cmd, check=True)
    # local_eval writes local-results/task-results-<tag>.csv and traces
    result.write_text(json.dumps({"temperature": temperature, "seed": seed}))
    return result
```

- [ ] **Step 3: Test one task smoke**

Run:

```bash
python -m phase6.dataset.generate --tasks phase6/dataset/data/task_pool.jsonl --limit 1 --temps 0.2
```

Expected: one `result.json` and one trace.

- [ ] **Step 4: Commit**

```bash
git add phase6/dataset/generate.py phase6/tests/test_generate.py
git commit -m "feat(dataset): add multi-temperature generation wrapper"
```

---

## Phase 3 — Execution validation

### Task 3.1: Validate a trajectory patch

**Files:**
- Create: `phase6/dataset/validate.py`
- Test: `phase6/tests/test_validate.py`

**Interfaces:**
- `validate_trajectory(task, trace_path, result_path, config) -> ValidationRecord`

- [ ] **Step 1: Write the failing test**

```python
def test_empty_patch_rejected(tmp_path):
    task = TaskRecord("t1", "a/b", "abc", "public", "train")
    trace = tmp_path / "trace.json"
    trace.write_text('{"steps": []}')
    result = tmp_path / "result.json"
    result.write_text('{"prediction": ""}')
    rec = validate_trajectory(
        task, trace, result, None,
        verify=lambda *_: (_ for _ in ()).throw(AssertionError("must not verify empty patch")),
    )
    assert rec.resolved is False
    assert rec.exit_code == -1
```

- [ ] **Step 2: Implement using `verify_task`**

```python
import json
from pathlib import Path
from phase6.dataset.schema import ValidationRecord

def extract_patch(result_path: Path) -> str:
    d = json.loads(Path(result_path).read_text())
    return d.get("prediction", "")

def validate_trajectory(task, trace_path, result_path, config, verify=None):
    patch = extract_patch(result_path)
    tid = Path(trace_path).stem
    if not patch.strip():
        return ValidationRecord(
            trajectory_id=tid,
            resolved=False, f2p_pass=False, p2p_pass=False,
            patch_applies=False, test_files_touched=False,
            deterministic=False, exit_code=-1, test_output_path="",
        )
    verify = verify or _verify_with_harness
    r1 = verify(task, patch)
    r2 = verify(task, patch)
    deterministic = (r1.exit_code, r1.resolved) == (r2.exit_code, r2.resolved)
    return ValidationRecord(
        trajectory_id=tid,
        resolved=r1.resolved and r2.resolved,
        f2p_pass=r1.f2p_pass,
        p2p_pass=r1.p2p_pass,
        patch_applies=r1.patch_applies,
        test_files_touched=r1.test_files_touched,
        deterministic=deterministic,
        exit_code=r1.exit_code,
        test_output_path=r1.test_output_path,
    )
```

- [ ] **Step 3: Test gold vs empty vs test-edit**

Run:

```bash
pytest phase6/tests/test_validate.py -q
```

Expected: gold passes; empty fails; test-edit rejected.

- [ ] **Step 4: Commit**

```bash
git add phase6/dataset/validate.py phase6/tests/test_validate.py
git commit -m "feat(dataset): add execution validation and determinism check"
```

---

## Phase 4 — Process scoring

### Task 4.1: TQS-like structural score

**Files:**
- Create: `phase6/dataset/process_score.py`
- Test: `phase6/tests/test_process_score.py`

**Interfaces:**
- `score_trajectory(trace) -> ProcessScore`

- [ ] **Step 1: Write the failing test**

```python
def test_good_beats_bad():
    good = synthetic_trace(first_edit=3, repeats=0, errors=0)
    bad = synthetic_trace(first_edit=20, repeats=8, errors=5)
    assert score_trajectory(good).composite > score_trajectory(bad).composite
```

- [ ] **Step 2: Implement features**

```python
import json

def score_trajectory(trace):
    steps = trace["steps"]
    edits = [i for i, s in enumerate(steps) if any(tc["function_name"] in ("edit_file", "write_file") for tc in s.get("tool_calls", []))]
    first_edit_fraction = edits[0] / max(1, len(steps)) if edits else 1.0
    calls = [json.dumps(tc.get("arguments"), sort_keys=True) for s in steps for tc in s.get("tool_calls", [])]
    repeated_call_rate = 1 - len(set(calls)) / max(1, len(calls))
    errors = [s for s in steps if '"error"' in json.dumps(s.get("observation", {}))]
    repeated_error_rate = 1 - len({json.dumps(s.get("observation", {}), sort_keys=True) for s in errors}) / max(1, len(errors))
    no_edit = not edits
    scratch = any("repro" in str(tc.get("arguments", {})) for s in steps for tc in s.get("tool_calls", []))
    malformed = sum(1 for s in steps if "mandatory input parameters" in json.dumps(s.get("observation", {}))) / max(1, len(calls))
    composite = (
        0.30 * (1 - first_edit_fraction)
        + 0.25 * (1 - repeated_call_rate)
        + 0.20 * (1 - repeated_error_rate)
        + 0.15 * (1 - malformed)
        + 0.10 * (0 if no_edit else 1)
    )
    return ProcessScore(
        trajectory_id=trace.get("trajectory_id", ""),
        first_edit_fraction=first_edit_fraction,
        repeated_call_rate=repeated_call_rate,
        repeated_error_rate=repeated_error_rate,
        scope_narrowing=0.0,
        intent_action_consistency=0.0,
        no_edit=no_edit,
        early_stop=first_edit_fraction > 0.8,
        scratch_file=scratch,
        malformed_call_rate=malformed,
        composite=composite,
    )
```

**v1 scope:** the first implementation uses first-edit fraction, repeated-call rate, repeated-error rate, malformed-call rate, no-edit, and scratch-file signals. `scope_narrowing` and `intent_action_consistency` are Phase 4.2 extensions and must be implemented before the final dataset; the v1 score is a gate, not the final TQS.

- [ ] **Step 3: Test score range and ordering**

Run: `pytest phase6/tests/test_process_score.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add phase6/dataset/process_score.py phase6/tests/test_process_score.py
git commit -m "feat(dataset): add TQS-like process scoring"
```

---

## Phase 5 — Human review and gold tier

### Task 5.1: Stratified review sampling

**Files:**
- Create: `phase6/dataset/human_review.py`
- Test: `phase6/tests/test_human_review.py`

**Interfaces:**
- `sample_for_review(trajectories, n=200, seed=0) -> list[str]`
- `write_review_sheet(ids, path)`
- `compute_agreement(reviews) -> dict`

- [ ] **Step 1: Write the failing test**

```python
def test_sample_stratified():
    ids = sample_for_review(records, n=10, seed=0)
    assert len(ids) == 10
    assert len(set(ids)) == 10
```

- [ ] **Step 2: Implement sampling and rubric CSV**

```python
FIELDS = ["correctness", "localization", "minimality", "process_efficiency",
          "tool_discipline", "reasoning_action_alignment", "recovery", "format_compliance"]

def write_review_sheet(ids, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trajectory_id", "reviewer", *FIELDS, "notes", "gold"])
        for tid in ids:
            w.writerow([tid, "", *([""] * len(FIELDS)), "", ""])
```

- [ ] **Step 3: Implement Cohen's kappa**

```python
def cohen_kappa(a, b):
    assert len(a) == len(b) and len(a) > 0
    n = len(a)
    labels = set(a) | set(b)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)
```

- [ ] **Step 4: Test agreement and duplicate reviewer prevention**

Run: `pytest phase6/tests/test_human_review.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add phase6/dataset/human_review.py phase6/tests/test_human_review.py
git commit -m "feat(dataset): add human review sampling and agreement"
```

---

## Phase 6 — Per-call SFT formatting

### Task 6.1: Format one assistant turn per row

**Files:**
- Create: `phase6/dataset/format_sft.py`
- Test: `phase6/tests/test_format_sft.py`

**Interfaces:**
- `build_rows(trajectory, system_prompt, tool_block, task_prompt) -> list[dict]`
- `tokenize_rows(rows, tokenizer, max_length) -> Dataset`

- [ ] **Step 1: Write the failing test**

```python
def test_prompt_masks_completion():
    rows = build_rows(trace, system_prompt, tool_block, task_prompt)
    assert rows[0]["prompt"].endswith("<|turn>model\n")
    assert rows[0]["completion"].startswith("<|channel>thought")
    # tokenized labels must be -100 on prompt tokens
```

- [ ] **Step 2: Reuse the `phase5/build_agent_sft.py` format**

Use the same `<bos><|turn>system`, `<|turn>user`, `<|turn>model`, `<|channel>thought`, `<|tool_call>`, `<|tool_response>` serialization.

- [ ] **Step 3: Tokenize with completion-only loss**

```python
def tokenize_row(row, tokenizer, max_length):
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    comp_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + comp_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + comp_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}
```

- [ ] **Step 4: Test labels and template hash**

Run: `pytest phase6/tests/test_format_sft.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add phase6/dataset/format_sft.py phase6/tests/test_format_sft.py
git commit -m "feat(dataset): format per-call SFT rows"
```

---

## Phase 7 — Dedupe, decontamination, diversity

### Task 7.1: Dedupe and contamination scan

**Files:**
- Create: `phase6/dataset/dedupe.py`
- Test: `phase6/tests/test_dedupe.py`

**Interfaces:**
- `dedupe(rows, threshold=0.9) -> list[dict]`
- `assert_no_eval_overlap(tasks, holdout)`
- `balance(rows, max_per_task=3) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
def test_injected_duplicate_removed():
    rows = [{"completion": "abc"}, {"completion": "abc"}, {"completion": "xyz"}]
    assert len(dedupe(rows, 0.9)) == 2

def test_cap_enforced():
    rows = [{"task_id": "t"} for _ in range(10)]
    assert len(balance(rows, 3)) == 3
```

- [ ] **Step 2: Implement MinHash/SimHash or exact hash for v1**

```python
def dedupe(rows, threshold=0.9):
    seen = set()
    out = []
    for r in rows:
        h = hashlib.sha256(r["completion"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out
```

- [ ] **Step 3: Test eval overlap and cap**

Run: `pytest phase6/tests/test_dedupe.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add phase6/dataset/dedupe.py phase6/tests/test_dedupe.py
git commit -m "feat(dataset): add dedupe, contamination, and cap"
```

---

## Phase 8 — Dataset card and manifest

### Task 8.1: Dataset card

**Files:**
- Create: `phase6/dataset/card.py`
- Test: `phase6/tests/test_card.py`

**Interfaces:**
- `write_card(out_dir, stats, hashes) -> Path`

- [ ] **Step 1: Write the failing test**

```python
def test_card_contains_required_fields(tmp_path):
    p = write_card(tmp_path, stats, hashes)
    text = p.read_text()
    for field in ["source tasks", "prompt_sha256", "tool_schema_sha256", "gold", "silver", "bronze", "limitations"]:
        assert field in text
```

- [ ] **Step 2: Implement**

```python
def write_card(out_dir, stats, hashes):
    lines = [
        "# Gemma 4 Agent Dataset v1",
        f"- source tasks: {stats['tasks']}",
        f"- repos: {stats['repos']}",
        f"- trajectories: {stats['trajectories']}",
        f"- gold/silver/bronze: {stats['gold']}/{stats['silver']}/{stats['bronze']}",
        f"- prompt_sha256: {hashes['prompt']}",
        f"- tool_schema_sha256: {hashes['tools']}",
        f"- filters: {stats['filters']}",
        f"- limitations: {stats['limitations']}",
    ]
    p = out_dir / "README.md"
    p.write_text("\n".join(lines) + "\n")
    return p
```

- [ ] **Step 3: Test and commit**

Run: `pytest phase6/tests/test_card.py -q`

```bash
git add phase6/dataset/card.py phase6/tests/test_card.py
git commit -m "feat(dataset): add dataset card"
```

---

## Phase 9 — Training and evaluation

### Task 9.1: Train on gold+silver

**Files:**
- Create: `phase6/train/train_adapter.py`
- Test: smoke command

- [ ] **Step 1: Wrap `phase5/train_gold_sft.py`**

```python
# phase6/train/train_adapter.py
import subprocess, sys
subprocess.run([
    sys.executable, "phase5/train_gold_sft.py",
    "--model", "~/gemma4/models/gemma-4-31B-qat-unquantized",
    "--data", "~/gemma4/dataset-v1/final/train.jsonl",
    "--all-data",
    "--output-dir", "~/gemma4/adapters/dataset-v1-r16",
    "--rank", "16", "--alpha", "8", "--epochs", "2", "--max-length", "8192",
], check=True)
```

- [ ] **Step 2: Remap and serve**

```bash
python phase5/remap_adapter_prefix.py \
  --src ~/gemma4/adapters/dataset-v1-r16 \
  --dst ~/gemma4/adapters/dataset-v1-r16-vllm
```

- [ ] **Step 3: Activation A/B**

Compare base vs adapter logprobs; require a difference.

- [ ] **Step 4: Held-out evaluation**

Run `phase5/local_eval.py` on held-out tasks; compare base, random-subset adapter, and top-quality adapter. Metrics: resolved, patch rate, calls, time-to-first-edit, loop rate, malformed calls.

- [ ] **Step 5: Promote only if held-out resolve improves**

Record the comparison in `phase6/eval/results.json`.

---

## Phase 10 — Hard-example iteration

### Task 10.1: Mine failures and add recovery/contrastive pairs

**Files:**
- Create: `phase6/dataset/hard_examples.py`
- Test: `phase6/tests/test_hard_examples.py`

- [ ] **Step 1: Select failures where localization is correct but patch is wrong**
- [ ] **Step 2: Add successful recovery trajectories for the same task**
- [ ] **Step 3: Add contrastive pairs (good process vs bad process)**
- [ ] **Step 4: Re-run phases 3–9**
- [ ] **Step 5: Update dataset card and version**

---

## Acceptance Criteria

- [ ] 50–100 tasks across 20–30 repos.
- [ ] 500–2,000 successful trajectories.
- [ ] ≤3 trajectories per task.
- [ ] 100–200 human-reviewed trajectories; 20% double-reviewed; kappa ≥ 0.6.
- [ ] Gold/silver/bronze split documented.
- [ ] No train/eval repo overlap; contamination scan clean.
- [ ] Every trajectory execution-validated; no test edits; deterministic rerun.
- [ ] Every SFT row uses the exact inference prompt/tool schema; prompt labels masked.
- [ ] Adapter activation proven by logprobs.
- [ ] Held-out resolve improves over base and random-subset adapter.
- [ ] Dataset card and manifest hashes complete.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Synthetic distribution shift | Mix real PR tasks, synthetic bugs, and PR mirrors; evaluate on real holdout |
| Weak tests / false positives | F2P + P2P, deterministic rerun, human review |
| Overfitting to easy tasks | Cap 3 per task; balance repos/difficulty; hard-example mining |
| Contamination | Repo-level split; scan eval strings; no test patches in training |
| Model collapse | Mix teacher, real, synthetic, and human trajectories |
| Cost/time | Start with 50 tasks; scale only after held-out gain |
| Format mismatch | Freeze prompt/tool hashes; regenerate if either changes |
