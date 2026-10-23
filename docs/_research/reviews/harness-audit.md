# Harness audit — `01`–`04` vs. local sources

Scope: `docs/01-harness-and-evaluation.md`, `docs/02-agent-engineering.md`, `docs/03-post-training.md`,
`docs/04-runbook.md`, checked against `harness_readme_raw.md` (627 lines), `getting_started_notebook.txt`
(615 lines), `docs/_research/harness-facts.md`, `docs/_research/notebook-facts.md`, and the `docs/SOURCES.md`
ledger. Audit only; no doc was edited.

Method: every numeric constant, command, flag, file path, and `[S5:*]`/`[S6:*]` line reference was located
in the local files with `grep -n`/`sed -n` and compared to the claim. Community (`S23`–`S40`) claims were
checked against the ledger entry text, since the underlying pages are not available offline.

**Verdict:** the `S5`/`S6` line numbers themselves are almost all correct (see §6). The defects are
(a) six claims that conflict with the local sources (C1, C2, H1, H5, H6) or with another doc (C3);
(b) runbook commands that would fail, install the wrong thing, or cannot be run as written (C1, H1, M8,
M9, M15, L11); (c) claims attributed to an `S`-number whose ledger entry does not contain them (H3, H4,
M10–M12, M14, L5, plus the S38/S37 mix-up in C2); and (d) boundary/conditional statements written more
strongly than the source supports (M1, M2, M5, M6, M16).

---

## 1. Severity summary

| ID | Sev | Where | One-line |
|---|---|---|---|
| C1 | Critical | `04:208`, `04:567` | `build_submission_limits()` returns a 2-tuple; passing it as `limits=` is wrong |
| C2 | Critical | `01:348` | "starter notebook's 8-minute cap" — the starter notebook's cap is 1 minute |
| C3 | Critical | `02:111`, `02:137` vs `01:331` | Phase-2 pytest timeout: 240 s vs 300 s, both stated as fact |
| C4 | Critical | `01:25` | "skill `.py` scripts run inside the sandbox" — not stated anywhere in S5 |
| H1 | High | `04:136` | wheelhouse install omits cutlass/`+cu128` handling that the notebook requires |
| H2 | High | `01:230` | lifecycle step 16 says `exit_code == 0` is sufficient; contradicts `01:§9.2` and S5 |
| H3 | High | `02:49,53,61` | three claims cited to `S28`, none of which appear in the `S28` ledger entry |
| H4 | High | `01:477`, `02:228` | Gemma tool-call token/parsing claim cited to `S39`; `S39` has no such fact |
| H5 | High | `03:576` | "starter notebook's config" listed as fixed LoRA values; notebook computes them |
| H6 | High | `03:371`, `03:419` vs `03:99` | 31B recipes set `chat_template: gemma4` while §4.2 requires thinking-matched training |
| M1 | Medium | `01:93`, `03:230`, `04:585` | `<3 GiB` vs the notebook's `<=` check, and `= 3,221,225,472` is not `<` 3 GiB |
| M2 | Medium | `01:293` | `budget_warning` condition is `max_tool_calls >= 20`, not "after 20 calls" |
| M3 | Medium | `01:297` | `run_command` timeout omits the `max(5, …)` floor |
| M4 | Medium | `01:67` | `eval_config.yaml` key list omits `verification_timeout_seconds` (S6:344) |
| M5 | Medium | `01:188` | "all agents must declare the same model" — source says "at most one unique model" |
| M6 | Medium | `03:215` | "rank 128 is the maximum" — nothing clamps it; 128 is the fallback |
| M7 | Medium | `03:43`, `03:301` | 61.4 GB BF16 attributed to `S7`, which says 69.9 GB |
| M8 | Medium | `04:252-289` | manual `vllm serve` + `swegemma eval` sequence has no model-name/endpoint bridge |
| M9 | Medium | `04:283`, `04:362` | `--sandbox docker` used without ever building `swebench-sandbox:latest` |
| M10 | Medium | `02:183` | "36 calls worse than 28 (S39)" — not in the `S39` entry |
| M11 | Medium | `02:224` | "~3,700 characters (S36)" — not in the `S36` entry |
| M12 | Medium | `03:123` | "12,602 tokens (S25)" — not in the `S25` entry |
| M13 | Medium | `03:464-465`, `03:499` | `unsloth/gemma-4-31B-it` and `SFTConfig(max_seq_length=…)` unverified / stale API |
| M14 | Medium | `04:631` | "Kaggle auto-selects" cited to `S1`; `S1` has no such fact |
| M15 | Medium | `04:405` | `swegemma eval ...` with a literal `...` cannot be run |
| M16 | Medium | `01:63` | `/kaggle/working/submission` presented as the extraction target; S5:166 only mentions it |
| M17 | Medium | `02:300` | "continue from `end_line`" re-reads the last line; should be `end_line + 1` |
| L1 | Low | `01:52-56` | `skills/*/scripts/` and `resources/` added to a tree cited to S5:63-82 |
| L2 | Low | `01:§3.4`, `02:§2.1` | temp 0.2 recommended without noting S8's recommended sampling (1.0/0.95/64) |
| L3 | Low | `01:246` | drops S5:268's "unless required by the task" qualifier |
| L4 | Low | `02:§5.6` | "256-d embeddings … public and private" — 256-d is only established for public (S2) |
| L5 | Low | `04:§1.2` | `wheels/` role cited to S6:55; S2 is the source |
| L6 | Low | `04:§7.1` | `--max_steps 10` is not a documented Axolotl CLI override |
| L7 | Low | `01:5` | "no way to modify the test runner" — edits are possible, they are just stripped |
| L8 | Low | `03:§4.2` | `gemma-4` described as the "non-thinking variant"; S10 calls it the small-model template |
| L9 | Low | `04:§2.3` | "so `agent.yaml` is the first path" — zip entry order is not guaranteed |
| L10 | Low | `01:67` | notebook also accepts top-level eval keys (`S6:341`); docs imply `evaluation:` is mandatory |
| L11 | Low | `04:375` | the "Collect" snippet reads `summary.json` from cwd; the run wrote it under `$RUNS/<run>/` |

---

## 2. Critical

### C1 — `build_submission_limits()` returns a tuple, not a `SubmissionLimits`

**Where:** `docs/04-runbook.md:208` (§2.2) and `docs/04-runbook.md:567` (§9.1).

**Claim as written:**
```python
validate_directory(Path(os.environ["AGENT"]), limits=build_submission_limits())
...
validate_directory(str(agent), limits=build_submission_limits())
```

**Evidence:** `getting_started_notebook.txt:352` — `limits, gen_constraints = build_submission_limits()`.
The function returns `(SubmissionLimits, GenerationConstraints)`; the same object is later passed as
`validate_directory(AGENT_DIR, limits=limits)` at `getting_started_notebook.txt:597` (a `Path`, not `str`).
`docs/_research/notebook-facts.md` records the same two-value unpacking.

**Fix:** in both snippets, insert `limits, _ = build_submission_limits()` and pass `limits=limits`. In §9.1
pass the `Path` (`validate_directory(agent, limits=limits)`) to match S6:597. Also note that §9.1 imports
`MAX_SUBMISSION_SIZE_BYTES` and `ALLOWED_SUBMISSION_EXTENSIONS` and never uses them — the size/extension
checks in §9.2 are manual, so either wire them up (copy S6:606-611) or drop the imports.

---

### C2 — "The starter notebook's 8-minute per-task cap" is false

**Where:** `docs/01-harness-and-evaluation.md:348` (§8.3).

**Claim:** "The starter notebook’s 8-minute per-task cap is not a guarantee of global completion; the
public 0.18 notebook explicitly says so (S38)."

**Evidence:** the starter notebook's sample `eval_config.yaml` is `max_time_minutes: 1`
(`harness_readme_raw.md:493-497`), and the observed starter run shows `Time 1.0m / 1m (0.0m left)`
(`getting_started_notebook.txt:509`, second task at `:575`). The 8-minute cap belongs to the public
0.24/0.22/0.18 configs (`docs/SOURCES.md` S36/S37/S38). The "not a guarantee of the global 12-hour limit"
statement is attributed in the ledger to **S37** ("Explicitly says the 8-minute cap is not a guarantee…"),
not S38.

**Fix:** "The public 0.18/0.22 notebook's 8-minute per-task cap is not a guarantee of global completion
(S37); the starter notebook's cap is 1 minute (S5:493-497)." If S38 is intended, add that sentence to the
S38 ledger entry first.

---

### C3 — Phase-2 pytest timeout is 300 s in `01` and 240 s in `02`

**Where:** `docs/01-harness-and-evaluation.md:331` (§8.1 table: "Phase-2 pytest timeout | 300 s (fixed on
scorer)") vs `docs/02-agent-engineering.md:111` and `:137` ("The 240 s command timeout is also used by the
scoring pytest run"; "**240 s timeout:** the scoring pytest run uses the same value").

**Evidence:** `harness_readme_raw.md:506` — "Verification pytest Timeout (Phase 2) … 300 sec — (fixed at
300 sec on scorer)". `docs/SOURCES.md` S38 reports that the scoring pytest run uses the same 240 s as the
agent's command timeout. These two sources conflict, and both docs state their side as fact.

**Fix:** pick one wording and use it in both files, e.g. "S5 (authoritative) fixes Phase-2 pytest at 300 s
(`S5:506`); a community report says the scoring pytest run uses the submission's 240 s command timeout
(S38). Unresolved — treat 300 s as the safer assumption and do not set `timeout_seconds` below the time a
full test file needs." Also add the conflict to `docs/05-risks-and-blockers.md` R-list, since it changes
how `timeout_seconds: 240` should be read.

---

### C4 — "Skill `.py` scripts run inside the sandbox" is unsourced

**Where:** `docs/01-harness-and-evaluation.md:25` (§1).

**Claim:** "Skills can contain `.py` scripts, but they run inside the sandbox via `run_skill_script`
(S5:130, S5:326)."

**Evidence:** S5:130 says only "`.py` (ADK skill scripts)" in the allowed-extension row. S5:326 says only
that `_wrap_skill_toolset_tools` coerces single-element lists for `load_skill`, `load_skill_resource`,
`run_skill_script`. Neither line says where the script executes. The README's security model (S5:57)
describes the *compiler* as not executing competitor Python; it says nothing about skill-script execution
at runtime. This is a security-relevant claim presented as fact.

**Fix:** replace with: "S5 permits `.py` for ADK skill scripts (`S5:130`) and the skill toolset exposes
`run_skill_script` (`S5:326`), but S5 does not state where the script executes. Treat skill scripts as
untrusted code with unknown host/network reach until verified; prefer prompts and YAML over scripts."

---

## 3. High

### H1 — Runbook wheelhouse install drops three required preprocessing steps

**Where:** `docs/04-runbook.md:136` (§1.2).

**Claim as written:** `pip install --no-deps --force-reinstall $WORK/wheelhouse/*.whl`.

**Evidence:** `getting_started_notebook.txt:57-84` does four things before installing:
(1) deletes broken `*cutlass*.pth` hooks from `dist-packages`/`site-packages` (`:58-66`);
(2) skips every wheel whose name contains `cutlass` (`:71-73`);
(3) restores the PEP 440 local tag that Kaggle strips — `cu128` → `+cu128` in the filename (`:74-78`);
(4) symlinks into `/tmp/wheelhouse`, falling back to copy (`:79-84`).
The install itself is `sys.executable -m pip install -q --no-deps --force-reinstall *wheels` (`:88-91`).

**Fix:** replace the one-liner with the notebook cell (or a script that reproduces all four steps). A bare
glob install does none of them, so the resulting environment will not match the notebook's — and the `.pth`
hooks the notebook deletes are described there as "broken" (`S6:57`). At minimum:
`find /usr/local/lib/python*/{dist,site}-packages -name '*cutlass*.pth' -delete` before install, and skip
`*cutlass*` wheels.

---

### H2 — Lifecycle step 16 contradicts the resolution criterion in the same document

**Where:** `docs/01-harness-and-evaluation.md:230` (§5, item 16: "`exit_code == 0` → resolved.") versus
`docs/01-harness-and-evaluation.md:§9.2` item 7.

**Evidence:** S5's flowchart says the score path is "`exit_code == 0 & JUnit XML valid`"
(`harness_readme_raw.md:52`); S5:565-567 makes resolution a conjunction of exit code 0 **and** the JUnit
check (`passed_tests > 0`, `failures == 0`, `errors == 0`, every required node passed and not skipped).
The sequence diagram's step 14 (`S5:230`) is the simplified version, but `01` presents it as the contract.

**Fix:** change item 16 to "`exit_code == 0` **and** JUnit XML valid (see §9.2) → resolved", so the
lifecycle list and §9.2 cannot be read as alternatives.

---

### H3 — Three `S28` citations have no support in the `S28` ledger entry

**Where:** `docs/02-agent-engineering.md:49`, `:53`, `:61`.

**Claims and what `docs/SOURCES.md` S28 actually contains:**
- `:49` "there is no per-agent budget in the schema (S28, S5:499-509)" — S28 is about compaction losing
  tool results, LoRA KV cache, double-encoded tool output, dropped thoughts, and `thinking_budget`/`seed`
  not being sent. It says nothing about per-agent budgets.
- `:53` "the public 0.18 analysis found 7 of 8 timeouts came from the first stage in a two-stage test
  (S28)" — no `SOURCES.md` entry and no other doc contains "7 of 8" or any two-stage timeout split
  (verified by `grep -rn "7 of 8" docs/` outside this review).
- `:61` "`exit_loop` is not in the tool registry (S28)" — the nine-tool list (`S5:341`) does establish
  that, but S28 does not.

**Fix:** for `:49`, cite `S5:499-509` and label the inference ("the schema exposes one `EvaluationBudget`;
no per-agent budget field is documented"). For `:53`, either re-source the 7-of-8 number to the notebook
it came from (and add it to that ledger entry) or delete the sentence — it currently carries a specific
statistic with no traceable origin. For `:61`, cite `S5:341` (tool registry) and mark the `LoopAgent`
exit behaviour as not documented in S5.

---

### H4 — The tool-call-grammar claim is cited to a source that does not contain it

**Where:** `docs/01-harness-and-evaluation.md:477` (§11 item 12) and `docs/02-agent-engineering.md:228`
(§4.3), plus `02:190` and `02:235`.

**Claim:** "Gemma 4 uses special tokens, not JSON; malformed quoting can swallow the next argument (S39)."
and "the parser error may only say a parameter is missing."

**Evidence:** `grep -n "special token\|malformed\|quoting\|swallow" docs/SOURCES.md` returns nothing. The
S39 ledger entry covers evaluator-version drift, 0.10/0.15 for the same package, the thinking sweep, the
12 h planning model, split sizes, and file-overlap statistics. S5 does document the special-token channel
format (`<|channel>thought…<channel|>`, `S5:152`) and the truncation nudge for an unclosed `<|tool_call>`
(`S5:335`), but not argument-swallowing.

**Fix:** split the claim. Keep what S5 supports: "tool calls use `<|tool_call>` … `<|tool_call|>` markers;
a truncated call is a documented failure mode (`S5:335`), so keep edits small." Re-label the
"malformed quoting swallows the next argument" part `[COMMUNITY]/[INFERENCE]` and, if kept, add the
observation to S39 (or a new S-number) with the exact quote.

---

### H5 — `03` describes the notebook's vLLM config as fixed values; it is computed

**Where:** `docs/03-post-training.md:576` (§9.1, "Use the starter notebook’s config (S6:272-290): …
`enable_lora=True`, `max_loras=1`, `max_lora_rank=16`").

**Evidence:** `getting_started_notebook.txt:282-284` sets `enable_lora=bool(adapter_map)`,
`max_loras=max(1, len(adapter_map))`, `max_lora_rank=max(detected_ranks) if detected_ranks else 128`.
Tensor parallel is also computed: `tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)`
(`:266-267`), not fixed at 4.

**Fix:** "With one rank-16 adapter present, the notebook resolves to `enable_lora=True`,
`max_loras=1`, `max_lora_rank=16` (`S6:282-284`); with no adapter it resolves to `enable_lora=False`,
`max_loras=1`, `max_lora_rank=128`. TP is `4/2/1` by visible GPU count (`S6:266-267`)." This matters for
the KV-cache numbers in §9.2, which assume the single-adapter case.

---

### H6 — Both 31B recipes train with the non-thinking template while §4.2 requires thinking

**Where:** `docs/03-post-training.md:371` (§7.1) and `docs/03-post-training.md:419` (§7.2), against
`docs/03-post-training.md:99` (§4.2) and `:518` (§7.3 note).

**Claim:** the two Axolotl recipes set `chat_template: gemma4`; §4.2's recommended path is "generate
trajectories with thinking enabled, include the thought tokens in the training context/labels, and run
inference with thinking enabled", and §7.3 correctly says "Use `gemma-4-thinking` template if training
with thoughts (S10)".

**Evidence:** S10 — "Use `gemma-4-thinking` chat template for 31B/26B; `gemma-4` for small models". The
recipes are 31B, so `gemma4` selects the wrong template for the recommended thinking-on path. This is the
exact training/inference mismatch that §4.2 and the blockers table (`03:§11`) warn about, baked into the
copy-paste config.

**Fix:** set `chat_template: gemma-4-thinking` in §7.1 and §7.2, keep `gemma4` only for the 12B/E4B recipe
in §7.4, and add the one-line check already recommended at `03:412` ("verify the exact token stream against
the tokenizer"). Also fix the §4.2 wording (see L8) so the two template names are described as a
size split, not a thinking toggle.

---

## 4. Medium

### M1 — `<3 GiB` vs the notebook's `<=` check, and an equality that cannot be strict

**Where:** `docs/01-harness-and-evaluation.md:93` (§2.4 table), `docs/04-runbook.md:585` (§9.2),
`docs/03-post-training.md:230`.

**Evidence:** S5:115/S5:120 say "< 3 GiB (3,221,225,472 bytes)". 3 GiB *is* 3,221,225,472 bytes, so
"< 3 GiB" and "= 3,221,225,472" cannot both be the threshold. The only enforcement visible locally is
`assert total_size <= MAX_SUBMISSION_SIZE_BYTES` (`getting_started_notebook.txt:607`), i.e. `<=`.

**Fix:** "≤ 3 GiB (3,221,225,472 bytes) per the notebook's `<=` check (`S6:607`); S5 writes it as
'< 3 GiB' (`S5:115,120`). Stay a few MB under either reading." Apply the same wording at `04:585`.

---

### M2 — `budget_warning` triggers on the configured budget, not on calls used

**Where:** `docs/01-harness-and-evaluation.md:293` (§7); repeated as guidance in
`docs/02-agent-engineering.md:§3.2` ("after 20 calls/≤10 remaining").

**Evidence:** `harness_readme_raw.md:348` — "If `budget.tool_calls >= 20` **and** `remaining_tool_calls
<= 10`". `budget.tool_calls` is the configured cap, not the count used. With the recommended 28-call
budget the warning therefore appears at call 18, not call 20.

**Fix:** "When `max_tool_calls >= 20` and ≤10 calls remain (with the 28-call baseline: from call 18),
every tool response carries `budget_warning` (`S5:348-350`)."

---

### M3 — `run_command` timeout formula is incomplete

**Where:** `docs/01-harness-and-evaluation.md:297` (§7 table: "timeout `min(300, remaining)`").

**Evidence:** `harness_readme_raw.md:358` — `min(command_timeout_seconds [300s],
max(5, int(remaining_time_seconds)))`. The 5-second floor is dropped.

**Fix:** "timeout `min(command_timeout_seconds, max(5, remaining_seconds))` (`S5:358`) — a command near the
end of the budget still gets 5 s."

---

### M4 — `eval_config.yaml` key list is missing `verification_timeout_seconds`

**Where:** `docs/01-harness-and-evaluation.md:67` (§2.2) and `docs/04-runbook.md:§2.1`.

**Evidence:** S5:85 lists only `timeout_seconds`, `max_tool_calls`, `max_time_minutes`, `max_turns`.
`getting_started_notebook.txt:344` additionally reads `verification_timeout_seconds` (default 300). So a
submission can set a key that S5 does not document, and the docs present the four-key list as complete.

**Fix:** add one row/line: "The starter notebook also honors `evaluation.verification_timeout_seconds`
(default 300, `S6:344`); S5 does not list this key, so its effect on the scorer is unverified. Leave it
unset unless you have measured a need."

---

### M5 — "All agents must declare the same base model" overstates the rule

**Where:** `docs/01-harness-and-evaluation.md:188` (§4.2); same wording in `README.md` §6 item 1.

**Evidence:** `harness_readme_raw.md:173` — "All agents in a submission must declare **at most ONE** unique
base model (`len(models) == 1`)". `harness_readme_raw.md:96` also allows `model` to be omitted when a
default is set.

**Fix:** "All agents that declare a model must resolve to the same one, and at most one unique base model
may appear in the tree (`S5:173`). `model` may be omitted only if a default is configured (`S5:96`), so
declare it explicitly."

---

### M6 — "Rank 128 is the maximum" is not enforced anywhere

**Where:** `docs/03-post-training.md:215` (§5.2).

**Evidence:** S5:153 and S6:284 both compute `max_lora_rank = max(detected_ranks) if detected_ranks else
128` — 128 is the *fallback*, not a clamp. S5:166 and S5:201 treat 128 as a planning ceiling. A submitted
adapter with rank > 128 therefore yields `max_lora_rank > 128`.

**Fix:** "128 is the planning ceiling and the fallback when no rank is detected (`S5:153`, `S6:284`);
nothing in S5/S6 clamps a higher submitted rank, so >128 is untested rather than forbidden."

---

### M7 — 61.4 GB BF16 is attributed to `S7`, which says 69.9 GB

**Where:** `docs/03-post-training.md:301` (§6.3: "BF16 weights are ~61.4 GB total (S7)") and `:43` (§2:
"31B BF16 weights are ~62 GB … (S7, S11)").

**Evidence:** S7's ledger entry gives 31B BF16 inference memory as **69.9 GB**. 61.4 GB is
30.7B parameters × 2 bytes, and the parameter count comes from S8 (not S7). S11 contains no weight-size
number.

**Fix:** "31B is 30.7B parameters (`S8`), so BF16 weights are ~61.4 GB; S7's 69.9 GB figure is inference
memory including overhead. Use ~62 GB for the weights and plan the larger number for the card." Mark the
derivation `[ESTIMATE]`. Drop `S11` from the `:43` citation.

---

### M8 — The manual-serve → `swegemma eval` bridge is missing

**Where:** `docs/04-runbook.md:252-289` (§3.2–§3.3).

**Problem:** three gaps that make the sequence fail or silently miss the adapter:
1. `vllm serve /path/to/model` registers the served model name as the *path*. The harness binds
   `openai/<declared_model>` (`S5:187`), so `openai/gemma-4-31b-it-qat-w4a16-ct` will not match. No
   `--served-model-name` is passed.
2. `swegemma eval` is invoked with no `--models-yaml` and no endpoint flag; S5:598-605 lists no flag that
   points the CLI at an already-running server, so it is unstated whether it reuses `:8000` or starts its
   own.
3. The 12B `--hf-overrides` requirement (`S6:236-241`) is not mentioned, so the command fails or loads the
   wrong architecture if the reader swaps models.

**Fix:** add `--served-model-name <declared_model>` to the serve command; add a sentence "the CLI either
starts its own server or uses `--models-yaml` — verify which before assuming this one is reused"; add the
`--hf-overrides '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'` line with the note
that only the 12B needs it (`S6:238-239`).

---

### M9 — `--sandbox docker` is used without building `swebench-sandbox:latest`

**Where:** `docs/04-runbook.md:283` (§3.3) and `docs/04-runbook.md:362` (§5).

**Evidence:** S5:234 names the image `swebench-sandbox:latest`; S2 records a `docker/` directory of sandbox
build files. The runbook downloads the data (§1.1) but never builds or pulls the image before running the
Docker backend.

**Fix:** add a step before §3.3: `docker build -t swebench-sandbox:latest $DATA/docker` (adjust to the
actual Dockerfile path in the dataset) and `docker image ls | grep swebench-sandbox`, or change the smoke
test to `--sandbox subprocess` and say why.

---

### M10 — "36 calls worse than 28 (S39)" is not in the ledger

**Where:** `docs/02-agent-engineering.md:183` (§3.2).

**Evidence:** the S39 entry records the 22-task thinking sweep (5/22, 7/22, 15/22), the 0.10/0.15 repeat,
the 12 h planning model, and split sizes. It contains no 36-call comparison.

**Fix:** re-source the number to the post that reported it and add it to S39, or delete the clause and keep
"32–40 calls increase loops/time; measure it".

---

### M11 — "~3,700 characters (S36)" is not in the ledger

**Where:** `docs/02-agent-engineering.md:224` (and `:222`).

**Evidence:** the S36 entry lists the config values, the nine declared tools, and the prompt rules, but no
prompt length.

**Fix:** label it `[ESTIMATE]` ("the 0.24 prompt reads as roughly 3–4k characters") or add the measured
length to S36.

---

### M12 — "12,602 tokens (S25)" is not in the ledger

**Where:** `docs/03-post-training.md:123` (§4.4).

**Evidence:** the S25 entry records 697/99 + 60/11 examples, the adapter shape, the alpha sweep, and the
thinking mismatch — no token-count verification.

**Fix:** add the 12,602-token check to S25 with the exact claim, or drop the number and keep the procedure
("verify the patched template reproduces the stock token stream exactly").

---

### M13 — Unverified model id and stale TRL argument in the Unsloth sketch

**Where:** `docs/03-post-training.md:464-465` and `:499` (§7.3).

**Evidence:** `model_name="unsloth/gemma-4-31B-it"` appears in no ledger entry (S10 documents the Unsloth
API surface, not that repo id). `SFTConfig(max_seq_length=8192)` is the pre-rename TRL argument
(`max_length` in current TRL), and `dataset_text_field="text"` contradicts §4.3–§4.4, which build
per-call tokenized samples with `input_ids`/`labels` and no `text` column. The sketch also passes
`dataset=dataset` with `dataset` undefined.

**Fix:** mark the repo id "verify on the Hub"; use `max_length=` (or pin the TRL version); and either show
a `"text"`-column dataset or pass the pre-tokenized dataset with `dataset_text_field=None`. Add
`dataset = ...` or state that the variable comes from §6.3.

---

### M14 — "Kaggle auto-selects" is cited to `S1`, which does not record it

**Where:** `docs/04-runbook.md:631` (§10.4).

**Evidence:** the S1 entry records dates, prizes, the 12 h limit, allowed models, tools, and machine specs.
It says nothing about auto-selection when fewer than two final submissions are chosen. S3 says "up to two
final submissions selected for judging".

**Fix:** either verify the auto-selection behaviour on the submissions page and add it to S1, or delete the
sentence and keep "select two; if you select one, that is your entry."

---

### M15 — A literal `...` inside the trajectory-generation loop

**Where:** `docs/04-runbook.md:405` (§6.1).

**Claim as written:** `swegemma eval ... --submission-dir $WORK/agents/t$T \ --results-dir $RUNS/gen-t$T
--concurrency 2`.

**Fix:** expand the full flag set (copy §3.3) or define a shell function once, e.g.
`eval_run() { swegemma eval --tasks $DATA/tasks.jsonl --snapshots-dir $DATA/snapshots "$@"; }` and call
`eval_run --submission-dir … --results-dir … --concurrency 2`. As written it cannot run.

---

### M16 — `/kaggle/working/submission` is presented as the extraction target

**Where:** `docs/01-harness-and-evaluation.md:63` (§2.1).

**Evidence:** S5:166 says the size limit "Ensures rapid extraction into `/kaggle/working/submission`". It
does not state that the platform extracts there, nor that the path is a requirement on the submitter.

**Fix:** "S5:166 mentions extraction into `/kaggle/working/submission` as the reason for the size limit; it
does not specify the platform's extraction path. This does not affect how you build the zip."

---

### M17 — "Continue from `end_line`" re-reads the last line

**Where:** `docs/02-agent-engineering.md:300` (§5.2).

**Evidence:** `read_file` is 1-indexed **inclusive**, and `end_line` is "the last complete line returned"
(`S5:401`). Continuing at `end_line` duplicates one line; the next unread line is `end_line + 1`.

**Fix:** "If truncated, continue with `start_line=end_line + 1`."

---

## 5. Low

**L1 — Skills tree adds entries not in the cited source.** `docs/01-harness-and-evaluation.md:52-56` shows
`skills/repo_navigation/{SKILL.md, scripts/, resources/}` cited to `S5:63-82`; S5's tree shows only
`SKILL.md` (`harness_readme_raw.md:80-82`). *Fix:* cite the ADK skill layout for `scripts/`/`resources/`,
or drop them and note that `.py` is allowed under a skill directory (`S5:130`).

**L2 — Recommended sampling deviates from the model card without a note.** `docs/01:§3.4` and
`docs/02:§2.1` recommend `temperature: 0.2`, `top_p: 0.95`; S8 recommends `temperature=1.0`,
`top_p=0.95`, `top_k=64`. No doc mentions the official recommendation (grep of `docs/*.md` finds it only
in `SOURCES.md:115`). *Fix:* one sentence: "This deliberately departs from the model card's sampling
(`S8`); 0.2 is an empirical choice from the 0.24 config, not a vendor recommendation."

**L3 — The `pytest.ini`/`conftest.py` warning drops a qualifier.** `docs/01:246` says "Do not edit or
delete them"; S5:268 says "unless required by the task". *Fix:* restore the qualifier, since some tasks
genuinely need a conftest change and the reader should know the harness strips it either way.

**L4 — 256-d embeddings scoped too widely.** `docs/02:§5.6` says graphs and "256-d embeddings" exist "for
public and private tasks (S2, S5:443)". S2 establishes 256-d for the public set; S33 establishes only that
private graphs exist. *Fix:* "public tasks ship 256-d embeddings (`S2`); private tasks have graphs (`S33`),
dimensionality unstated."

**L5 — `wheels/` role cited to the wrong source.** `docs/04:§1.2` attributes "the competition `wheels/`
directory is for sandbox dependencies" to `S6:55`; S6:55 is only the wheelhouse path. *Fix:* cite S2
(`wheels/: 124 pre-compiled offline wheels`) alongside S6:55.

**L6 — `--max_steps 10` as an Axolotl CLI override.** `docs/04:§7.1` shows
`accelerate launch -m axolotl.cli.train config.yml --max_steps 10`. No ledger entry documents that flag.
*Fix:* put `max_steps: 10` in the YAML and run the plain launcher; keep the "check the current launcher"
hedge.

**L7 — "no way to modify the test runner".** `docs/01:5`. An agent can edit `conftest.py`/`pytest.ini`; the
harness strips the hunks and resets the paths (`S5:535,549`). *Fix:* "no way to make test-runner edits
stick."

**L8 — `gemma-4` described as the "non-thinking variant".** `docs/03:§4.2` says Gemma 4 has "a thinking
variant (`gemma-4-thinking`) and a non-thinking variant (S10, S8)". S10 gives `gemma-4-thinking` for
31B/26B and `gemma-4` for small models — a size split, not a thinking toggle. *Fix:* "`gemma-4-thinking`
for 31B/26B and `gemma-4` for the small models (`S10`); neither name implies thoughts are disabled."

**L9 — "so `agent.yaml` is the first path".** `docs/04:§2.3` (and the same phrasing at `docs/01:§2.1`).
Info-ZIP entry order follows directory traversal; nothing requires `agent.yaml` to be the first entry, only
that it is at the root. *Fix:* "so `agent.yaml` is at the archive root (check with `unzip -l`)", and drop
"first".

**L10 — Top-level eval keys are accepted but undocumented.** `docs/01:67` says the block is "under
`evaluation:`". `getting_started_notebook.txt:341` is
`raw_eval_cfg.get('evaluation', raw_eval_cfg)`, so a flat file also works locally. *Fix:* add "the starter
notebook also accepts the keys at the top level (`S6:341`); use the `evaluation:` block because that is
what S5 documents."

**L11 — The "Collect" snippet reads the wrong path.** `docs/04:375` runs
`p=pathlib.Path("summary.json")`, but the run wrote `$RUNS/baseline-<UTC>/summary.json`. *Fix:* use
`pathlib.Path(os.environ["RUNS"]) / "<run-dir>" / "summary.json"` or `cd` into the run directory first.

---

## 6. Line-reference audit (`S5` / `S6`)

Checked by locating each cited string with `grep -n` and printing the surrounding lines. Result: the line
numbers themselves are sound; the defects are in what is *attributed* to those lines or to the `S`-number.

**Verified correct (sample of the ranges used across `01`–`04`):**

| Reference | Used for | Actual content at that line |
|---|---|---|
| `S5:21-23` | three-library table | 21/22/23 = `adk-submission`/`adk-eval-core`/`swegemma` rows (incl. `SessionTrace` on 22) |
| `S5:57` | declarative-only | "Competitors do not submit Python code entrypoints…" |
| `S5:61` | zip root | "packaged at the root of `submission.zip`" |
| `S5:84-89` | root config, `eval_config`, `!include` | matches |
| `S5:115,120` | 3 GiB | "< 3 GiB (3,221,225,472 bytes)"; table row |
| `S5:121-131` | limits + extensions | matches row-for-row |
| `S5:132-139` | generation constraints | matches |
| `S5:144-160` | L4 hardware, vLLM, model table | matches |
| `S5:164-173` | four rules, single-model validation | matches |
| `S5:187-201` | serving, LoRA, sizing | matches |
| `S5:204-230` | two-phase lifecycle | 14-step diagram spans 212-230 |
| `S5:233-242` | sandbox limits | matches |
| `S5:243-268` | bootstrap | 8 bullets under a "7 steps" heading (S5's own inconsistency) |
| `S5:273-278` | session state | matches |
| `S5:282-319` | prompt sections | matches |
| `S5:325-326` | tool error recovery | matches |
| `S5:328-338` | termination/nudges | matches |
| `S5:341-350` | tools + envelopes + budget warning | matches (348 = the `>= 20` condition) |
| `S5:353-486` | all nine tool entries | each range lands on the named tool |
| `S5:394` | path handling | matches |
| `S5:499-511` | budgets + timer tip | matches (511 = the `[!TIP]`) |
| `S5:518-525` | compaction/cache/retries | 518-521 compaction, 523 cache, 525 retries |
| `S5:529-537` | extraction | 531-532 commands, 535 stripping, 536 fallback, 537 recording |
| `S5:540-567` | Phase 2 | 543-547 four passes, 548 failure, 549-553 reset, 559-562 pytest, 565-567 criterion |
| `S5:584-620` | CLI + artifacts | 584-593 invocation, 598-605 flags, 608-620 artifacts |
| `S5:624-627` | gotchas | 624 pytest.ini, 625 scratch, 626 agent_tool, 627 submit last |
| `S6:20,280,287` | notebook vLLM env/config | `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`, `gpu_memory_utilization`, `startup_timeout` |
| `S6:266-267,282-284` | TP + LoRA sizing | computed values (see H5) |
| `S6:491-516` | starter run | 491 `BudgetExceeded (10 calls)`, 507-509 dashboard, 516 result line |
| `S6:600-611` | packaging | 600-601 `make_archive`, 606-607 size `<=`, 608-611 extensions |

**Line references that are correct but support less than the doc claims:** `S5:130` (extension list only —
see C4), `S5:166` (rationale sentence only — see M16), `S5:326` (argument coercion only — see C4),
`S5:443` (preconditions only — see L4), `S5:506` (300 s — see C3).

**Wrong `S`-number (line number inside the source may be fine):** `01:348` (S38 should be S37 for the
"not a guarantee" statement), `02:111` (same), `02:49/53/61` (S28), `01:477` + `02:228/190/235` (S39),
`02:183` (S39), `02:224` (S36), `03:123` (S25), `03:301` + `03:43` (S7, and S11 for a weight-size claim),
`04:§1.2` (S6:55 where S2 is the source), `04:631` (S1).

**One factual note on the sources themselves:** `harness_readme_raw.md` has no trailing newline, so
`wc -l` reports 626 while the file has 627 logical lines (SOURCES/fact sheet say 627 — correct).
`getting_started_notebook.txt` has 615 logical lines, matching both fact sheets.

---

## 7. Numbers re-derived and confirmed

These were recomputed rather than assumed; no finding.

| Claim | Check |
|---|---|
| 96 GB VRAM; `0.90` → 21.6 GB/GPU, 86.4 GB total | 24×4=96; 24×0.9=21.6; ×4=86.4 |
| 3 GiB = 3,221,225,472 bytes | 3×1024³ exact (boundary issue only, M1) |
| 5.5 min × 120 = 11 h; 4.5 × 120 = 9 h; 8 × 120 = 16 h | correct |
| 10,000 chars ≈ 2,500 tokens; 3,700 chars ≈ 900–1,000 tokens | ~4 chars/token |
| Adapter 61.4 GB → 15.4 GB/GPU over 4 cards | 61.4/4=15.35 |
| Adapter size bands 110–220 / 220–450 / 450–900 MB / 0.9–1.8 GB | match `S5:198-201` |
| 410 modules, 122.4M params, 245 MB at r=16 | match `S25` |
| 27,712 KV tokens at util 0.80 | match `S27` |
| 5/22, 7/22, 15/22 + 0.05 | match `S39` |
| 30.9 → 32.2 SWE-bench Verified | match `S18` |
| 697 examples / 99 tasks; 60 / 11 held out | match `S25` |
| 43/129 and 52/129 plain, 127/129 calibrated | match `S31`/`S32` (range in `04:§4.1` is fair) |
| Workstation: no GPU, 15 GB RAM, 57 GB free | confirmed on this machine (`nvidia-smi` absent, `free -g` 15, `/home` 57 G) |

---

## 8. Claims that could not be checked offline

Not scored as findings; listed so the main agent knows the residual risk.

- Anything behind `S23`–`S40` URLs (community notebooks, discussions, blogs) — only the ledger summaries
  were available, so ledger *omissions* (M10–M12, H3, H4) may be "true but unledgered" rather than false.
- `swegemma`/`adk-submission` source: `validate_directory`'s accepted path type (`Path` vs `str`),
  `VllmConfig` defaults (`reasoning_config`), whether the CLI reuses an existing `:8000` endpoint, and
  `MAX_SUBMISSION_SIZE_BYTES`'s literal value — none of these packages exist in this workspace. For C1 the
  notebook establishes the 2-value unpacking (`S6:352`); only the exact declared type is unseen.
- Whether the scorer's Phase-2 timeout is 300 s or 240 s (C3); only S5 and S38 are available, and they
  disagree.
