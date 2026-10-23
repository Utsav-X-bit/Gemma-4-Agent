# Runbook executability review — final consistency pass

Scope: `docs/04-runbook.md` (777 lines), `docs/03-post-training.md` §7 (363–531), checked against
`docs/_research/notebook-facts.md`, `getting_started_notebook.txt`, `harness_readme_raw.md`,
`docs/SOURCES.md`, and the two prior reviews (`harness-audit.md`, `strategy-audit.md`).
Read-only: no doc was edited. No network access, so Axolotl/TRL/Unsloth API facts are marked
`[VERIFY]` where no local source establishes them.

Method: every command block was executed mentally against the shell state the doc has created at that
point (cwd, defined variables, files actually written by earlier steps); every `swegemma eval` flag and
every imported symbol was located in `harness_readme_raw.md` / `getting_started_notebook.txt`; every
snippet was checked for undefined names and for a cwd it never sets.

**Verdict: 1 critical, 10 high, 14 medium, 3 low.** The `build_submission_limits()` fix (prior C1) is
correct in both places, and most prior findings landed. The fixes themselves introduced four new
defects (H2, H5, H10, and the §9 relative-path cluster C1), all listed below.

---

## 1. Severity summary

| ID | Sev | Where | One-line |
|---|---|---|---|
| C1 | Critical | `04:648-666`, `04:683` | §9 mixes three cwds; zip validation and `kaggle competitions submit` both open a path that does not exist |
| H1 | High | `04:162-166` | fresh venv + `--no-deps`: the harness import can only succeed if the wheelhouse is a superset of the Kaggle image |
| H2 | High | `04:150` vs `04:162` | cutlass `.pth` cleanup targets `/usr/local/lib/...`, but the install target is `$WORK/harness-venv` — the step is inert |
| H3 | High | `04:566-567` | literal `outputs/...` in `ls`/`du`; cannot run |
| H4 | High | `03:371`, `03:419` | `chat_template: gemma-4-thinking` is a model template name, not an Axolotl `chat_template` value |
| H5 | High | `03:523` (§7.4) | "use the same recipe" now propagates `gemma-4-thinking` + the 31B target-module regex to 12B/E4B |
| H6 | High | `03:495` | `train_dataset=dataset` — `dataset` is never defined anywhere in the doc |
| H7 | High | `03:497` | `dataset_text_field="text"` contradicts the pre-tokenized `input_ids`/`labels` samples of §6.3 |
| H8 | High | `03:497-515` | `SFTConfig` sets no `max_length`; TRL default truncates to 1024 while the recipe claims 8192 |
| H9 | High | `04:91` | `hf download` repo ID unverified, no `--revision`, casing differs from `ALLOWED_MODEL_NAMES` |
| H10 | High | `04:315` | `-f $DATA/docker/Dockerfile.sandbox` — no source names this file |
| M1 | Medium | `04:13-17`, `254`, `469`, `648` | `$RUNS`, `$WORK/agents`, `$AGENT` never created; §9 silently switches from `$AGENT` to `submission` |
| M2 | Medium | `04:383`, `412` | `open("tasks.jsonl")` relative; `holdout.txt` is never written |
| M3 | Medium | `04:424` | `$RUNS/summary.json` — the run wrote `$RUNS/baseline-<UTC>/summary.json` |
| M4 | Medium | `04:462-467` | prose says five temperatures, loop runs four, list disagrees with both |
| M5 | Medium | `04:289-303` | `/path/to` model path, no `--served-model-name`, no `--reasoning-config`, bare flag fragment |
| M6 | Medium | `04:531` | `--max_steps 10` CLI override still shown as the primary command |
| M7 | Medium | `04:578` | `p="adapters/main_lora"` relative to a cwd that is not `$AGENT` |
| M8 | Medium | `04:658` vs `04:668` | prose "stay below the constant" vs code `<=` |
| M9 | Medium | `03:393`, `03:440` vs `03:504` | two names for 8-bit Adam in one document |
| M10 | Medium | `03:494` | `tokenizer=` is the pre-rename TRL argument |
| M11 | Medium | `03:370`, `03:418` | `base_model: google/gemma-4-31B` casing/ID not in the registered alias list |
| M12 | Medium | `04:285-297` | `vllm serve` never activates `$WORK/harness-venv`, which §8.1 then requires |
| M13 | Medium | `03:402` | `activation_offloading: true` in the single-GPU recipe |
| M14 | Medium | `04:259` | "no symlinks" cannot be checked with `unzip -l` |
| L1 | Low | `04:157` | bash-only `${name/cu128/+cu128}`; empty-glob install |
| L2 | Low | `04:111` | expected dataset listing omits `submission.parquet` (S2) |
| L3 | Low | `04:303` | LoRA flag fragment has no `vllm serve` and a path that disagrees with §7.3 |

---

## 2. Critical

### C1 — §9 cannot be run end to end: three cwds, one of which has the file

**Where:** `04:648` (`cd submission`), `04:649` (`zip -r -X ../submission.zip .`), `04:666`
(`zf=zipfile.ZipFile("submission.zip")`), `04:683` (`kaggle competitions submit ... -f submission.zip`).

**What happens as written:**

1. `04:648-649` leaves the shell in `$WORK/submission` and writes `$WORK/submission.zip`.
2. `04:666` then opens `submission.zip` **relative to `$WORK/submission`**, i.e.
   `$WORK/submission/submission.zip`, which was never created → `FileNotFoundError`. The size/extension
   gate silently becomes a no-op that looks like a pass if the reader skips the traceback.
3. `04:683` submits `submission.zip` from the same cwd → `kaggle` reports the file does not exist.

The three blocks use three different bases (`submission`, `submission.zip`, `../submission.zip`) and the
doc never states the working directory for §9, while §2.3 used `cd $AGENT`.

**Fix:** make §9 absolute and consistent:

```bash
cd $AGENT                     # 9.2
zip -r -X $WORK/submission.zip .
unzip -l $WORK/submission.zip
```

```python
zf=zipfile.ZipFile(os.environ["WORK"] + "/submission.zip")   # or Path(os.environ["WORK"])/"submission.zip"
```

```bash
kaggle competitions submit -c gemma-4-developer-agent -f $WORK/submission.zip -m "..."
```

and change `04:634` from `agent=Path("submission")` to `agent=Path(os.environ["AGENT"])` so §9.1 has the
same base as everything else.

---

## 3. High

### H1 — Fresh venv plus `--no-deps` is not the notebook's starting state

**Where:** `04:162-166`.

**Claim as written:** `python3 -m venv $WORK/harness-venv` … `pip install --no-deps --force-reinstall
/tmp/wheelhouse/*.whl` … `python -c "import swegemma, adk_submission"`.

**Evidence:** the notebook installs the same 41 wheels **into Kaggle's pre-populated image**, where every
transitive dependency (yaml, openai/litellm, ADK, numpy, …) already exists (`notebook-facts.md` §2:
"`--no-deps` means the wheelhouse must be self-contained" — self-contained *relative to that image*).
`getting_started_notebook.txt:88-91` is `sys.executable -m pip install … --no-deps` in that image, not in
an empty venv. A fresh venv contains only pip/setuptools, so `--no-deps` guarantees that any wheelhouse
gap becomes an `ImportError` at `04:166`, and the doc's own remedy ("If import fails, inspect the wheel
names", `04:171`) misattributes the cause to the wheels.

**Fix:** either mirror the notebook's starting state —

```bash
python3 -m venv --system-site-packages $WORK/harness-venv
```

— or drop `--no-deps` for this one install (`pip install --force-reinstall /tmp/wheelhouse/*.whl`), and
add one line: "`--no-deps` is only safe on top of an environment that already has the harness
dependencies; a bare venv needs resolution."

### H2 — The cutlass `.pth` cleanup does not touch the environment being installed into

**Where:** `04:150` (`find /usr/local/lib/python*/…`) versus `04:162` (`python3 -m venv $WORK/harness-venv`).

**Evidence:** the notebook deletes the hooks from the *same* interpreter that then runs the install
(`getting_started_notebook.txt:58-66` then `:88-91`). Here the deletion targets the system interpreter and
the install targets a venv created eight lines later; a venv never reads `/usr/local/lib/python*/dist-packages`
`.pth` files unless created with `--system-site-packages`. So step 1 is a no-op for the install that
follows — the fix for prior H1 reproduced the command but not its scope.

Secondary: GNU `find` warns and applies `-maxdepth` globally when it appears after the path arguments
("you have specified the global option -maxdepth after the argument …"), which reads like an error to a
reader who is told to stop when output does not match.

**Fix:** create the venv first, then scope the delete to it:

```bash
python3 -m venv $WORK/harness-venv
source $WORK/harness-venv/bin/activate
find "$VIRTUAL_ENV/lib"/python*/site-packages -maxdepth 1 -name '*cutlass*.pth' -delete
```

(keep the `/usr/local/lib/...` variant only in a comment, labelled "Kaggle-only, matches S6:58-66").

### H3 — Literal `outputs/...` in §7.3

**Where:** `04:566-567`.

```bash
ls outputs/.../adapter_config.json outputs/.../adapter_model.safetensors
du -sh outputs/.../
```

`...` is a real (empty) directory name, so both commands exit non-zero with "No such file or directory".
This is the one remaining literal ellipsis in a command block (prior M15 fixed §6.1 only), and the export
step is where a wrong path silently ships the wrong adapter.

**Fix:** use the `output_dir` the recipes actually set — `03:379` `./outputs/gemma4-31b-qlora`,
`03:427` `./outputs/gemma4-31b-lora-fsdp`:

```bash
OUT=${1:-outputs/gemma4-31b-qlora}
ls "$OUT"/adapter_config.json "$OUT"/adapter_model.safetensors
du -sh "$OUT"
```

### H4 — `chat_template: gemma-4-thinking` is not an Axolotl template value

**Where:** `03:371`, `03:419`.

**Evidence:** `gemma-4-thinking` is the *model's* chat-template variant (S10, quoted in §4.2 and §7.3's
note). Axolotl's `chat_template` field is trainer-side: it is resolved by name against Axolotl's own
registry, or used verbatim as Jinja if it contains `{{`/`{%`. `gemma-4-thinking` is in neither set, so
the config fails at load with an unknown-template error. The prior fix (H6 in `harness-audit.md`) moved
the name from `gemma4` to `gemma-4-thinking` to satisfy §4.2 but never checked the trainer's accepted
values — that is a regression, not a fix. The inline comment "verify the trainer accepts this name" is
the only guard, and it sits in the one line that must be correct.

**Fix:** keep the intent (thinking-matched tokens) but express it with a value the trainer resolves:

```yaml
chat_template: tokenizer_default   # uses google/gemma-4-31b-it's own template (thinking variant)
```

or paste the tokenizer's Jinja (`tokenizer.chat_template`) into the field, and change the note at
`03:411` to: "`gemma-4-thinking` is the model's template variant, not an Axolotl template name; set
`chat_template: tokenizer_default` (or the literal Jinja) and assert `<|channel>thought` appears in the
rendered sample." `[VERIFY]` the exact accepted names against the installed Axolotl version.

### H5 — §7.4's "same recipe" now inherits 31B-only values

**Where:** `03:523` ("Use the same recipe with the appropriate base") against `03:371-403` and `03:419-461`.

**Evidence:** with the H6 fix in place, "the same recipe" now carries `chat_template: gemma-4-thinking`,
`eot_tokens: ["<turn|>"]`, the target-module regex
`model.language_model.layers.[\d]+...(_checkpoint_wrapped_module.)?(mlp|self_attn)...`, and
`transformer_layer_cls_to_wrap: Gemma4TextDecoderLayer`. S10 assigns `gemma-4` (not `gemma-4-thinking`) to
the small models, and the E4B checkpoint is a different parameter layout, so the regex and wrap class are
not transferable. Before the H6 fix, §7.4 inherited `gemma4` and was accidentally closer to S10.

**Fix:** replace `03:523` with an explicit substitution table:

| Field | 31B (§7.1/§7.2) | 12B / E4B (§7.4) |
|---|---|---|
| `base_model` | `google/gemma-4-31b-it` | official 12B / E4B ID `[VERIFY]` |
| `chat_template` | `tokenizer_default` (thinking) | `tokenizer_default` (S10: `gemma-4` for small models) |
| `lora_target_modules` | 31B regex | re-derive from `model.named_modules()` |
| `transformer_layer_cls_to_wrap` | `Gemma4TextDecoderLayer` | verify per checkpoint |
| `sequence_len` | 8192 | 8192 |

### H6 — `dataset` is undefined in the Unsloth snippet

**Where:** `03:495` (`train_dataset=dataset`).

Nothing in §6.3/§6.4 or §7.3 assigns `dataset`; §6.3 only says to save `input_ids`,
`attention_mask`, `labels`. Prior M13 flagged this and it is still present.

**Fix:** insert before `trainer = SFTTrainer(`:

```python
from datasets import load_from_disk
dataset = load_from_disk("/path/to/samples")   # the §6.3 output, columns: input_ids, attention_mask, labels
```

### H7 — `dataset_text_field="text"` contradicts the data §6.3 produces

**Where:** `03:497` versus `04:493-494` (§6.3 steps 4–5) and `03:129-137` (§4.4 assistant-only loss).

**Evidence:** §6.3 emits pre-tokenized `input_ids`/`attention_mask`/`labels` with `-100` masking; there is
no `text` column. `SFTConfig(dataset_text_field="text")` makes `SFTTrainer` look for that column and fail
(or, with `max_length` unset, re-tokenize something else). It also silently discards the §4.4 masking.

**Fix:** for the pre-tokenized path use `dataset_text_field=None` (or omit it) and
`dataset_kwargs={"skip_prepare_dataset": True}`, with a one-line note that this is the path that keeps
`labels`; keep `"text"` only in a clearly-labelled alternative snippet for the plain-text fallback.

### H8 — No `max_length` in `SFTConfig`: silent 1024-token truncation

**Where:** `03:497-515`.

`FastModel.from_pretrained(max_seq_length=8192)` (`03:475`) sets the model/tokenizer ceiling, but the
trainer's own truncation length is a separate `SFTConfig` field (`max_length` in current TRL,
`max_seq_length` before the rename). With neither set, TRL's default (1024) applies, so the run trains on
the first 1k tokens of 8k samples — the failure is silent: loss looks fine, the adapter learns truncated
agent turns, and §7.2's "loss 1–3" monitor never notices.

**Fix:** add to `SFTConfig`:

```python
max_length=8192,          # TRL >= 0.20; older: max_seq_length=8192
```

and extend the `03:467` caveat to name the consequence ("an unset truncation length silently caps
training at 1024 tokens"), since that caveat currently only mentions argument *names*.

### H9 — `hf download` ID is unverified and unpinned

**Where:** `04:91`.

```bash
hf download google/gemma-4-31B-it-qat-w4a16-ct --local-dir $WORK/models/gemma-4-31b-ct
```

Three problems: (a) the ID is not established by any local source — S2/S5 establish only the *served*
name `gemma-4-31b-it-qat-w4a16-ct` (lowercase, no `google/` prefix), and §3 (`04:271`) tells the reader to
"use the exact model name and revision" without giving one; (b) no `--revision`, although the same step
says "Record the model revision" and §6.4/§7 freeze a revision in the manifest — an unpinned download
makes that record impossible; (c) the casing (`31B`) differs from every other occurrence in the docs, and
HF repo IDs are case-sensitive.

**Fix:**

```bash
export GEMMA4_REV=<commit-sha>          # from the model card; [VERIFY repo ID]
hf download google/gemma-4-31b-it-qat-w4a16-ct --revision "$GEMMA4_REV" \
  --local-dir $WORK/models/gemma-4-31b-ct
```

plus `[VERIFY repo ID]` on the line and a note that the gated repo may live under a different owner.

### H10 — `Dockerfile.sandbox` is invented

**Where:** `04:315`.

**Evidence:** S2 records only "`docker/`: sandbox build files" (`SOURCES.md:45`); S5:234 records only the
image name `swebench-sandbox:latest`. No local source names a Dockerfile, and the runbook's own §1.1
listing (`04:111`) shows the directory without its contents. `docker build -f` on a non-existent file
fails immediately, which is exactly the "missing image" wall §3.2b says not to hit.

**Fix:** make the step discover the file instead of guessing:

```bash
ls -la $DATA/docker
DOCKERFILE=$(ls $DATA/docker/Dockerfile* 2>/dev/null | head -1)
docker build -t swebench-sandbox:latest -f "$DOCKERFILE" "$(dirname "$DOCKERFILE")"
```

and add the file names actually observed to `04:111`'s expected listing.

---

## 4. Medium

### M1 — Directories the doc uses but never creates, and the `$AGENT` → `submission` switch

**Where:** `04:13-17` (exports), `04:52` (only `mkdir -p $WORK/env`), `04:254` (`cd $AGENT`), `04:469`
(`cp -r $AGENT $WORK/agents/t$T`), `04:648` (`cd submission`).

`$RUNS` (`$WORK/runs`), `$AGENT` (`$WORK/submission`) and `$WORK/agents` are never created. `04:469`
fails with "No such file or directory" because `cp -r` cannot create the intermediate `$WORK/agents`;
`$AGENT` relies on the reader interpreting "Create:" at `04:179` as a command; `$RUNS` is only safe if
`swegemma eval` creates `--results-dir` (unverified). §9 then drops `$AGENT` for a bare `submission`.

**Fix:** after the export block at `04:17`:

```bash
mkdir -p $DATA $RUNS $AGENT $WORK/agents $WORK/models $WORK/env
```

and change `04:648` to `cd $AGENT`.

### M2 — Holdout list is printed, never written; path is relative

**Where:** `04:383` (`open("tasks.jsonl")`), `04:412` (`--task-ids $(cat holdout.txt | tr '\n' ' ')`).

`holdout.txt` is never created by the §4.3 snippet (it prints to stdout and says "Save the list"), so
`$(cat holdout.txt …)` expands to nothing and `--task-ids` gets no value → argparse error. The snippet
also depends on the cwd being `$DATA`, set only transiently at `04:107`.

**Fix:** end the snippet with

```python
open(os.environ["DATA"] + "/holdout.txt", "w").write("\n".join(hold) + "\n")
```

and use `rows=[json.loads(l) for l in open(os.environ["DATA"] + "/tasks.jsonl")]`. Note: `--task-ids`
takes multiple ids as separate words (`S5:598`), so the unquoted substitution is correct — only the file
is missing.

### M3 — Baseline summary path is one level short

**Where:** `04:424` (`p=pathlib.Path(os.environ["RUNS"]) / "summary.json"  # adjust to the actual run dir`).

The run wrote `$RUNS/baseline-<UTC>/summary.json` (`04:410`), and `--results-dir` artifacts are per-run
(`S5:608-620`). Prior L11 was only half-applied: the env var is now used, the run subdirectory is still
missing, and the "# adjust" comment moves the work back to the reader.

**Fix:**

```python
run = sorted(pathlib.Path(os.environ["RUNS"]).glob("baseline-*"))[-1]
s = json.loads((run / "summary.json").read_text())
```

### M4 — "Five temperatures", a four-iteration loop, and a third list

**Where:** `04:462` ("129 tasks × 5 temperatures ≈ 645"), `04:464` ("For each of five temperatures …
e.g. 0.2, 0.4, 0.6, 0.8, 1.0"), `04:467` (`for T in 0.2 0.5 0.8 1.0`).

The arithmetic is built on 5 (645 = 129×5; 258 = 129×2), the prose names five values, and the loop runs
four — including `0.5`, which appears in neither list. The `--results-dir $RUNS/gen-t$T` dirs will
therefore not match the stated run count.

**Fix:** pick one list and use it in all three places, e.g. `for T in 0.2 0.4 0.6 0.8 1.0; do` and keep
`[ESTIMATE] 645`; or change the prose to four and recompute to 516.

### M5 — Manual `vllm serve` still cannot be reached from `swegemma eval`

**Where:** `04:289-303`.

Three residual gaps from prior M8 (the hedge at `04:308` was added, the flags were not):

1. `vllm serve /path/to/gemma-4-31b-it-qat-w4a16-ct` — `/path/to` is not `$WORK/models/gemma-4-31b-ct`
   from §0.4, so the command needs editing before it runs; and without `--served-model-name` the served
   name is the path, while the harness binds `openai/<declared_model>` (`S5:187`).
2. `reasoning_config = {"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}` is part
   of the harness's serving config (`S5:152`) and is absent here, so a manual server parses thinking
   differently from the scorer.
3. The 12B `--hf-overrides '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'`
   (`S6:236-241`) is still unmentioned; swapping to 12B per §3 fails or loads the wrong architecture.

**Fix:** add to the command

```bash
vllm serve "$WORK/models/gemma-4-31b-ct" \
  --served-model-name gemma-4-31b-it-qat-w4a16-ct \
  --reasoning-config '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}' \
  ...
# 12B only: --hf-overrides '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'
```

and change `04:303` to `--lora-modules main_lora=$AGENT/adapters/main_lora` (matches §7.3).

### M6 — `--max_steps 10` is still the headline smoke command

**Where:** `04:531` (`accelerate launch -m axolotl.cli.train config.yml --max_steps 10`), caveat at `04:534`.

Prior L6 recommended inverting this: YAML first, override second. The doc still leads with the
unverified override, so the first command a reader runs may exit on an unrecognised argument.

**Fix:** lead with

```bash
accelerate launch -m axolotl.cli.train config.yml   # with max_steps: 10 added to the YAML
```

and keep the `--max_steps 10` form as the "if your version supports it" alternative.

### M7 — Adapter inspection path does not match the copy destination

**Where:** `04:570` ("Copy only the adapter files into `$AGENT/adapters/main_lora/`"), `04:578`
(`p="adapters/main_lora"`).

The snippet opens `adapters/main_lora/adapter_config.json` relative to whatever cwd the reader has; the
files were copied to `$AGENT/adapters/main_lora`. If the cwd is the training dir (where §7.1/§7.2 ran),
this raises `FileNotFoundError`; if it is `$AGENT`, it works by luck.

**Fix:** `p=os.environ["AGENT"] + "/adapters/main_lora"` (and `import os`).

### M8 — "Stay below the constant" versus `<=`

**Where:** `04:658` versus `04:668`.

The prose says "unpacked size ≤3,221,225,472 bytes (the README says `<3 GiB`; stay below the constant)",
while the check is `assert total <= MAX_SUBMISSION_SIZE_BYTES`. A zip of exactly 3,221,225,472 bytes
passes locally and is at the boundary of S5's "< 3 GiB" (`S5:115,120`). Prior M1 asked for consistent
wording; the runbook now states both readings and implements the permissive one.

**Fix:** use `<` in the assert and say so: `assert total < MAX_SUBMISSION_SIZE_BYTES` with the note
"S5 says `< 3 GiB`; the notebook's `<=` (`S6:607`) is the only enforcement seen locally, so leave
headroom."

### M9 — Two names for the same 8-bit Adam

**Where:** `03:393`, `03:440` (`optimizer: adamw_torch_8bit`) versus `03:504` (`optim="adamw_8bit"`).

The Axolotl recipes and the Unsloth sketch use different identifiers for bitsandbytes 8-bit Adam. Only one
can be the name the respective trainer accepts; `adamw_torch_8bit` appears in no local source, and
Axolotl validates `optimizer` against a fixed list (`[VERIFY]` against the installed version — the
transformers/Axolotl spelling is `adamw_8bit`, with `paged_adamw_8bit` as the paged variant).

**Fix:** set both Axolotl configs to `optimizer: adamw_8bit` (or `paged_adamw_8bit`) and add one line to
the §7.1 notes: "Axolotl's 8-bit Adam is `adamw_8bit`/`paged_adamw_8bit`; `adamw_torch_8bit` is not a
valid value."

### M10 — `tokenizer=` is the pre-rename TRL argument

**Where:** `03:494`.

The caveat at `03:467` acknowledges the rename but the code still passes the old keyword, so on current
TRL the snippet emits a deprecation warning (or raises, depending on version) — the hedge covers the
reader's attention, not the snippet.

**Fix:** `processing_class=tokenizer,` with a comment `# TRL < 0.13: tokenizer=tokenizer`.

### M11 — Training base ID casing is not the registered ID

**Where:** `03:370`, `03:418` (`base_model: google/gemma-4-31B  # verified BF16 ID (S12)`).

S5:160 registers the BF16 CLI aliases as `gemma-4-31b-it` / `gemma-4-31b` (lowercase), and every other
model ID in the docs is lowercase. `google/gemma-4-31B` is not in any local source, and HF repo IDs are
case-sensitive, so the load can 404. The `# verified` comment asserts verification the repo does not
contain.

**Fix:** `base_model: google/gemma-4-31b-it  # [VERIFY repo ID + revision]`, matching `S5:160`.

### M12 — `vllm serve` runs whatever `vllm` is on PATH

**Where:** `04:289-303` versus `04:594` ("Use the **same wheelhouse vLLM build** … the
`$WORK/harness-venv` from §1.2").

§3.1 exports environment variables but never activates the venv built in §1.2, so a manual serve uses the
system `vllm`. §8.1 then claims the reader is using the harness build. Since the Gemma 4 LoRA fix may be
present or absent per build (S23), the A/B in §8.2 can be measuring the wrong engine.

**Fix:** add as the first line of §3.1: `source $WORK/harness-venv/bin/activate` and
`python -c "import vllm; print(vllm.__version__)"`.

### M13 — `activation_offloading: true` in the single-GPU recipe

**Where:** `03:402` (also `03:449`, where FSDP is enabled).

`activation_offloading` is an FSDP activation-offload knob (`03:316` groups it with `fsdp_version`,
`transformer_layer_cls_to_wrap`, `reshard_after_forward`). In the single-GPU QLoRA recipe it is at best a
no-op and at worst a config error `[VERIFY]`; a reader debugging a failed smoke test will not know whether
to remove it.

**Fix:** drop it from §7.1, or annotate: "FSDP-only; ignored/rejected without `fsdp_version: 2`."

### M14 — "no symlinks" is not observable with `unzip -l`

**Where:** `04:259`.

`zip` follows symlinks unless `-y` is given, so a symlink is stored as a regular file and `unzip -l`
cannot show it. The check as written passes on a submission that `validate_directory` will reject.

**Fix:** "`zip` follows symlinks by default, so `unzip -l` cannot prove their absence — rely on the §9.1
`os.walk` assert, or add `zip -y` to store them as links so the listing exposes them."

---

## 5. Low

- **L1 — `04:157`.** `${name/cu128/+cu128}` is a bash-ism; the block has no shebang and the doc's other
  snippets are POSIX-safe. Under `sh`/`dash` it aborts the loop. Fix: add `#!/usr/bin/env bash` or use
  `name=$(printf '%s' "$name" | sed 's/cu128/+cu128/')`. Related: `pip install … /tmp/wheelhouse/*.whl`
  (`04:165`) fails with a literal glob if the staging loop produced nothing — add
  `[ -n "$(ls -A /tmp/wheelhouse)" ] || { echo "no wheels staged"; exit 1; }`.
- **L2 — `04:111`.** The expected dataset listing omits `submission.parquet` (S2 lists it with
  `sandbox/setup.py`). Add it so a partial download is detectable from the expected line alone.
- **L3 — `04:303`.** The LoRA fragment begins at `--enable-lora` with no `vllm serve <model>`, so it is
  not runnable on its own, and its path (`/path/to/adapters/main_lora`) disagrees with §7.3's
  `$AGENT/adapters/main_lora`. Either merge it into the main command or prefix it with
  `vllm serve "$WORK/models/gemma-4-31b-ct" \`.

---

## 6. Checked and consistent (no finding)

- **`build_submission_limits()` unpacking — fixed and correct.** `04:243` and `04:639` both do
  `limits, _ = build_submission_limits()` and pass `limits=limits`, matching
  `getting_started_notebook.txt:352` + `:597`. `validate_directory` receives a `Path` at `04:244`
  (`Path(os.environ["AGENT"])`) and `04:640` (`Path("submission")`) — the same type the notebook passes
  (`AGENT_DIR`). The only defect left is the *base* of that path (C1/M1), not the call.
- **`swegemma eval` flags exist.** `--tasks`, `--snapshots-dir`, `--submission-dir`, `--results-dir`,
  `--sandbox`, `--task-id`, `--task-ids` (multi-value), `--max-tool-calls`, `--max-time-minutes`,
  `--concurrency`, `--display`, `--skip-agent-patch`, `--models-yaml` all appear in `S5:584-605`.
  No invented flag in §3.3/§5/§6.1.
- **Result artifacts match.** `summary.json`, `task_results.jsonl`, `patches/`, `traces/trace_<id>.json`
  (`04:347-348`, `04:478`) match `S5:608-620` exactly.
- **`+cu128` staging logic** matches the notebook's intent (`S6:68-78`): skip cutlass, restore the tag
  only when `cu128` is present without `+`, symlink with copy fallback. (Only the scope is wrong — H2.)
- **Size/extension check semantics** (`04:664-673`) match `S6:606-611`: unpacked `file_size` sum,
  per-member suffix in `ALLOWED_SUBMISSION_EXTENSIONS`, directories skipped. `.safetensors` is in the
  allowed set (`S5:121-131`), so the §9.2 checklist does not contradict the code.
- **Zip root behaviour.** `cd $AGENT` + `zip .` is equivalent to `root_dir=AGENT_DIR` (`S6:600-601`), and
  the prior "first path" claim is gone (`04:259` now says order is not guaranteed).
- **Tool names in §2.1** are exactly the nine registered tools (`S5:353-472`).
- **§1.2 wheelhouse slug** is correctly labelled as inferred with a discovery command (`04:169`).
- **Budget values** in §2.1 (`28/8/80` reproduction, `24/5/60` primary) are internally consistent and
  inside S5's bounds; `timeout_seconds: 240` is a valid `evaluation.timeout_seconds` key (S5:505).
- **Prior-fix regression check:** C1 (unpacking), H1 (staging), H5 (`03:576` now says "computed"), H6
  (template intent), M9 (docker build step), M15 (§6.1 `...`), L9 (zip order), L11 (partial) all landed
  in some form. New defects introduced by those fixes: **H2, H4, H5, H10, C1**.

---

## 7. Not verifiable offline

- Axolotl's accepted `chat_template` names, `optimizer` list, `activation_offloading` semantics and
  `--max_steps` CLI support (H4, M6, M9, M13) — no Axolotl source or docs are present locally.
- TRL's current `SFTConfig` field name and default truncation length (H8, M10).
- The HF repo ID/revision for the QAT W4A16 CT checkpoint (H9) and for the 12B/E4B trainable bases.
- `$DATA/docker/` file names (H10).
- Whether `swegemma eval` creates `--results-dir` and whether it reuses an already-running `:8000`
  server (M1, M5) — the `swegemma` package is not in this workspace.
