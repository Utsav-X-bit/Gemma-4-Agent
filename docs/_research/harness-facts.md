# Harness Facts — swegemma Evaluation Harness

Source of truth: `harness_readme_raw.md` (627 lines, read in full). Citations are `harness_readme_raw.md:<line>`; quotes are verbatim. Anything not stated in that file is marked as absent, not inferred.

Stack identity:
- Three libraries: `adk-submission` (declarative compiler + inference server manager), `adk-eval-core` (task models, sandboxes, `apply_replacement`, `TokenBudget`/`PricingTable`, `EventDisplayPlugin`/`ModelRetryPlugin`, ATIF v1.7 `SessionTrace`), `swegemma` (SWE-bench lifecycle `Evaluator`/`run_agent_sandbox`/`verify_task`, the 9 tools, prompt/nudges, diff extraction, pytest+JUnit scoring) — `harness_readme_raw.md:21-23`.
- Competitors submit **no Python entrypoint** (`agent.py` / `agent_fn`); `importlib` is never used; everything is YAML resolved against closed registries `ToolRegistry`, `ModelRegistry`, `SkillRegistry`, `CallbackRegistry` — `harness_readme_raw.md:57`.
- Data flow: submission → validator → compiler → ADK runner ↔ local server on `:8000` (OpenAI-compatible `/v1`) ↔ tools ↔ Container A → diff → Container B → score — `harness_readme_raw.md:44-52`.

## 1. Submission contract, layout, limits, `!include`

### Packaging and layout
- Submission is "packaged at the root of `submission.zip`" and evaluated by `scripts/inference.py` in Stage 1 to produce `submission.parquet` — `harness_readme_raw.md:61`.
- Canonical tree — `harness_readme_raw.md:63-82`: `agent.yaml` (required), `eval_config.yaml` (optional), `configs/sampling.yaml` (optional, loaded via `!include`), `prompts/*.md` (optional, via `!include`), `sub_agents/*.yaml` (optional sub-agent / AgentTool configs), `adapters/<name>/{adapter_config.json,adapter_model.safetensors}`, `skills/<name>/SKILL.md`.
- **AMBIGUOUS** — zip root: line 61 says the directory is packaged "at the root of `submission.zip`", while the tree is drawn with a top-level `submission/` node (`harness_readme_raw.md:63`) and line 84 requires the root config "in the submission root". Reading A: zip contains `agent.yaml` at top level. Reading B: zip contains one `submission/` folder. The file does not disambiguate.
- Extraction target on Kaggle is `/kaggle/working/submission` — `harness_readme_raw.md:166`.

### Root config discovery
- Exactly one of `agent.yaml`, `agent.yml`, `root_agent.yaml`, `root_agent.yml` must exist in the submission root — `harness_readme_raw.md:84`.
- Zero → `MissingRootConfigError`; more than one → `MultipleRootConfigsError` — `harness_readme_raw.md:84`.

### `eval_config.yaml`
- Optional, at root, under an `evaluation:` key with `timeout_seconds`, `max_tool_calls`, `max_time_minutes`, `max_turns`; loaded by `load_submission_eval_config` during Stage 1 inference — `harness_readme_raw.md:85`.
- Sample block — `harness_readme_raw.md:493-497`:
  ```yaml
  evaluation:
    timeout_seconds: 300
    max_tool_calls: 10
    max_time_minutes: 1
    max_turns: 50
  ```
- Gotcha: `evaluation.timeout_seconds` maps to the **per-command** timeout (`config.harness.command_timeout_seconds`, 300 s), not to session wall clock; session wall clock is `evaluation.max_time_minutes` — `harness_readme_raw.md:500,505`.

### `!include`
- Syntax `!include <relative_path>` — `harness_readme_raw.md:86`.
- `.md` / `.txt` → raw UTF-8 string (intended for `instruction` / `global_instruction`) — `harness_readme_raw.md:87`.
- `.yaml` / `.yml` → parsed recursively, maximum include depth 10, with cycle detection — `harness_readme_raw.md:88`.
- Blocked: absolute paths, null bytes, `..` traversal components, symlinks resolving outside the submission root → `PathTraversalError` — `harness_readme_raw.md:89`.
- **AMBIGUOUS** — `max_yaml_size_bytes` is "50 MiB per YAML file / cumulative `!include` expansion" (`harness_readme_raw.md:123`); it is not stated whether the 50 MiB is per file *and* a separate cumulative cap, or one cap applied to both.

### Hard size limit
- Total **unpacked** submission size `< 3 GiB` = `3,221,225,472` bytes, including all `adapters/` weights — `harness_readme_raw.md:115,120`.
- Rationale given: fast extraction into `/kaggle/working/submission`, adapter fit, and `max_lora_rank <= 128` — `harness_readme_raw.md:166`.
- **Not stated**: any compressed-size cap on `submission.zip` itself (only unpacked size appears, `harness_readme_raw.md:115,120`).

### Structural ceilings (table, `harness_readme_raw.md:121-129`)
| Parameter | Limit | Line |
|---|---|---|
| `max_file_count` | 10,000 files | 121 |
| `max_yaml_files` | 1,000 YAML files | 122 |
| `max_yaml_size_bytes` | 50 MiB per YAML file / cumulative include expansion | 123 |
| `max_skill_size_bytes` | 50 MiB per skill directory | 124 |
| `max_instruction_chars` | 1,000,000 per agent; 10,000,000 total across all agents | 125 |
| `max_agents` | 500 agents in the compiled tree | 126 |
| `max_sub_agent_depth` | 50 levels (`sub_agents` and `agent_tool`) | 127 |
| `max_skills` | 1,000 | 128 |
| `max_loop_iterations` | 500 per `LoopAgent` | 129 |

- These are described as "generous safety ceilings solely to guard against runaway recursion or zip/YAML anchor bombs" — `harness_readme_raw.md:117`.
- `LoopAgent.max_iterations` default 500, bounded `1..500` — `harness_readme_raw.md:112`.
- **AMBIGUOUS** — `max_file_count` = "10,000 files" does not say whether it counts every unpacked file (adapter shards, skill assets) or only config/prompt files; the row sits under "Structural Safety Ceilings" with no scope note (`harness_readme_raw.md:121`).

### Allowed extensions
- Permitted: `.yaml`, `.yml`, `.md`, `.txt`, `.py` (ADK skill scripts), `.json` (`adapter_config.json`), `.safetensors` (`adapter_model.safetensors`) — `harness_readme_raw.md:130`.
- Rejected: pickle-based weights `.bin`, `.pt`, `.pth`, and other binary/archive extensions — `harness_readme_raw.md:130`.
- Adapters: `.safetensors` only, at `adapters/<name>/adapter_model.safetensors` — `harness_readme_raw.md:131`.
- **AMBIGUOUS** — `.py` is annotated "ADK skill scripts" (`harness_readme_raw.md:130`); the file never states whether `.py` is accepted outside `skills/` (e.g. a loose helper at submission root).

### Security posture
- Declarative YAML only; "no symlinks, no path traversal" — `harness_readme_raw.md:115`.

## 2. Agent schema (adk-submission)

- Four discriminated types; `agent_class` omitted → `LlmAgent` — `harness_readme_raw.md:92`.

### `LlmAgent` fields (`harness_readme_raw.md:95-106`)
- `name` (str, required) — 95.
- `model` (str): "required unless a default model is set"; alias from the registry — 96. **AMBIGUOUS**: what the "default model" is, and where it is configured, is not stated.
- `adapter` (str | None): adapter name matching a subdirectory or weight file under `adapters/` — 97.
- `description` (str, up to 1,000,000 chars), used by parents when delegating — 98.
- `instruction` (str, up to 1,000,000 chars): system prompt; supports ADK session-state templating `{problem_description}`, `{hints}`, or custom `output_key` variables — 99.
- `tools` (list): tool-name strings, `agent_tool: {config_path: ..., skip_summarization: true}` references, **or inline agent dictionaries** — 100.
- `skills` (`list[str] | None`): relative paths to skill directories inside the archive — 101.
- `sub_agents` (list): `{config_path: ...}` references for multi-agent transfer — 102.
- `output_key` (str | None): writes the agent's final text into `session.state[output_key]` — 103.
- `include_contents` (`"default" | "none"`): whether conversation history is passed to the agent — 104.
- `disallow_transfer_to_parent` / `disallow_transfer_to_peers` (`bool | None`) — 105. **AMBIGUOUS**: only the type is documented; no semantics or defaults are given.
- `generate_content_config` (`dict | None`) — 106.

### Other classes
- `SequentialAgent`: runs `sub_agents` in order; accepts `name`, `description`, `sub_agents`, agent callbacks — `harness_readme_raw.md:108`.
- `ParallelAgent`: runs `sub_agents` concurrently in isolated branches; same accepted fields — `harness_readme_raw.md:110`.
- `LoopAgent`: repeats `sub_agents` up to `max_iterations` (default 500, bounded 1..500) — `harness_readme_raw.md:112`.

### `generate_content_config` / thinking (`harness_readme_raw.md:132-137`)
- Allowed fields: `temperature`, `top_p`, `top_k`, `max_output_tokens`, `presence_penalty`, `frequency_penalty`, `stop_sequences`, `response_mime_type`, `seed`, `thinking_config` — 132.
- `max_output_tokens`: 1..32,768, default 16,384, bounded by vLLM `max_model_len=32768` — 133.
- `thinking_config.thinking_budget`: 0..32,768, default 4,096; `>0` sets `enable_thinking: true` and forwards `extra_body.thinking_token_budget` to vLLM's `ThinkingTokenBudgetLogitsProcessor`; `0` sets `enable_thinking: false` — 134.
- `thinking_config.thinking_level`: `"MINIMAL" | "LOW" | "MEDIUM" | "HIGH" | "NONE"`, case-insensitive — 135.
- Guidance: on vLLM's `/v1` endpoint (especially with LoRA), omit `thinking_level` and use `include_thoughts: true` + `thinking_budget: 4096`, because `thinking_level` maps to OpenAI `reasoning_effort` and swegemma sets `litellm.drop_params = True` — 135.
- `include_thoughts: true` keeps `Part(thought=True)` in `LlmResponse` and later tool-call context (`<|channel>thought...<channel|>`); `false` still enables reasoning on the current turn but strips thoughts from the response and outgoing history; if neither `thinking_budget` nor `thinking_level` is set, it acts as the fallback `enable_thinking` toggle — 136.
- Unrestricted fields: temperature ≥ 0.0, 0.0 ≤ top_p ≤ 1.0, top_k ≥ 1; `seed` forwarded in the request body to LiteLLM/vLLM — 137.
- Rejected inside `generate_content_config`: `tools`, `system_instruction`, `http_options`, `safety_settings`, `response_schema` — excluded by schema design, validation error if set — `harness_readme_raw.md:139`.

## 3. Models, single-model rule, vLLM, LoRA

### Hardware
- 4 × NVIDIA L4, 24 GB GDDR6 each, 96 GB total VRAM — `harness_readme_raw.md:144,146`.

### vLLM serving parameters (`harness_readme_raw.md:147-154`)
- `tensor_parallel_size = 4` — 148.
- `gpu_memory_utilization = 0.90` (~21.6 GB/GPU, ~86.4 GB total) — 149.
- `max_model_len = 32768` (prompt + reasoning + output combined) — 150.
- `scheduler_reserve_full_isl = False` — 151.
- `enable_auto_tool_choice = True`; `tool_call_parser = "gemma4"`; `reasoning_parser = "gemma4"`; `reasoning_config = {"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}` — 152.
- LoRA right-sizing: `enable_lora = bool(adapter_map)`, `max_loras = max(1, len(adapter_map))`, `max_lora_rank = max(detected_ranks) if detected_ranks else 128` — 153.
- Model-specific args: for `gemma-4-12b-it-qat-w4a16-ct`, `--hf-overrides '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'` — 154.

### Model table (`harness_readme_raw.md:155-160`)
- `gemma-4-31b-it-qat-w4a16-ct` (starter-kit default), W4A16, ~16–18 GB total (~4.5 GB/GPU) — 156.
- `gemma-4-12b-it-qat-w4a16-ct`, W4A16, ~6–8 GB (~1.8 GB/GPU) — 157.
- `gemma-4-e4b-it-qat-w4a16-ct`, W4A16, ~3–4 GB (~1.0 GB/GPU) — 158.
- `gemma-4-e2b-it-qat-w4a16-ct`, W4A16, ~2–3 GB (~0.7 GB/GPU) — 159.
- bf16 variants (`gemma-4-31b-it`, `-27b-it`, `-26b-a4b-it`, `-12b-it`, `-9b-it`, `-e4b-it`, `-e2b-it`) ~4–62 GB, "local CLI evaluation only" — 160.

### Four rules imposed by the hardware (`harness_readme_raw.md:164-167`)
- One base model per submission from the four official quantized models — 164.
- 32,768-token ceiling for `max_output_tokens` and `thinking_budget` — 165.
- Unpacked submission < 3 GiB including adapters — 166.
- 4 GiB RAM / 2 vCPU per container so repo test suites cannot starve vLLM — 167.

### Single-model validation (`harness_readme_raw.md:170-173`)
- `validate_single_declared_model(agent_dir)` wraps `adk_submission.discovery.discover_declared_models`; traverses root `agent.yaml`, every `sub_agents[*].config_path`, every `tools[*].agent_tool.config_path`, and standalone agent YAMLs — 170.
- Prefixes `openai/`, `google/`, `hosted_vllm/`, `custom/` stripped by `normalize_model_name` — 172.
- Must satisfy `len(models) == 1` and (in Kaggle scoring) be in `ALLOWED_MODEL_NAMES`; violation → `ParticipantVisibleError` — 173.

### Registry aliases (`harness_readme_raw.md:178-185`)
- Official quantized: `gemma-4-31b-it-qat-w4a16-ct`, `gemma-4-12b-it-qat-w4a16-ct`, `gemma-4-e4b-it-qat-w4a16-ct`, `gemma-4-e2b-it-qat-w4a16-ct` — 178.
- Short aliases: `gemma-4-31b-it`/`gemma-4-31b`; `gemma-4-27b-it`/`gemma-4-27b`; `gemma-4-26b-a4b-it`/`gemma-4-26b-a4b`/`diffusiongemma-26b-a4b-it`; `gemma-4-12b-it`/`gemma-4-12b`; `gemma-4-9b-it`/`gemma-4-9b`; `gemma-4-e4b-it`/`gemma-4-e4b`; `gemma-4-e2b-it`/`gemma-4-e2b` — 179-185.
- Registered in `models.yaml` and `_build_cli_model_registry()` (`src/swegemma/cli.py`) — 176.

### Serving / routing
- Server: `VllmServer` on `127.0.0.1:8000` with `tool_call_parser='gemma4'`, `reasoning_parser='gemma4'`, `max_model_len=32768` — `harness_readme_raw.md:187`.
- Model binding: `LiteLlm(model=f"openai/{served_model}", api_base="http://127.0.0.1:8000/v1")` via `server_instance.create_model_registry(...)`; fallback `LiteLlm(..., num_retries=3)` when attaching to an already-active endpoint; installs `install_litellm_reasoning_patch()` — 187.

### LoRA (`harness_readme_raw.md:191-201`)
- Different agents may use different adapters even though the base model is fixed — 191.
- Layout: `adapters/<adapter_name>/` with `adapter_config.json` + `adapter_model.safetensors`; names must match `[A-Za-z0-9][A-Za-z0-9._-]*` — 193.
- Per-agent selection: `adapter: <adapter_name>` on any `LlmAgent` (e.g. `main_lora` on the coder, `tool_lora` on a read-only analyzer AgentTool) — 194.
- `discover_adapters()` registers all adapters, passes `--lora-modules name1=path1 name2=path2 ...`; `resolve_swegemma_adapter` (`src/swegemma/harness/agent_runner.py`) routes each agent to `openai/<adapter_name>` — 195.
- Size planning for a 31B bf16 adapter over all linear projections (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) — 197: r=16 → ~110–220 MB (≈8 fit); r=32 → ~220–450 MB (6–8 fit); r=64 → ~450–900 MB (3–6 fit); r=128 (max) → ~0.9–1.8 GB (1–3 fit) — 198-201.

## 4. Two-phase lifecycle, sandbox, bootstrap

- Every task runs in two completely separate sandbox lifecycles: Container A (agent) and Container B (verification) — `harness_readme_raw.md:204`.
- 14-step sequence — `harness_readme_raw.md:212-230`: start sandbox+wheels → extract snapshot at `base_commit` → `.git/info/exclude` + editable install → stream test site-packages + `pytest.ini`/`conftest.py` → `git add -A && git commit -m "baseline"` → start timer and send `build_agent_prompt()` → tool loop → extract patch → destroy A → fresh B repeating steps 1–5 as `eval_baseline` → apply `agent_patch` → reset tests → apply `test_patch` → hermetic pytest → `exit_code == 0 → resolved = True`.
- Loop-exit conditions in the diagram: `submit_patch()`, timeout, budget exhausted, or 3 nudges — `harness_readme_raw.md:218`.

### Sandbox hardware/network (`harness_readme_raw.md:233-242`)
- Docker backend default: image `swebench-sandbox:latest` — 234.
- `network_mode="none"` — "completely air-gapped; no internet or PyPI access" — 235.
- Memory `4g`; exceeding it kills the process with exit code 137 / SIGKILL — 236.
- CPU: 2 vCPUs (`cpu_period=100_000`, `cpu_quota=200_000`) — 237.
- Working dir `/workspace`, `TEST_TMPDIR=/tmp` — 238.
- `reuse_containers=False` default; warm pooling wipes `/workspace` with `find /workspace -mindepth 1 -maxdepth 1 -exec rm -rf {} +`, removes `/tmp` and `/var/tmp` patches/stubs, returns container to idle pool — 239.
- Subprocess backend (no Docker daemon): temp dir `swegemma_sandbox_<id>_` containing `workspace/`, `tmp/`, `wheels/`, `venv/`; rewrites `/workspace`, `/tmp`, `/wheels`, `/usr/local/bin` to sandbox paths; runs in a new process group (`start_new_session=True`) and kills the group with `os.killpg(..., signal.SIGKILL)` on timeout — 240-242.

### Container bootstrap (`harness_readme_raw.md:245-265`)

- The text claims "7 deterministic setup steps" (`harness_readme_raw.md:245`) but lists **8** bullets (wheel staging through baseline commit, `harness_readme_raw.md:247-265`). Treated here as 8 actual operations; the "7" is a documentation inconsistency, not a behavior claim.

1. `setup_container_wheels`: prepares `/wheels` — 247.
2. `extract_snapshot`: extracts the repo `.tgz` (or reconstructs from deduplicated base snapshot + binary patch) into `/workspace`, removing circular symlinks — 248.
3. Zero future history: `git fast-export` up to `base_commit` piped into a fresh `git fast-import` repo on branch `main`; later commits do not exist in `.git` — 249.
4. `setup_git_exclude`: appends `__pycache__/`, `*.pyc`, `.pytest_cache/`, `*.egg-info/`, `build/`, `dist/`, `.coverage` to `/workspace/.git/info/exclude` — 250-257.
5. `install_editable_package`: `pip install --no-index --find-links=/wheels --no-build-isolation --no-deps -e /workspace 2>/dev/null || true` — 259-260.
6. `install_test_dependencies`: streams cached unpacked site-packages (`<20 MB` base wheels and `>=20 MB` workspace-imported wheels), runs `/sandbox/setup.py --fast-path <repo>` — 262.
7. `setup_workspace_test_config`: writes `/workspace/pytest.ini` (disables anyio, ignores hardware/device/example dirs) and prepends a hermetic collection hook to `/workspace/conftest.py` — 263.
8. `setup_baseline_commit`: `cd /workspace && git add -A && git commit -m "baseline" --allow-empty -q` — 264-265.
- Ordering consequence: `pytest.ini` and `conftest.py` are committed into baseline HEAD, so edits to them appear in `git diff HEAD` — `harness_readme_raw.md:268`.
- Phase 2 repeats the same setup in a fresh container and commits `eval_baseline` as HEAD — `harness_readme_raw.md:225,542`.

### Global 12h rule
- **Not stated anywhere in this file.** There is no 12-hour, notebook-runtime, or global evaluation deadline. The only time limits are per-task: `max_time_minutes` default 60.0 (`harness_readme_raw.md:500`), per-command 300 s (`harness_readme_raw.md:505`), Phase 2 pytest 300 s (`harness_readme_raw.md:506`), and the prompt's "Session terminates when any budget is exhausted" (`harness_readme_raw.md:295`).
- **AMBIGUOUS** — Phase-2 entry condition is written "If `agent_patch` is non-empty (or `no agent_error`), proceed to Phase 2" (`harness_readme_raw.md:224`); the parenthetical's boolean relation to the first clause is not resolvable from the text. Line 540 gives the operative rule for scoring: proceed "when `agent_patch` (prediction) is non-empty".

## 5. Harness ↔ agent protocol

### Session state (`harness_readme_raw.md:273-278`)
- ADK session created with `app_name='swegemma_eval'`, `user_id='eval_user'` — 273.
- `session.state["problem_description"] = task.problem_statement` — 275.
- `session.state["hints"] = task.hints_text.strip()`, present only when `hints_text` is non-empty — 276.
- `{variable_name}` placeholders in `instruction` are interpolated from session state by ADK, so `{problem_description}` is directly usable — 278.

### Initial user prompt — up to 7 sections (`harness_readme_raw.md:282-319`)
- Sections are filtered against `_extract_declared_tools(agent)` so undeclared tools are never advertised — 282.
1. Header: "You are evaluating a software engineering task for repository {task.repo}." + "Problem Statement:" + statement — 284-288.
2. Hints, only if non-empty: `## Hints:` — 290-292.
3. Task budget block: "## Task Budget (Session terminates when any budget is exhausted)" with time allowance (e.g. 60.0 minutes), tool calls, max loop iterations, cost budget — 294-299.
4. Execution environment rules: 300 s command timeout ("commands exceeding this fail without ending the session"), 5000-char command output limit, 150 lines/file view, 10000 chars/file view, offline notice — 301-307. The two `read_file` limits are omitted when `read_file` is not declared — 301.
5. Standard instructions (labelled "0–5"): work strictly under `/workspace`, inspect existing conventions, verify with targeted tests/inline assertions, call `submit_patch` once complete, return a final text response — 309-310. The README does not enumerate all five.
6. Code intelligence tools: appended only when graph `.json` and embedding `.npz` files `>100 bytes` exist for `task.repo`, and only for declared graph/embedding tools; lists `search_similar_code`, `get_code_neighbors`, `get_code_subgraph` — 312-317.
7. Workspace layout: up to 200 breadth-first, hierarchically sorted entries from `find . -mindepth 1 -maxdepth 3` in `/workspace`, excluding hidden entries, `__pycache__`, `*.pyc`, `*.egg-info`, and pruning bulk dirs (`docs`, `docs_src`, `tests`, `examples`, `benchmarks`) below depth 1 — 319.
- `enable_sandbox_testing = True` by default (EvalConfig and `agent_runner.py`), so `pytest`/`unittest` work in Container A; if set `False`, the harness masks both and Instruction 3 switches to inline `python3 -c "..."` assertions — 311.
- **AMBIGUOUS** — the cost line is always shown as `Cost budget: $<X.XX> USD` (`harness_readme_raw.md:299`) while `cost_usd` defaults to `None` (`harness_readme_raw.md:503`); whether the line is omitted or rendered as `$0.00` is not stated.

### Tool error recovery (`harness_readme_raw.md:324-326`)
- `ToolErrorPlugin` registers `on_tool_error_callback` on the ADK App; undeclared/hallucinated tool calls and runtime exceptions return `{"status": "error", "error_type": "<ExcName>", "error": "<Message>"}` instead of aborting the task — 325.
- `ToolErrorPlugin.before_tool_callback` and `_wrap_skill_toolset_tools` coerce single-element list strings (`["name"]` → `"name"`) for `skill_name` and `file_path` on `load_skill`, `load_skill_resource`, `run_skill_script` — 326. (Source contains a LaTeX artifact `$\to$` at this line.)

### Termination and nudges (`harness_readme_raw.md:327-338`)
- `submit_patch()` sets `context.patch_submitted = True`; the loop breaks as soon as a final text response is emitted or the current `run_async` turn ends with `patch_submitted == True` — 328-329.
- `max_nudges = 3`; a turn that executed ≥1 valid tool call resets `consecutive_nudges` to 0; otherwise it increments and `> 3` terminates the loop — 330-333.
- Nudge selection — 335-338:
  - Unclosed `<|tool_call>` tag → "Your previous response reached the token limit before the tool call finished closing (`<|tool_call|>` was cut off)… split the change into smaller incremental edits."
  - `finish_reason` contains `MAX_TOKENS`/`LENGTH` → "…reached the token limit while thinking before a tool call was completed… keep reasoning under a few sentences…"
  - Normal stop without `submit_patch` → "Please continue your work using the available tools, or call `submit_patch` when you have completed and verified your changes."

## 6. The 9 built-in tools

- Registered by `SwegemmaContext.create_tools()` in `ToolRegistry`; attach any subset by name under `tools:` — `harness_readme_raw.md:341`.
- Envelopes: success `{"status": "ok", ...}`; error `{"status": "error", "error_type", "error_message", "details"}` — `harness_readme_raw.md:345-346`.
- Budget warning injected into every tool response when `budget.tool_calls >= 20` **and** `remaining_tool_calls <= 10`: `"budget_warning": "Only X tool call(s) remaining (Y/Z used). Finalize your edits and call submit_patch soon."` — `harness_readme_raw.md:348-350`.

### 1. `run_command(command: str) -> str` — `harness_readme_raw.md:353-362`
- Runs `/bin/bash -c` inside `/workspace` — 355.
- Counts toward `tool_calls`: yes (`@budget_gated`) — 357.
- Timeout: `min(command_timeout_seconds [300s], max(5, int(remaining_time_seconds)))` — 358.
- Timeout returns `error_type: "TimeoutExceeded"` without ending the session, unless the overall time budget has also expired — 358.
- stdout and stderr each truncated to `max_stdout_chars` (5,000 default) — 359.
- Exit 0 → `{"status": "ok", "stdout", "stderr", "exit_code": 0}`; non-zero → `{"status": "error", "error_type": "CommandError", "error_message", "details": {"stdout", "stderr", "exit_code"}}` — 361-362.

### 2. `submit_patch() -> str` — `harness_readme_raw.md:363-370`
- Runs `git add -N .`, captures `git diff --binary _swegemma_baseline 2>/dev/null || git diff --binary HEAD` from `/workspace`, stores it in `ctx.submitted_patch`, sets `ctx.patch_submitted = True` — 365.
- Does **not** count toward `tool_calls` (`count_tool_call=False`); callable even when `tool_calls_used == budget.tool_calls` as long as wall-clock time remains — 367.
- Ends the agent loop once the current turn's response finishes — 368.
- Returns `{"status": "ok", "patch_size": 1420, "files_changed": 2}` (example values) — 370.

### 3. `get_status() -> str` — `harness_readme_raw.md:372-390`
- Live budget/patch status; does not count toward `tool_calls` — 374-376.
- Fields: `tool_calls_used`, `patch_submitted`, `patch_size`, `tool_calls_remaining`, `max_tool_calls`, `time_seconds_remaining`, `max_time_minutes`, `agent_elapsed_seconds`, `max_turns`, `command_timeout_seconds` — 379-389.

### Path handling shared by file tools — `harness_readme_raw.md:394`
- Paths resolve relative to `/workspace` via `_resolve_workspace_path`; leading `/` or `/workspace/` is stripped (`/workspace/src/app.py` ≡ `src/app.py`) — 394.
- Any `..` traversal raises `ValidationError` — 394.
- String-encoded ints/bools from the model (e.g. `"10"`, `"true"`) are coerced to Python types — 394.

### 4. `read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> str` — `harness_readme_raw.md:396-411`
- 1-indexed inclusive slicing — 398.
- Dual cap: 150 lines (`max_file_lines`) **and** 10,000 chars (`max_file_chars`); hitting either sets `is_truncated: true`, and `end_line` is the last complete line returned — 401.
- Returns `filepath`, `content`, `start_line`, `end_line`, `total_lines`, `is_truncated` — 404-410.

### 5. `edit_file(filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str` — `harness_readme_raw.md:413-431`
- Requires an existing **non-empty** file; uses `apply_replacement` from adk-eval-core — 415.
- Tier 1 `exact`: character-for-character after normalizing `\r\n` → `\n` — 419.
- Tier 2 `flexible`: line-by-line after stripping leading/trailing whitespace; `new_string` is re-indented to the matched block's baseline indentation — 420.
- Tier 3 `regex`: tokenizes `old_string` around delimiters `( ) : [ ] { } > = <`, joins tokens with `\s*` — 421.
- Uniqueness: >1 occurrence with `allow_multiple=False` → `FileEditError`, no modification; also `FileEditError` for missing file, 0-byte file, or empty `old_string` — 422.
- Returns `filepath`, `occurrences`, `strategy`, `diff` (truncated at 5,000 chars), `is_truncated` — 423-430.

### 6. `write_file(filepath: str, content: str) -> str` — `harness_readme_raw.md:433-439`
- Creates or overwrites `/workspace/<filepath>`, creating parent dirs (`mkdir -p`) — 435.
- Returns `{"status": "ok", "filepath", "size"}` — 439.

### 7. `get_code_neighbors(node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str` — `harness_readme_raw.md:445-454`
- Incoming + outgoing neighbors in the repo call/dependency graph — 447.
- `resolve_node_name` tiers: exact → `.` / `;` / `/` suffix match → case-insensitive → substring (source text is garbled at this point) — 450.
- `edge_type` is an optional case-insensitive filter (e.g. `"calls"`) — 451.
- Neighbor names capped at `max_stdout_chars` (5,000), setting `is_truncated` — 452.
- Returns `node`, `neighbors`, `count`, `is_truncated` — 454.

### 8. `search_similar_code(query: str, k: int = 10) -> str` — `harness_readme_raw.md:456-470`
- Top-k nodes by cosine similarity from the `.npz` embedding archive — 458.
- Offline: `query` is resolved against node keys/suffixes in the graph and `.npz` dictionary, then cosine similarity is computed against all other nodes; lexical token-overlap fallback for multi-word keyword queries — 461.
- Total code characters across all k results capped at 5,000, budget divided across results with unused characters rolling over to larger nodes; `is_truncated` set when any snippet is truncated — 462.
- Returns `query`, `results[{node_name, code, similarity}]`, `count`, `is_truncated` — 464-469.

### 9. `get_code_subgraph(nodes: list[str]) -> str` — `harness_readme_raw.md:472-486`
- Induced subgraph (nodes + interconnecting edges) for a symbol list — 474.
- Nodes/edges lists capped at 5,000 chars — 477.
- Returns `nodes`, `edges[{from, to, type}]`, `node_count`, `edge_count`, `is_truncated` — 479-485.

### Graph-tool preconditions — `harness_readme_raw.md:443`
- Available only with `data/graphs/<repo>.json` and `data/embeddings/<repo>.npz`; backed by an in-memory NetworkX `CustomMultiDiGraph` — 443.
- Auto-coerces string ints for `max_neighbors`/`k` and JSON or comma-separated strings for `nodes` — 443.

## 7. Budgets, eval_config, compaction, retries

### Defaults (`harness_readme_raw.md:499-509`)
| Item | Config path | Default | `eval_config.yaml` key | Env override |
|---|---|---|---|---|
| Session wall clock | `config.budget.time_minutes` | 60.0 min (1 min in sample submission) | `evaluation.max_time_minutes` | `SWE_MAX_TIME_MINUTES` |
| Max tool calls | `config.budget.tool_calls` | 100 in `inference.py` / `None` in CLI (10 in sample) | `evaluation.max_tool_calls` | `SWE_MAX_TOOL_CALLS` |
| Max reasoning turns | `config.budget.turns` | 500 (`None` → 500; 50 in sample) | `evaluation.max_turns` | `SWE_MAX_TURNS` / `SWE_MAX_LOOP_ITERATIONS` |
| Max cost | `config.budget.cost_usd` | `None` | — | — |
| Max total tokens | `config.budget.total_tokens` | `None` | — | — |
| Phase-1 command timeout | `config.harness.command_timeout_seconds` | 300 s | `evaluation.timeout_seconds` | `SWE_TIMEOUT_SECONDS` |
| Phase-2 pytest timeout | `config.harness.verification_timeout_seconds` | 300 s | fixed 300 s on scorer | `SWE_TIMEOUT_SECONDS` (metric.py) |
| Max command/diff/graph output | `config.harness.max_stdout_chars` | 5,000 chars | — | — |
| Max lines per `read_file` | `config.harness.max_file_lines` | 150 lines | — | — |
| Max chars per `read_file` | `config.harness.max_file_chars` | 10,000 chars | — | — |

- Rows: `harness_readme_raw.md:500-509`.
- Container setup is excluded from the agent budget: `task_start_time` is set at `SwegemmaContext` creation but `agent_start_time` only via `context.start_agent_session()` after container creation, wheel install, and snapshot extraction — `harness_readme_raw.md:511`.
- `cost_usd` and `total_tokens` default to `None`, i.e. no cap configured by default — `harness_readme_raw.md:503-504`.

### Compaction & caching (`harness_readme_raw.md:515-523`)
- `EventsCompactionConfig`: `compaction_interval = 15`, `overlap_size = 2`, `token_threshold = 14,336`, `event_retention_size = 5` — 518-521.
- `ContextCacheConfig`: `min_tokens = 2,048`, `ttl_seconds = 1,800`, `cache_intervals = 10` — 523.

### Retries (`harness_readme_raw.md:524-525`)
- `ModelRetryPlugin` intercepts 429/500/502/503/504, `RateLimitError`, `APIConnectionError`, `APITimeoutError`; up to 5 retries; exponential backoff 2.0 s initial, 2.0× multiplier, 60.0 s max, ±20% jitter — 525.

## 8. Patch extraction and Phase 2 verification

### Extraction (`harness_readme_raw.md:529-537`)
- Exact commands, identical for `submit_patch()` and the fallback — 531-532:
  ```sh
  cd /workspace && git add -N .
  cd /workspace && (git diff --binary _swegemma_baseline 2>/dev/null || git diff --binary HEAD)
  ```
- `git add -N .` registers new untracked files with empty content so the diff includes creations, while `.git/info/exclude` patterns stay excluded — 534.
- Protected stripping (`strip_protected_files_from_patch`) applies to both paths: hunks touching `test_*.py`, `*_test.py`, any `.py` under `tests/`, `test/`, `testing/`, or runner config `conftest.py`, `pytest.ini`, `pyproject.toml`, `tox.ini`, `setup.cfg`, `.pytest.ini`, `sitecustomize.py`, `usercustomize.py`, `_swegemma_stubs.py`, `*.pth` — 535.
- Automatic recovery (`agent_runner.py` lines 760–805): if the agent finishes, hits `max_nudges`, exhausts tool calls/turns, or times out without `submit_patch()`, the harness still runs `git add -N .`, `git diff --binary _swegemma_baseline`, and stripping before tearing down Container A — 536.
- A non-empty diff over non-protected files is recorded as `agent_patch` in `submission.parquet` as `[id, prediction]` and is still graded — 537.

### Phase 2 (`harness_readme_raw.md:540-567`)
- Triggered during Stage 2 scoring by `score(solution, submission, row_id_column_name)` in `scripts/metric.py` when the prediction is non-empty; runs in a fresh Container B — 540.
- Fresh baseline: original snapshot at `base_commit`, synthetic hardware stubs, `.git/info/exclude`, editable install, cached test deps, `pytest.ini`/`conftest.py`, `eval_baseline` commit as HEAD — 542.
- 4-pass application until one succeeds — 543-547:
  1. `git apply --unsafe-paths` (`-p1`), then `-3`, then `--ignore-space-change --ignore-whitespace`, then `--recount`.
  2. Symlink-normalized `git apply --unsafe-paths` (`-p1`), resolving symlinked directory prefixes.
  3. Prefixless `-p0` `git apply --unsafe-paths`.
  4. Non-interactive GNU `patch -p1` / `patch -p0` (`--batch --forward`, `-l`), gated by `--dry-run`.
- All four failing → task fails immediately, `resolved = False`, `error = "Failed to apply agent patch: ..."` — 548.
- Test/config reset: paths referenced in `task.test_patch` plus any protected test/config files touched by `agent_patch` are reset to `eval_baseline` via `git ls-tree`-filtered `git checkout HEAD --` and `git clean -f --` (missing pathspecs do not abort) — 549-553.
- Any agent edits to tests or runner config are stripped and discarded before `task.test_patch` — 556.
- Pytest invocation — 559-562:
  ```sh
  cd /workspace && PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -m pytest <pytest_targets> \
    --junitxml=/tmp/_swegemma_junit_<id>.xml \
    -p no:anyio -o timeout=0 \
    -o python_classes="Test* *Test" -q
  ```
- Resolution criterion — `resolved = True` (score 1.0) iff both hold — 565-567:
  1. `test_res.exit_code == 0`, and
  2. `_validate_junit_xml` finds the report, `passed_tests > 0`, `failures == 0`, `errors == 0`, and every required node (`FAIL_TO_PASS`, `PASS_TO_PASS`, or functions extracted from `test_patch`) explicitly passed and was not skipped.
- Competition metric = Resolution Rate = resolved tasks / total tasks ∈ [0.0, 1.0] — 568-578.

## 9. Local CLI and artifacts

- Invocation — `harness_readme_raw.md:584-593`:
  ```sh
  swegemma eval --tasks tasks.jsonl --snapshots-dir snapshots --submission-dir sample_submission \
    --results-dir results/run_01 --sandbox docker --max-tool-calls 50 --max-time-minutes 30 \
    --concurrency 2 --display auto
  ```
- Flags — `harness_readme_raw.md:598-605`:
  - `--task-id <id>` / `--task-ids <id1> <id2>` — specific `instance_id`s for debugging.
  - `--sandbox {docker,subprocess}` — Docker default.
  - `--timeout-seconds <sec>` — Phase 1 per-command timeout, default 300.
  - `--verification-timeout-seconds <sec>` — Phase 2 pytest timeout, default 300.
  - `--concurrency <N>` — parallel tasks; activates the multi-slot dashboard UI.
  - `--shard-index <k> --num-shards <M>` — deterministic shard, `task_idx % M == k`.
  - `--models-yaml <path>` — custom `models.yaml` for aliases and token pricing.
  - `--skip-agent-patch` — bypass Phase 1 (`agent_patch = ""`) to observe baseline test failures in Phase 2.
- Artifacts under `--results-dir`, written incrementally and crash-safe — `harness_readme_raw.md:608-620`: `summary.json` (resolution_rate, resolved count, per-repo breakdown, errors); `task_results.jsonl` (append-only, one line per finished task with metrics/exit codes); `patches/<instance_id>.patch`; `test_outputs/<instance_id>.log` (Phase 2 stdout/stderr); `traces/trace_<instance_id>.json` (full `SessionTrace`, ATIF-compatible); `logs/<instance_id>.log` (Phase 1 transcript).

## 10. Best practices and gotchas

- Keep `edit_file` payloads small: a huge thought block plus a large `edit_file`/`write_file` call hitting `max_output_tokens` truncates the `<|tool_call>` tag; use e.g. `thinking_budget: 4096` with `max_output_tokens: 16384` and incremental edits — `harness_readme_raw.md:623`.
- Do not modify `/workspace/pytest.ini` or `/workspace/conftest.py`; both are harness-written and committed into baseline, so edits land in `agent_patch` — `harness_readme_raw.md:624`.
- Clean up reproduction scripts or put them in `/tmp`: `submit_patch()` diffs all of `/workspace`, so `/workspace/repro.py` would ship in the patch — `harness_readme_raw.md:625`.
- Use `agent_tool` with `skip_summarization: true` for read-only exploration sub-agents (e.g. `sub_agents/code_analyzer.yaml`) to keep intermediate `read_file` output out of the root context — `harness_readme_raw.md:626`.
- Call `submit_patch()` explicitly, last: it is free, `get_status()` is free, and the loop terminates as soon as a turn completes with `patch_submitted == True` — `harness_readme_raw.md:627`.
- Budgets are hard stops: the prompt says "Session terminates when any budget is exhausted" — `harness_readme_raw.md:295`.
- Partial work still counts if it changed non-protected files, because the fallback diff is captured — `harness_readme_raw.md:536-537`.
- Test/runner tampering is futile: those hunks are stripped in Phase 1 and the paths are reset in Phase 2 — `harness_readme_raw.md:535,549,556`.
- Prompt advertising is filtered to declared tools, so a tool not declared anywhere in the hierarchy is never described to the model — `harness_readme_raw.md:282`.

# Needs external verification

- **Global 12-hour (or any notebook-level) runtime limit** — absent from `harness_readme_raw.md`; confirm from Kaggle competition rules. Only per-task budgets appear (`harness_readme_raw.md:500-506`).
- **`_swegemma_baseline` provenance** — the diff commands reference this ref (`harness_readme_raw.md:365,532,536`), but the 7 bootstrap steps only create `baseline` / `eval_baseline` commits (`harness_readme_raw.md:265,542`). Whether a tag/branch of that name is created elsewhere must be checked in `agent_runner.py`/`container_setup.py`.
- **Phase-2 entry condition** — the "non-empty (or no agent_error)" wording at `harness_readme_raw.md:224` cannot be resolved without `evaluate.py`.
- **`max_file_count` scope** — whether it counts adapter shards and skill assets or only configs (`harness_readme_raw.md:121`).
- **`.py` placement** — whether `.py` is allowed only under `skills/` (`harness_readme_raw.md:130`).
- **Zip layout** — whether `submission.zip` has a top-level folder (`harness_readme_raw.md:61,63,84`).
- **`model` "default"** — which default model applies when `model` is omitted (`harness_readme_raw.md:96`).
- **`disallow_transfer_to_parent` / `disallow_transfer_to_peers` semantics and defaults** — type only (`harness_readme_raw.md:105`).
- **Cost/token budget rendering** — how `$<X.XX>` appears when `cost_usd` is `None` (`harness_readme_raw.md:299,503`).
- **Whether `cost_usd` / `total_tokens` are enforced at all**, and by which component (`harness_readme_raw.md:503-504`).
- **Actual `swebench-sandbox:latest` image contents**, wheel cache availability, and `/sandbox/setup.py` behavior (`harness_readme_raw.md:234,247,262`).
- **Whether the 4×L4 machine is the Kaggle scoring host or a separate evaluation host** — the file says "dedicated machine" (`harness_readme_raw.md:144`) while extraction targets `/kaggle/working/submission` (`harness_readme_raw.md:166`).
- **Sample submission's 1-minute / 10-tool-call / 50-turn values** — whether these are starter defaults or merely an example (`harness_readme_raw.md:493-497,500-502`).
- **`hints_text` source and format** — where hints come from is not stated, only that they are interpolated when non-empty (`harness_readme_raw.md:276,290`).
- **Exact `test_patch` / `FAIL_TO_PASS` / `PASS_TO_PASS` schema** — referenced but not defined (`harness_readme_raw.md:549,567`).
- **Whether `response_mime_type` is accepted in practice** — listed as allowed (`harness_readme_raw.md:132`) but `response_schema` is rejected (`harness_readme_raw.md:139`); interaction unverified.
