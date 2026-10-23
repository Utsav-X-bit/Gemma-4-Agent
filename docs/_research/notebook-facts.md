# Notebook fact sheet — `getting_started_notebook.txt`

Scope: verbatim facts from the Kaggle starter notebook (615 lines, 6 code cells) with line
references. Harness README comparisons cite `harness_readme_raw.md`. Anything the notebook
does not establish is tagged `[INFERENCE]` or listed under
`# Needs external verification`.

Notebook structure (section headers in the file):
- `1. Environment Configuration and Package Installation` — `getting_started_notebook.txt:2`, cell `In [1]` at `getting_started_notebook.txt:3`
- `2. Competition Dataset and Sample Submission` — `getting_started_notebook.txt:99`, cell `In [2]` at `getting_started_notebook.txt:100`
- `3. Code Graph Functions` — `getting_started_notebook.txt:151`, cell `In [3]` at `getting_started_notebook.txt:152`
- `4. Start vLLM Server` — `getting_started_notebook.txt:220`, cell `In [4]` at `getting_started_notebook.txt:221`
- `5. Run Phase 1 Inference and Phase 2 Verification` — `getting_started_notebook.txt:303`, cell `In [5]` at `getting_started_notebook.txt:304`
- `6. Package Submission Archive` — `getting_started_notebook.txt:586`, cell `In [6]` at `getting_started_notebook.txt:587`

## 1. Environment variables

Set before any package import that could read them (top of cell 1, imports at `getting_started_notebook.txt:4-11`).

| Variable | Value | Line |
|---|---|---|
| `TRANSFORMERS_NO_TF` | `1` | `getting_started_notebook.txt:14` |
| `TRANSFORMERS_NO_FLAX` | `1` | `getting_started_notebook.txt:15` |
| `TRANSFORMERS_OFFLINE` | `1` | `getting_started_notebook.txt:16` |
| `VLLM_NO_USAGE_STATS` | `1` | `getting_started_notebook.txt:17` |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | `getting_started_notebook.txt:18` |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `1200` | `getting_started_notebook.txt:19` |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | `1` | `getting_started_notebook.txt:20` |
| `NCCL_DEBUG` | `WARN` | `getting_started_notebook.txt:21` |
| `LITELLM_LOCAL_MODEL_COST_MAP` | `True` | `getting_started_notebook.txt:22` |
| `LITELLM_TELEMETRY` | `False` | `getting_started_notebook.txt:23` |
| `LITELLM_LOG` | `WARNING` | `getting_started_notebook.txt:24` |
| `OPENAI_API_KEY` | existing value, else `EMPTY` | `getting_started_notebook.txt:25` |
| `LOCAL_API_KEY` | existing value, else `EMPTY` | `getting_started_notebook.txt:26` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | `getting_started_notebook.txt:27` |
| `OTEL_SDK_DISABLED` | `true` | `getting_started_notebook.txt:28` |
| `OTEL_PYTHON_DISABLED` | `true` | `getting_started_notebook.txt:29` |

- The notebook's own stated purpose for the block is "offline vLLM serving and LiteLLM routing" (`getting_started_notebook.txt:13`).
- Offline serving: `TRANSFORMERS_OFFLINE=1` (`getting_started_notebook.txt:16`) plus `LITELLM_LOCAL_MODEL_COST_MAP=True` (`getting_started_notebook.txt:22`) keep model/token-cost lookups from reaching the network; `VLLM_NO_USAGE_STATS=1` (`getting_started_notebook.txt:17`) and `LITELLM_TELEMETRY=False` (`getting_started_notebook.txt:23`) disable outbound telemetry. `[INFERENCE]` for the mechanism of each individual flag; only the group comment at `getting_started_notebook.txt:13` is stated.
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` (`getting_started_notebook.txt:18`) matters because vLLM is launched from a live Jupyter kernel; `spawn` avoids fork-after-CUDA-init. `[INFERENCE]`
- Timeout: `VLLM_ENGINE_READY_TIMEOUT_S=1200` (`getting_started_notebook.txt:19`) is the same 20-minute allowance passed to the server as `startup_timeout=60 * 20` (`getting_started_notebook.txt:287`). Both are 1200 s, i.e. engine-ready and client-side startup timeout agree.
- Memory profiling: `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` (`getting_started_notebook.txt:20`) is set together with `gpu_memory_utilization=0.90` (`getting_started_notebook.txt:280`); estimating CUDA-graph capture in the profiler reduces the chance of OOM at that utilization. `[INFERENCE]`
- Allocator: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (`getting_started_notebook.txt:27`) reduces fragmentation across the vLLM process and the sandbox work. `[INFERENCE]`
- `OPENAI_API_KEY`/`LOCAL_API_KEY` are preserved-if-present, defaulted to `EMPTY` (`getting_started_notebook.txt:25-26`); `LOCAL_API_KEY` is then read as `INFERENCE_API_KEY` (`getting_started_notebook.txt:243`) and handed to the model registry (`getting_started_notebook.txt:298`).
- `configure_logging()` (`getting_started_notebook.txt:32-53`) sets 10 logger names to `WARNING` (`getting_started_notebook.txt:34-47`); for any name containing `litellm` it also sets `propagate = False` and clears handlers (`getting_started_notebook.txt:48-50`). It is called at `getting_started_notebook.txt:53` and again after wheel install at `getting_started_notebook.txt:93`.
- `NCCL_DEBUG=WARN` (`getting_started_notebook.txt:21`) and the OTEL disables (`getting_started_notebook.txt:28-29`) are log-volume controls; no behavior claim beyond that is made in the notebook.

## 2. Wheelhouse, data paths, model paths, allowed models

Wheelhouse install (`getting_started_notebook.txt:55-94`):
- Source dir: `WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')` (`getting_started_notebook.txt:55`).
- Broken cutlass `.pth` hooks are deleted first, matching `python*/dist-packages` and `python*/site-packages` (`getting_started_notebook.txt:58-66`); `OSError` is swallowed (`getting_started_notebook.txt:63-66`).
- Kaggle uploads strip the PEP 440 local version tag, so `+cu128` is restored in filenames: any `*.whl` whose name contains `cu128` but not `+` gets `cu128` → `+cu128` (`getting_started_notebook.txt:68-78`).
- cutlass wheels are skipped from the staging dir (`getting_started_notebook.txt:71-73`).
- Staging dir is `/tmp/wheelhouse` (`getting_started_notebook.txt:69`); each wheel is symlinked, falling back to `shutil.copyfile` on `OSError` (`getting_started_notebook.txt:79-84`). Existing targets are not rewritten (`getting_started_notebook.txt:80`).
- Install command: `pip install -q --no-deps --force-reinstall <all staged wheels>` with `check=True` (`getting_started_notebook.txt:88-91`), then `importlib.invalidate_caches()` (`getting_started_notebook.txt:92`).
- Observed output: `Installing 41 wheels from /kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse...` (`getting_started_notebook.txt:96`) and `Wheelhouse installation complete.` (`getting_started_notebook.txt:97`). So the staged set is 41 wheels, cutlass excluded.
- `--no-deps` means the wheelhouse must be self-contained; `--force-reinstall` overrides the image's preinstalled versions (`getting_started_notebook.txt:89`).

Paths:
- `DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')` (`getting_started_notebook.txt:104`).
- `WORKING_DIR = Path('/kaggle/working')`, created with `mkdir(parents=True, exist_ok=True)` (`getting_started_notebook.txt:105-106`).
- `TASKS_PATH = DATA_DIR / 'tasks.jsonl'` (`getting_started_notebook.txt:133`).
- `GRAPH_DIR = str(DATA_DIR / 'graphs')` (`getting_started_notebook.txt:156`), `EMBEDDINGS_DIR = str(DATA_DIR / 'embeddings')` (`getting_started_notebook.txt:157`) — both coerced to `str`.
- Eval-time `snapshots_dir = DATA_DIR / 'snapshots'` (`getting_started_notebook.txt:358`), `results_dir = WORKING_DIR / 'results'` (`getting_started_notebook.txt:359`), `wheels_dir = DATA_DIR / 'wheels'` (`getting_started_notebook.txt:380`).
- Model weights: `MODEL_PATH = Path(f'/kaggle/input/models/google/gemma-4/other/{declared_model}/2')` (`getting_started_notebook.txt:253`), asserted to exist (`getting_started_notebook.txt:254`). The `/2` segment is a Kaggle model-version directory `[INFERENCE]` — the notebook does not explain it.

Allowed model list — the four official quantized checkpoints:
- `gemma-4-31b-it-qat-w4a16-ct` (`getting_started_notebook.txt:109`, `getting_started_notebook.txt:231`)
- `gemma-4-12b-it-qat-w4a16-ct` (`getting_started_notebook.txt:110`, `getting_started_notebook.txt:232`)
- `gemma-4-e4b-it-qat-w4a16-ct` (`getting_started_notebook.txt:111`, `getting_started_notebook.txt:233`)
- `gemma-4-e2b-it-qat-w4a16-ct` (`getting_started_notebook.txt:112`, `getting_started_notebook.txt:234`)
- Enforced as `ALLOWED_MODEL_NAMES = frozenset({...})` (`getting_started_notebook.txt:230-235`) and checked with `assert declared_model in ALLOWED_MODEL_NAMES` (`getting_started_notebook.txt:251`).
- 31B is called out as "default in sample_submission" (`getting_started_notebook.txt:109`).

Model-specific vLLM args:
- `MODEL_VLLM_EXTRA_ARGS` has exactly one entry, keyed on `gemma-4-12b-it-qat-w4a16-ct` (`getting_started_notebook.txt:236-241`): `('--hf-overrides', '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')` (`getting_started_notebook.txt:238-239`).
- Applied via `extra_args=MODEL_VLLM_EXTRA_ARGS.get(declared_model, ())` (`getting_started_notebook.txt:288`) — the other three models get no extra args.

Dataset facts:
- `tasks = load_tasks(TASKS_PATH)` from `swegemma.models` (`getting_started_notebook.txt:102`, `getting_started_notebook.txt:134`).
- Observed: `Loaded 129 tasks from tasks.jsonl` (`getting_started_notebook.txt:144`).
- First five instance ids / repos / base commits: `fastapi_15661 @ ee22a4b8`, `fastapi_15588 @ cb83b83d`, `fastapi_15589 @ 22b02e26`, `fastapi_15030 @ 48d58ae3`, `fastapi_14962 @ 1e78a36b` (`getting_started_notebook.txt:145-149`).

Code-graph API used (cell 3):
- `sg.get_graph(repo_name=..., graph_dir=GRAPH_DIR, embeddings_dir=EMBEDDINGS_DIR, base_commit=...)` (`getting_started_notebook.txt:160-165`).
- `sg.get_neighbor(node=..., graph=..., max_neighbors=10)` (`getting_started_notebook.txt:172-176`).
- `sg.get_similar_nodes(node=..., repo_name=..., k=5, graph=..., graph_dir=..., embeddings_dir=..., base_commit=...)` (`getting_started_notebook.txt:182-190`).
- `sg.get_induced_subgraph(repo_graph, focal_nodes)` (`getting_started_notebook.txt:197`).
- Observed for `fastapi_15661`: `Nodes: 4263, Edges: 2430` (`getting_started_notebook.txt:203-204`); top similarity `1.0000` (`getting_started_notebook.txt:212`); induced subgraph over 4 focal nodes → 4 nodes, 6 edges (`getting_started_notebook.txt:218`).
- Neighbor query node selection: first node with degree ≥ 5, else the first node (`getting_started_notebook.txt:171`).

## 3. Exact vLLM configuration

Constructed as `VllmConfig(...)` at `getting_started_notebook.txt:272-289`:

| Field | Value | Line |
|---|---|---|
| `model` | `str(MODEL_PATH)` (`/kaggle/input/models/google/gemma-4/other/<declared_model>/2`) | `getting_started_notebook.txt:273` (path at `getting_started_notebook.txt:253`) |
| `port` | `8000` | `getting_started_notebook.txt:274` |
| `host` | `'127.0.0.1'` | `getting_started_notebook.txt:275` |
| `tool_call_parser` | `'gemma4'` | `getting_started_notebook.txt:276` |
| `reasoning_parser` | `'gemma4'` | `getting_started_notebook.txt:277` |
| `max_model_len` | `32768` | `getting_started_notebook.txt:278` |
| `dtype` | `'bfloat16'` if CUDA available and bf16 supported, else `'auto'` | `getting_started_notebook.txt:279` |
| `gpu_memory_utilization` | `0.90` | `getting_started_notebook.txt:280` |
| `enable_auto_tool_choice` | `True` | `getting_started_notebook.txt:281` |
| `enable_lora` | `bool(adapter_map)` | `getting_started_notebook.txt:282` |
| `max_loras` | `max(1, len(adapter_map))` | `getting_started_notebook.txt:283` |
| `max_lora_rank` | `max(detected_ranks)` if any positive rank, else `128` | `getting_started_notebook.txt:284` |
| `scheduler_reserve_full_isl` | `False` | `getting_started_notebook.txt:285` |
| `tensor_parallel_size` | `tp_size` | `getting_started_notebook.txt:286` |
| `startup_timeout` | `60 * 20` (= 1200 s) | `getting_started_notebook.txt:287` |
| `extra_args` | per-model tuple, `()` for non-12B | `getting_started_notebook.txt:288` |

- Tensor parallel is computed, not fixed: `gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1`; `tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)` (`getting_started_notebook.txt:266-267`).
- Adapter ranks come from `discover_adapters(...)` → `adapter_map` → `detected_ranks` = positive `info.rank` values (`getting_started_notebook.txt:256-264`).
- `max_lora_rank` is **not** clamped to 128 (`getting_started_notebook.txt:284`); 128 is only the fallback when no positive rank is detected.
- Any previously running server is stopped before reconfiguration: `if 'server_instance' in globals() and server_instance is not None: server_instance.stop()` (`getting_started_notebook.txt:269-270`).
- Server start: `VllmServer(vllm_cfg, adapter_manifest=adapters)` then `.start()` (`getting_started_notebook.txt:290-291`).
- Observed: `vLLM server started on http://127.0.0.1:8000/v1 (tp=4)` (`getting_started_notebook.txt:301`) — so the run had ≥ 4 visible GPUs.
- Registry: `server_instance.create_model_registry(aliases=[declared_model], model_prefix='openai/', api_key=INFERENCE_API_KEY)` (`getting_started_notebook.txt:295-299`).
- `litellm.drop_params = True` is set before any call (`getting_started_notebook.txt:228`).

## 4. Sample submission handling

- `SAMPLE_SUBMISSION_SRC = DATA_DIR / 'sample_submission'` (`getting_started_notebook.txt:116`); copied to `AGENT_DIR = WORKING_DIR / 'sample_submission'` (`getting_started_notebook.txt:117`).
- Copy is idempotent: `rmtree` if present, then `copytree` (`getting_started_notebook.txt:118-120`).
- Model switch is a text rewrite of the YAMLs, not a re-download (`getting_started_notebook.txt:123-130`):
  - Applies only when `MODEL_NAME != 'gemma-4-31b-it-qat-w4a16-ct'` (`getting_started_notebook.txt:123`).
  - Targets `AGENT_DIR/agent.yaml` plus every `AGENT_DIR/sub_agents/*.yaml`, sorted (`getting_started_notebook.txt:124`).
  - `model:` lines are rewritten in place: regex `^(\s*model:\s*).*$` → same prefix + `MODEL_NAME`, `MULTILINE` (`getting_started_notebook.txt:127`).
  - `adapter:` lines are deleted entirely: `^\s*adapter:\s*.*\n?` → `''` (`getting_started_notebook.txt:128`).
  - The whole `AGENT_DIR/adapters` tree is removed (`getting_started_notebook.txt:130`), because the shipped sample LoRA adapters are shaped for 31B (`getting_started_notebook.txt:122`, `getting_started_notebook.txt:246-247`).
- Declared-model validation (`getting_started_notebook.txt:248-251`): `discover_declared_models(AGENT_DIR)` must yield exactly one model (`getting_started_notebook.txt:249`), and that model must be in `ALLOWED_MODEL_NAMES` (`getting_started_notebook.txt:251`). Both are `assert`s with `sorted(...)` in the message.
- The validated declared model is what selects the weights path (`getting_started_notebook.txt:253`), the registry alias (`getting_started_notebook.txt:296`), and the extra-args lookup (`getting_started_notebook.txt:288`) — the notebook's `MODEL_NAME` variable is only used for the YAML rewrite.
- Adapter discovery: `discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)` (`getting_started_notebook.txt:256`), with `ALLOWED_ADAPTER_EXTENSIONS` imported from `swegemma.config` (`getting_started_notebook.txt:226`).
- Adapter names must match `_SAFE_PATH_COMPONENT = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')` (`getting_started_notebook.txt:242`), asserted per name (`getting_started_notebook.txt:258-261`).
- The discovered manifest is passed both to the server (`adapter_manifest=adapters`, `getting_started_notebook.txt:290`) and to `EvalConfig` (`getting_started_notebook.txt:370`).

## 5. Local evaluation

Config loading (`getting_started_notebook.txt:331-348`):
- Looks for `eval_config.yaml` then `eval_config.yml` in `AGENT_DIR`; first existing wins (`getting_started_notebook.txt:332-335`).
- `yaml.safe_load(...) or {}` (`getting_started_notebook.txt:336-340`); the budget block is `raw_eval_cfg.get('evaluation', raw_eval_cfg)` — top-level keys also accepted (`getting_started_notebook.txt:341`).
- Fallback defaults when a key is absent: `timeout_seconds=300` (`getting_started_notebook.txt:343`), `verification_timeout_seconds=300` (`getting_started_notebook.txt:344`), `max_tool_calls=100` (`getting_started_notebook.txt:345`), `max_time_minutes=60.0` (`getting_started_notebook.txt:346`), `max_turns` from `max_turns` or `max_llm_calls`, else `None` (`getting_started_notebook.txt:347-348`).

Task subset:
- `SAMPLE_TASKS = tasks[:2]` (`getting_started_notebook.txt:351`) — the notebook evaluates **2 of 129** tasks.
- `limits, gen_constraints = build_submission_limits()` (`getting_started_notebook.txt:352`), fed into `EvalConfig` (`getting_started_notebook.txt:368-369`).

`EvalConfig` fields (`getting_started_notebook.txt:356-382`):
- `tasks_path=TASKS_PATH` (`getting_started_notebook.txt:357`), `snapshots_dir=DATA_DIR/'snapshots'` (`getting_started_notebook.txt:358`), `results_dir=WORKING_DIR/'results'` (`getting_started_notebook.txt:359`), `submission_dir=AGENT_DIR` (`getting_started_notebook.txt:360`), `models=models` (`getting_started_notebook.txt:361`).
- `sandbox='subprocess'` (`getting_started_notebook.txt:362`) — explicitly the subprocess backend, not Docker.
- `timeout_seconds`, `verification_timeout_seconds`, `max_time_minutes`, `max_tool_calls`, `max_turns` from the block above (`getting_started_notebook.txt:363-367`).
- `limits=limits`, `generation_constraints=gen_constraints` (`getting_started_notebook.txt:368-369`), `adapter_manifest=adapters` (`getting_started_notebook.txt:370`).
- `verbose=False` (`getting_started_notebook.txt:381`), `graph_dir=GRAPH_DIR`, `embeddings_dir=EMBEDDINGS_DIR`, `wheels_dir=DATA_DIR/'wheels'` (`getting_started_notebook.txt:378-380`).

Compaction / cache config (`getting_started_notebook.txt:371-377`):
- `ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10)` (`getting_started_notebook.txt:371`).
- `EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5)` (`getting_started_notebook.txt:372-377`).
- Both are ADK experimental features; warnings emitted at `getting_started_notebook.txt:408-411` (`FeatureName.AGENT_CONFIG`, and "EventsCompactionConfig: This feature is experimental…").
- `run_sync` (`getting_started_notebook.txt:315-328`) drives the async `evaluate_task` from the notebook loop: if a loop is already running it uses a 1-worker `ThreadPoolExecutor` + `asyncio.run` (`getting_started_notebook.txt:325-327`), otherwise plain `asyncio.run` (`getting_started_notebook.txt:328`).

Run loop (`getting_started_notebook.txt:387-406`):
- Sequential per task: `evaluator.evaluate_task(task=..., task_index=idx, total_tasks=len(SAMPLE_TASKS))` (`getting_started_notebook.txt:389-394`).
- Empty patch becomes the literal string `NO_PATCH` (`getting_started_notebook.txt:395-396`); DataFrame columns are `['id', 'prediction']` (`getting_started_notebook.txt:405`).

Observed results for the sample run:
- Sandbox warning: `ensurepip unavailable in sandbox 6d7ab3b3-55b, creating venv without pip` (`getting_started_notebook.txt:412`); repeated for the second task's sandbox `110eb980-31d` (`getting_started_notebook.txt:514`).
- Task 1 `fastapi_15661`: hit `BudgetExceeded` — `Tool call budget exhausted (10 calls)` (`getting_started_notebook.txt:491-492`, echoed in detail pane `getting_started_notebook.txt:500-502`); dashboard shows `Tools 10 / 10 (0 left)` and `Time 1.0m / 1m (0.0m left)` (`getting_started_notebook.txt:507-509`); `Task fastapi_15661 agent timed out.` (`getting_started_notebook.txt:512`); `completed without explicit submit_patch call and no working tree modifications` (`getting_started_notebook.txt:513`); result `resolved=False, exit_code=-1, patch_chars=0, tool_calls=10, duration=70.2s` (`getting_started_notebook.txt:516`).
- Task 2 `fastapi_15588`: dashboard `Tools 5 / 10 (5 left)`, `Time 1.0m / 1m` (`getting_started_notebook.txt:571-575`); same timeout + no-submit warnings (`getting_started_notebook.txt:578-579`); result `resolved=False, exit_code=-1, patch_chars=0, tool_calls=5, duration=67.3s` (`getting_started_notebook.txt:581`).
- Final table: both rows `NO_PATCH` (`getting_started_notebook.txt:583-585`).
- Net: 0/2 resolved on the 12B model with the sample submission's budgets.

## 6. Packaging

- Imports: `zipfile`, `validate_directory` from `adk_submission`, `ALLOWED_SUBMISSION_EXTENSIONS` and `MAX_SUBMISSION_SIZE_BYTES` from `swegemma.config` (`getting_started_notebook.txt:588-590`).
- Symlink rejection is a manual pre-pass: `os.walk(AGENT_DIR, followlinks=False)`, asserting no entry is a symlink, message names the relative path (`getting_started_notebook.txt:593-596`).
- Structural validation: `validate_directory(AGENT_DIR, limits=limits)` (`getting_started_notebook.txt:597`), reusing the same `limits` object built at `getting_started_notebook.txt:352`.
- Archive creation: `shutil.make_archive(str(WORKING_DIR / 'submission'), 'zip', root_dir=AGENT_DIR)` (`getting_started_notebook.txt:600-601`) — `root_dir=AGENT_DIR` puts `agent.yaml` at the archive root.
- Size check: `total_size = sum(i.file_size for i in zf.infolist())`, asserted `<= MAX_SUBMISSION_SIZE_BYTES`, message says `Archive exceeds 3 GiB limit` (`getting_started_notebook.txt:606-607`). Because it sums uncompressed `file_size`, the check is on unpacked size, not zip bytes.
- Extension check: every non-dir member's `Path(filename).suffix.lower()` must be in `ALLOWED_SUBMISSION_EXTENSIONS` (`getting_started_notebook.txt:608-611`).
- Output line: `Created {zip_path} ({size MiB})` (`getting_started_notebook.txt:613`); observed `Created /kaggle/working/submission.zip (0.00 MiB)` (`getting_started_notebook.txt:615`).
- The 0.00 MiB is consistent with an all-text submission after `adapters/` was deleted at `getting_started_notebook.txt:130`. `[INFERENCE]`

# Discrepancies

1. **Sandbox backend.** Notebook pins `sandbox='subprocess'` (`getting_started_notebook.txt:362`). README documents Docker as the default (`harness_readme_raw.md:233`, `harness_readme_raw.md:589`, `harness_readme_raw.md:599`) and describes the subprocess backend as the fallback "used in environments without a Docker daemon" (`harness_readme_raw.md:240`). Kaggle notebooks have no Docker daemon, so the notebook is the no-Docker path `[INFERENCE]` — the two documents describe different backends for the same `EvalConfig` field.
2. **Tensor parallel.** Notebook derives TP from visible GPU count: 4 / 2 / 1 (`getting_started_notebook.txt:266-267`), observed 4 (`getting_started_notebook.txt:301`). README states a fixed `tensor_parallel_size = 4` (`harness_readme_raw.md:148`). On the 4×L4 scorer these agree; on a 2-GPU or CPU-only Kaggle session the notebook silently drops to 2 or 1, which the README does not describe.
3. **Budget defaults.** Notebook code defaults are 100 tool calls / 60.0 minutes / no turn cap (`getting_started_notebook.txt:345-348`). README lists 100 / 60.0 / 500 as the `inference.py` defaults and 10 / 1 / 50 as the **sample submission** values (`harness_readme_raw.md:500-502`, `harness_readme_raw.md:493-497`). The observed run enforced 10 calls and a 1-minute wall clock (`getting_started_notebook.txt:507-509`, `getting_started_notebook.txt:491`, `getting_started_notebook.txt:575`), which matches the sample-submission values — i.e. the notebook's fallbacks were overridden by the shipped `eval_config.yaml`. `[INFERENCE]` on the file contents; the enforced numbers are observed.
4. **`max_turns` when unset.** Notebook passes `max_turns=None` if neither `max_turns` nor `max_llm_calls` is present (`getting_started_notebook.txt:347-348`). README states `None → 500` (`harness_readme_raw.md:502`). The notebook does not reproduce that mapping.
5. **GPU utilization.** Both say `0.90` — notebook `getting_started_notebook.txt:280`, README `harness_readme_raw.md:149`. Not a discrepancy; noting it because it is a commonly misremembered value.
6. **`dtype` and `startup_timeout` are notebook-only.** `dtype='bfloat16'|'auto'` (`getting_started_notebook.txt:279`) and `startup_timeout=60*20` (`getting_started_notebook.txt:287`) appear in no README serving bullet (`harness_readme_raw.md:147-154`). README's only `bfloat16` mention is the unquantized local-CLI model table (`harness_readme_raw.md:160`).
7. **`reasoning_config` missing from the notebook.** README documents `reasoning_config = {"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}` as part of the serving config (`harness_readme_raw.md:152`). The notebook's `VllmConfig(...)` passes only `reasoning_parser='gemma4'` (`getting_started_notebook.txt:277`) and no `reasoning_config` — either it defaults inside `VllmConfig` or the notebook's config is less complete than the README's. Not established by the notebook.
8. **`max_lora_rank` ceiling.** README treats 128 as the maximum (`harness_readme_raw.md:166`, `harness_readme_raw.md:201`). Notebook uses `max(detected_ranks)` with no clamp (`getting_started_notebook.txt:284`), so a submitted adapter with rank > 128 would produce a rank > 128.
9. **`VLLM_ENGINE_READY_TIMEOUT_S` / startup timeout.** Notebook sets 1200 s in both places (`getting_started_notebook.txt:19`, `getting_started_notebook.txt:287`). README does not document any startup timeout value (`harness_readme_raw.md:147-154`).
10. **Wheelhouse install is notebook-only.** The 41-wheel `/tmp/wheelhouse` staging and `--no-deps --force-reinstall` install (`getting_started_notebook.txt:55-96`) have no README equivalent; README covers only in-sandbox wheel staging (`harness_readme_raw.md:247`, `harness_readme_raw.md:262`). The notebook's approach re-installs host packages for the notebook kernel; the scorer installs wheels inside its sandbox instead (`[INFERENCE]` that the notebook install is unnecessary for scoring).
11. **Submission-size semantics.** Notebook checks `sum(i.file_size)` over zip members against `MAX_SUBMISSION_SIZE_BYTES` with the message "3 GiB limit" (`getting_started_notebook.txt:606-607`). README defines the limit as "< 3 GiB (3,221,225,472 bytes) total unpacked size" (`harness_readme_raw.md:115`, `harness_readme_raw.md:120`). Semantically the same (uncompressed member sizes = unpacked size); the exact constant is imported, not inlined.
12. **Model path shape.** Notebook hardcodes the Kaggle mount `/kaggle/input/models/google/gemma-4/other/<model>/2` (`getting_started_notebook.txt:253`); README never states a weights path (`harness_readme_raw.md:147-159`).
13. **Task subset.** Notebook scores 2 tasks (`getting_started_notebook.txt:351`) out of 129 (`getting_started_notebook.txt:144`); README's local CLI examples select tasks by `--task-id`/`--task-ids` (`harness_readme_raw.md:598`) and the competition metric is over the full evaluation split (`harness_readme_raw.md:568-572`). A 2-task sample is not representative of the metric.

# Needs external verification

- Contents of `sample_submission/eval_config.yaml` (the 10-call / 1-minute budgets are observed at `getting_started_notebook.txt:507-509` but the file itself is not in this repo).
- Numeric value of `MAX_SUBMISSION_SIZE_BYTES` and the exact `ALLOWED_SUBMISSION_EXTENSIONS` set (`getting_started_notebook.txt:590`, used `getting_started_notebook.txt:607`, `getting_started_notebook.txt:611`) — imported from `swegemma.config`, source not present locally.
- `build_submission_limits()` return values (`getting_started_notebook.txt:352`).
- `VllmConfig` field defaults that the notebook does not override — especially `reasoning_config` (README `harness_readme_raw.md:152`) and how `host`/`dtype` are forwarded to the CLI.
- Whether `ALLOWED_ADAPTER_EXTENSIONS` is exactly `.safetensors` (README `harness_readme_raw.md:131`).
- Adapter ranks actually present in `sample_submission/adapters/` (drives `max_lora_rank` at `getting_started_notebook.txt:284`).
- Whether the scoring harness (`scripts/inference.py`) uses the same `startup_timeout` / `VLLM_ENGINE_READY_TIMEOUT_S` as the notebook (`getting_started_notebook.txt:19`, `getting_started_notebook.txt:287`).
- The Kaggle notebook's actual accelerator (tp=4 observed at `getting_started_notebook.txt:301` implies ≥4 GPUs, but the machine spec is not stated).
- What model version `/2` in the mount path denotes (`getting_started_notebook.txt:253`).
- Whether `ensurepip`-less venvs (`getting_started_notebook.txt:412`, `getting_started_notebook.txt:514`) change Phase 2 verification behavior; the notebook only logs the warning.
- Full `swegemma.graph` API surface beyond the four functions exercised in cell 3 (`getting_started_notebook.txt:160-197`).
