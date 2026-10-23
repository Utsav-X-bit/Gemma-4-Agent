# Source Ledger

Last verified: **2026-10-24**.

This file is the evidence base for the playbook. Every non-obvious claim in the other docs should point here with `[S#]`. Sources are classified:

- **OFFICIAL** — competition page, rules, host post, model card, vendor documentation.
- **LOCAL** — the two files supplied in this workspace: `harness_readme_raw.md` and `getting_started_notebook.txt`.
- **COMMUNITY** — public notebooks, discussion posts, independent experiments. Useful but not authoritative.
- **PAPER** — peer-reviewed/preprint research. Directional, not competition-specific.
- **ESTIMATE** — a planning number derived from the cited facts; not measured on the scorer.

When official and community sources conflict, the official source wins. When a community number is a single run, it is labeled as such. Competition state changes; re-check the live pages before spending a submission.

---

## S1 — Competition overview (OFFICIAL)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/overview
- Retrieved: 2026-10-24.
- Facts:
  - Start: 2026-09-23. Paper deadline: 2026-11-12. Entry/team-merger deadline: 2026-11-25. Final submission: 2026-12-02. All at 11:59 PM UTC unless noted.
  - Total prizes: **$110,000**. Main competition: **$65,000** (1st $37k, 2nd $18k, 3rd $10k). Paper track: **$35,000**. Small-model bonus: **$10,000**.
  - Submission is `submission.zip` containing an Agent Config; `agent.yaml` at the archive root.
  - Score: percentage of patched repositories whose validation tests pass.
  - 12-hour limit for submitting patches for all tasks, inclusive of sandbox setup, excluding patch validation.
  - Allowed models: `gemma-4-31b-it-qat-w4a16-ct`, `gemma-4-12b-it-qat-w4a16-ct`, `gemma-4-e4b-it-qat-w4a16-ct`, `gemma-4-e2b-it-qat-w4a16-ct`; one base model per submission.
  - Predefined tools: `run_command`, `submit_patch`, `get_status`, `read_file`, `edit_file`, `write_file`, `get_code_neighbors`, `search_similar_code`, `get_code_subgraph`.
  - L4x4 machines: 96 GB total GPU memory; internet disabled; L4x4 consumes GPU quota at 2× the older T4x2/P100 rate.
  - Scale at retrieval: 12,493 entrants, 3,959 participants, 3,548 teams, 22,065 submissions.
  - Submissions page: if fewer than two final submissions are selected, Kaggle automatically selects from the team’s best scoring submissions.

## S2 — Competition data page (OFFICIAL)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/data
- Retrieved: 2026-10-24.
- Facts:
  - Public training set: **129 tasks** from four repositories (`fastapi/fastapi`, `Textualize/rich`, `psf/requests`, `encode/httpx`).
  - Dataset size: **22.42 GB**, 524 files.
  - `embeddings/`: each node is a float32 vector of length **256** (the data page says “float32 NumPy array of length 256”).
  - `tasks.jsonl`: `instance_id`, `repo`, `base_commit`, `problem_statement`, `hints_text`, `patch`, `test_patch`, `created_at`.
  - `snapshots/`: 129 compressed Git working trees at `base_commit`, future history removed.
  - `graphs/` and `embeddings/`: 256 files each; AST call/dependency graphs and 256-d node embeddings.
  - `wheels/`: 124 pre-compiled offline wheels.
  - `docker/`: `Dockerfile.sandbox`, `Dockerfile.public`, `imp.py`, `telnetlib.py`; `sandbox/setup.py`; `sample_submission/`; `HARNESS_README.md`; `submission.parquet` schema.
  - Hidden test set: **about 120 tasks**, evenly divided between public and private splits; curated from **private repositories**.
  - Test-set validation: fail-to-pass baseline, pass-to-pass solution, and a frontier-model near-completion filter.

## S3 — Competition rules (OFFICIAL)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/rules
- Retrieved: 2026-10-24.
- Facts:
  - Maximum team size: 5.
  - Maximum **one submission per day**; up to **two final submissions** selected for judging.
  - External data is allowed, but it must be publicly available and equally accessible to all participants at no cost, or satisfy the Reasonableness Standard. External models are allowed unless specifically prohibited by the Host.
  - Winner license: **Apache 2.0** for the winning submission and source code used to generate it.
  - Winners must provide the final model software code, training code, inference code, and a reproducible description of methodology/hyperparameters/compute.
  - Competition Data may be used for any purpose including commercial/academic, but may not be redistributed to non-participants.

## S4 — Leaderboard (OFFICIAL)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/leaderboard
- Retrieved: 2026-10-24.
- Facts:
  - Public leaderboard is calculated with approximately **50%** of the test data; final results use the other 50%.
  - Top public score: **0.24** (four teams shared it at retrieval); a large cluster at **0.22**; then 0.20.
  - Scores are displayed to two decimals; one task is roughly 1/58 of the public split (inferred from the public split size, not an official denominator).

## S5 — Harness README (LOCAL, authoritative technical spec)

- File: `harness_readme_raw.md` (627 lines). Retrieved from the competition dataset on 2026-10-23/24.
- This is the primary source for:
  - submission contract, root config discovery, `!include`, limits;
  - agent schema (`LlmAgent`, `SequentialAgent`, `ParallelAgent`, `LoopAgent`);
  - model registry, single-model rule, vLLM serving, LoRA adapters;
  - two-phase evaluation lifecycle, container bootstrap, sandbox limits;
  - session state, prompt construction, nudges, tool error recovery;
  - all nine tools and their exact behavior;
  - budgets, compaction, caching, retries;
  - patch extraction, protected-file stripping, Phase 2 verification, JUnit criterion;
  - local CLI and result artifacts;
  - best practices and gotchas.
- Line references are used throughout the docs as `[S5:line]`.
- Important absence: the README does **not** state the global 12-hour limit; that comes from S1 and host/participant reports. It does not define `_swegemma_baseline` creation, the Phase-2 entry boolean, or the zip top-level layout unambiguously.

## S6 — Getting Started notebook (LOCAL)

- File: `getting_started_notebook.txt` (615 lines). Retrieved 2026-10-23.
- Facts:
  - Installs 41 wheels from `/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse` with `pip install --no-deps --force-reinstall`.
  - Sets `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`, `VLLM_ENGINE_READY_TIMEOUT_S=1200`, `TRANSFORMERS_OFFLINE=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - `VllmConfig`: `max_model_len=32768`, `gpu_memory_utilization=0.90`, `enable_auto_tool_choice=True`, `tool_call_parser='gemma4'`, `reasoning_parser='gemma4'`, `startup_timeout=1200`, TP=4 on ≥4 GPUs.
  - Local eval uses `sandbox='subprocess'`, 2 tasks, sample submission budgets (observed 1 minute / 10 calls), compaction threshold 14,336, context cache 2,048 tokens/1,800 s.
  - Observed sample run: 0/2 resolved, both `NO_PATCH`; task 1 hit 10/10 calls and 1.0/1 minute; task 2 hit 5/10 calls and 1.0/1 minute.
  - Packaging: `validate_directory`, symlink rejection, zip root = agent dir, unpacked size ≤ 3 GiB, allowed extension check.
  - Model-specific extra args only for 12B: `--hf-overrides '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'`.

## S7 — Gemma 4 model overview (OFFICIAL)

- URL: https://ai.google.dev/gemma/docs/core
- Retrieved: 2026-10-24.
- Facts:
  - Gemma 4 family: E2B, E4B, 12B, 26B A4B, 31B.
  - Approximate inference memory: E2B 11.4 GB BF16 / 2.9 GB Q4_0; E4B 17.9 / 4.5; 12B 26.7 / 6.7; 26B A4B 57.7 / 14.4; 31B 69.9 / 17.5 GB.
  - QAT compressed-tensors (`-w4a16-ct`) is the vLLM/SGLang deployment format; `-qat-q4_0-unquantized` is the half-precision QAT checkpoint for conversion/research.
  - QAT integrates quantization simulation into training; QAT checkpoints preserve quality better than post-training quantization.
  - Context: 128K for E2B/E4B; 256K for 12B/26B/31B (competition caps at 32,768).

## S8 — Hugging Face QAT model card (OFFICIAL)

- URL: https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct
- Retrieved: 2026-10-24.
- Facts:
  - License: Apache 2.0.
  - 31B dense: 30.7B total parameters, 60 layers, 1024-token sliding window, 256K context, 262K vocab.
  - Recommended sampling: `temperature=1.0`, `top_p=0.95`, `top_k=64`.
  - Thinking: enabled by `<|think|>` at the start of the system prompt; output uses `<|channel>thought\n...<channel|>`.
  - Multi-turn: historical model output should include only the final response; thoughts from previous turns should not be added, except tool-call turns where thinking content should be preserved.
  - QAT compressed-tensors variants are for vLLM; unquantized QAT checkpoints are for custom compilation/research.

## S9 — Gemma 4 Technical Report (PAPER)

- URL: https://arxiv.org/abs/2607.02770
- Retrieved: 2026-10-24.
- Facts: Gemma 4 is a natively multimodal family from 2.3B to 31B, with dense and MoE variants, thinking mode, improved inference/memory/long-context efficiency. Use for background only; not competition-specific.

## S10 — Unsloth Gemma 4 training guide (OFFICIAL vendor)

- URL: https://unsloth.ai/docs/models/gemma-4/train
- Retrieved: 2026-10-24.
- Facts:
  - Unsloth supports Gemma 4 12B, E2B, E4B, 26B-A4B, 31B.
  - Stated memory: **31B QLoRA works with ~22 GB**; E4B LoRA ~17 GB; E2B LoRA ~8–10 GB.
  - 31B QLoRA requires an A100-class GPU for the larger models; E2B/E4B can train on much less.
  - Loss note: Gemma 26B and 31B typically show loss around 1–3 or lower; E2B/E4B can show 13–15, which is normal for these multimodal models.
  - `FastModel.get_peft_model(...)`: `finetune_vision_layers=False`, `finetune_language_layers=True`, `finetune_attention_modules=True`, `finetune_mlp_modules=True`, `r=8`, `lora_alpha=8`, `lora_dropout=0`, `bias="none"`.
  - Use `gemma-4-thinking` chat template for 31B/26B; `gemma-4` for small models.
  - `train_on_responses_only` masks instruction tokens; response marker `<|turn>model\n`.
  - Multi-GPU support is via Accelerate/DeepSpeed/DDP; official multi-GPU support is still maturing.

## S11 — Axolotl Gemma 4 guide (OFFICIAL vendor)

- URL: https://docs.axolotl.ai/docs/models/gemma4.html
- Retrieved: 2026-10-24.
- Facts:
  - **31B Dense QLoRA on 1×80 GB at ~25.2 GiB** (example config; short sequence, packing).
  - **31B Dense LoRA FSDP2, 32k sequences (multi-GPU)**.
  - E2B vision LoRA on 1×80 GB at ~10.4 GiB.
  - Flash Attention 2/4 cannot serve Gemma 4's `global_head_dim=512` alone; use `flex_attention` or `gemma4_hybrid_attn_impl: true` (sliding layers FA2, global layers SDPA).
  - LoRA kernels are not supported for models with KV-sharing layers.
  - `lora_target_linear` is incompatible with multimodal models; use a regex restricted to the text backbone.
  - Gradient spikes: Gemma 4 text attention uses `scaling=1.0` with QK-RMSNorm; LoRA runs hit intermittent pre-clip gradient norms 100–1000× their neighbors; 31B configs use `learning_rate: 5e-5` and `max_grad_norm: 0.1`.
  - `gemma4_hybrid_attn_impl: true` trains ~2× faster than `flex_attention` on 31B.

## S12 — Axolotl 31B QLoRA config (OFFICIAL vendor)

- URL: https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/examples/gemma4/31b-qlora.yaml
- Retrieved: 2026-10-24.
- Facts: `base_model: google/gemma-4-31B`, `chat_template: gemma4`, `eot_tokens: ["<turn|>"]`, `load_in_4bit: true`, `adapter: qlora`, `lora_r: 16`, `lora_alpha: 32`, `lora_dropout: 0`, text-backbone target regex `model.language_model.layers.[\d]+.(_checkpoint_wrapped_module.)?(mlp|self_attn).(up|down|gate|q|k|v|o)_proj`, `sequence_len: 2048`, `sample_packing: true`, `micro_batch_size: 1`, `gradient_accumulation_steps: 4`, `num_epochs: 1`, `learning_rate: 0.00005`, `max_grad_norm: 0.1`, `bf16: auto`, `tf32: true`, `gradient_checkpointing: true`, `activation_offloading: true`, `gemma4_hybrid_attn_impl: true`, `attn_implementation: flash_attention_2`.

## S13 — Axolotl 31B LoRA FSDP config (OFFICIAL vendor)

- URL: https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/examples/gemma4/31b-lora-fsdp.yaml
- Retrieved: 2026-10-24.
- Facts: `adapter: lora`, `lora_r: 256`, `lora_alpha: 512`, `lora_dropout: 0.05`, `sequence_len: 32768`, `sample_packing: true`, `micro_batch_size: 1`, `gradient_accumulation_steps: 1`, `learning_rate: 0.00005`, `max_grad_norm: 0.1`, `fsdp_version: 2`, `transformer_layer_cls_to_wrap: Gemma4TextDecoderLayer`, `reshard_after_forward: true`, `state_dict_type: SHARDED_STATE_DICT`, `final_state_dict_type: FULL_STATE_DICT`, `activation_offloading: true`.
- The 256 rank in this example is not a recommendation for the competition; it is a memory/quality example.

## S14 — Axolotl FSDP + QLoRA docs (OFFICIAL vendor)

- URL: https://docs.axolotl.ai/docs/fsdp_qlora.html
- Retrieved: 2026-10-24.
- Facts:
  - FSDP + QLoRA requires `adapter: qlora`, FSDP enabled, and one of the supported model types: **llama, mistral, mixtral**. Gemma 4 is not listed as a supported FSDP+QLoRA model type in this doc.
  - FSDP2 is the only supported FSDP version in current Axolotl.
  - CPU offload/swap options exist for memory pressure.

## S15 — bitsandbytes FSDP-QLoRA docs (OFFICIAL vendor)

- URL: https://huggingface.co/docs/bitsandbytes/fsdp_qlora
- Retrieved: 2026-10-24.
- Facts:
  - FSDP-QLoRA stores quantized weights with `bnb_4bit_quant_storage=torch.bfloat16` so FSDP can shard them; `torch_dtype` must match.
  - `target_modules="all-linear"` is the QLoRA example.
  - PEFT notes: FSDP is not compatible with 8-bit bitsandbytes; multi-GPU QLoRA has `use_reentrant` constraints; normal LoRA + FSDP/DeepSpeed is preferred when memory allows.

## S16 — TRL SFTTrainer docs (OFFICIAL vendor)

- URL: https://huggingface.co/docs/trl/sft_trainer
- Retrieved: 2026-10-24.
- Facts:
  - `SFTTrainer` supports language-modeling and prompt-completion datasets.
  - `completion_only_loss=True` computes loss on completion tokens only.
  - `assistant_only_loss=True` requires the chat template to contain `{% generation %}` / `{% endgeneration %}` markers; for known families TRL may patch the template.
  - `packing=True` packs multiple examples; loss masks must be preserved.
  - PEFT integration: `peft_config=LoraConfig(...)`; adapter learning rates are typically higher (~1e-4).

## S17 — ATIF trajectory format (OFFICIAL vendor)

- URL: https://docs.harborframework.com/agents/atif
- Retrieved: 2026-10-24.
- Facts:
  - ATIF-v1.7 records `agent`, `steps`, `session_id`, `trajectory_id`, `final_metrics`, `subagent_trajectories`, `extra`.
  - Steps have `step_id`, `source` (system/user/agent), `message`, optional `tool_calls`, matching `observation`, and per-step `metrics`.
  - Tool calls have `tool_call_id`, `function_name`, `arguments`; observations have `source_call_id`, `content`.
  - The harness writes `traces/trace_<instance_id>.json` as an ATIF-compatible `SessionTrace` (S5).

## S18 — Step Rejection Fine-Tuning (PAPER)

- URL: https://ar5iv.labs.arxiv.org/html/2605.10674
- Retrieved: 2026-10-24.
- Facts:
  - RFT filters successful trajectories; SRFT uses a critic to mask harmful steps in unresolved trajectories while keeping them as context.
  - On SWE-bench Verified with Qwen2.5-Coder-32B + SWE-agent, RFT 30.9% vs SRFT 32.2%; base 7.0%; naive all-trajectory distillation 28.5%.
  - 5,000 resolved + 5,000 unresolved trajectories; critic cost $660; 7 runs per experiment.
  - Treat as directional; not competition-specific.

## S19 — SWE-Prime (PAPER)

- URL: https://arxiv.org/abs/2608.27449
- Retrieved: 2026-10-24.
- Facts:
  - Trajectory success does not guarantee high-quality supervision; successful trajectories can contain ineffective, redundant, or risky steps.
  - Two-stage selection: trajectory-level (process quality, result quality, representativeness), then segment-level (contribution, learnability, risk); all segments remain in context, only selected segments contribute to loss.
  - Training on the 10% selected subset outperformed the full resolved dataset on SWE-Bench Pro/Verified (up to 12.2%/24.2% relative).
  - Directional; not competition-specific.

## S20 — Trajectory data curation for LoRA code agents (PAPER)

- URL: https://arxiv.org/abs/2607.17205
- Retrieved: 2026-10-24.
- Facts:
  - Systematic study of trajectory filtering for LoRA fine-tuning of Qwen2.5-Coder-7B on 67,074 SWE trajectories (32,161 resolved).
  - Quality scoring axes: Efficiency and Style; error-retry rate is the dominant sub-dimension.
  - Quality-quantity trade-off is scale-dependent; at small scale, quantity may matter more than quality ranking.
  - Directional; 7B model, not Gemma 4.

## S21 — vLLM LoRA docs (OFFICIAL vendor)

- URL: https://docs.vllm.ai/en/stable/features/lora/
- Retrieved: 2026-10-24.
- Facts:
  - LoRA adapters can be served per request with `enable_lora=True`, `--lora-modules name=path`, `max_loras`, `max_lora_rank`.
  - Requests select an adapter via the `model` field.
  - Dynamic loading is possible with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`.
  - The base model must implement `SupportsLoRA`.

## S22 — vLLM Gemma 4 LoRA support PR (OFFICIAL vendor)

- URL: https://github.com/vllm-project/vllm/pull/39560
- Retrieved: 2026-10-24.
- Facts:
  - Adds `hf_to_vllm_mapper` for `Gemma4ForCausalLM` and `SupportsLoRA` for the multimodal wrapper.
  - Supported target modules: `q_proj`, `k_proj`, `v_proj` (packed as `qkv_proj`), `o_proj`, `gate_proj`, `up_proj` (packed as `gate_up_proj`), `down_proj`, plus MoE expert projections.
  - PR was closed with merge conflicts; the current vLLM version may include an equivalent fix.

## S23 — vLLM Gemma 4 adapter silently ignored (COMMUNITY/UPSTREAM BUG)

- URL: https://github.com/vllm-project/vllm/issues/41754
- Retrieved: 2026-10-24.
- Facts:
  - Multiple users report Gemma 4 LoRA adapters load successfully but have no effect; outputs are byte-identical to base.
  - Root cause/fix reported as PR #39816; users confirm cherry-picking it makes the adapter active on dense 31B, E2B, and MoE 26B.
  - Serving the exact base revision used for training matters; a different revision of the same repo cost quality (15/17 → 17/17 in one test).
  - Implication: **always A/B test the adapter at temperature 0**; do not trust "loaded successfully."

## S24 — vLLM W4A16 compressed-tensors LoRA instability (COMMUNITY/UPSTREAM BUG)

- URL: https://github.com/vllm-project/vllm/issues/50059
- Retrieved: 2026-10-24.
- Facts:
  - On a compressed-tensors int4 W4A16 (group size 32) base (OLMo-3.1-32B, not Gemma), rank-32 all-layer adapters produced weak, non-reproducible outputs across identical runs; rank-16 became reproducible; rank-8 targeting only the last 8 layers worked correctly.
  - The same rank-32 adapter was byte-deterministic and correct in llama.cpp, so the adapter file was fine; the issue is vLLM's LoRA application over that quantization.
  - vLLM 0.17.1 and 0.24.0 both showed it; 0.25+ could not load the checkpoint.
  - This is **not Gemma-specific**, but the competition base is also compressed-tensors W4A16. Treat rank/coverage as a variable to test.

## S25 — Competition discussion: 1,500-trajectory LoRA experiment (COMMUNITY)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/750713
- Retrieved: 2026-10-24.
- Facts:
  - Fine-tuned a LoRA on ~1,500 trajectories from the reference agent; 697 **per-call training samples** from 99 tasks, 60 held-out samples from 11 tasks (not 697 trajectories).
  - Trained on a **dequantized bf16 copy** of the packaged `gemma-4-31b-it-qat-w4a16-ct` because PEFT could not attach to the packaged checkpoint.
  - Adapter: rank 16, alpha 32, dropout 0.05, all seven projections across all 60 layers, 410 modules, 122.4M trainable parameters, 245 MB.
  - 3 epochs, LR 1e-4 cosine, warmup 20 steps, batch 1, grad accum 4, bf16, Liger, packing off, max_length 32768.
  - First submission scored 0.0: 257 s before first tool call, repeated `git log` commands, no edit. Cause: LoRA scaling `alpha/r = 2.0` pushed the model out of distribution.
  - Alpha sweep: alpha 32 (scale 2.0) degenerate; alpha 16 (scale 1.0) loops gone but more thinking/time (median 290 s/task); alpha 8 (scale 0.5) safest/fastest (median 163 s), always produced a patch, but only a quarter of the learned correction; aggregate was a wash vs base.
  - Training data had no thinking traces, but inference used thinking 4096; mismatch caused huge thought blocks/timeouts. At alpha 16, thought blocks grew to 2,500–14,700 characters per step and median task time rose to 290 s (vs 163 s at alpha 8).
  - The experiment generated trajectories at five sampling temperatures between 0.2 and 1.0; the patched chat template was verified to produce 12,602 tokens in both stock and patched forms.
  - Budget prompt is proportional; raising the cap can give more room to explore, not more room to finish.
  - `error (exit 0)` in the log means tool calls after budget exhaustion, not a tool failure.
  - Adapter must be referenced in `agent.yaml`; vLLM logs label requests with the base model, so `/metrics` does not prove the adapter is active.

## S26 — Competition discussion: LoRA startup OOM / rank 64 scored (COMMUNITY + HOST)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/749191
- Retrieved: 2026-10-24.
- Facts:
  - LoRA bundles failed at startup on 4×L4 with `CUDA out of memory occurred when warming up sampler with 256 dummy requests`.
  - Root cause: with `--enable-lora`, vLLM captures CUDA graphs twice; graph memory is not counted in the KV budget unless `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`.
  - Verified fixes: set the env var (graphs budgeted; KV 27,712 tokens), `--compilation-config '{"cudagraph_specialize_lora": false}'` (graphs 2.41 GiB; KV 45,440), or `--max-num-seqs 64` (graphs 1.50 GiB; KV 48,384).
  - Host: the Getting Started notebook uses the same configuration as the scorer; `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` is the key setting; successful submissions with LoRA adapters up to **rank 64** have been seen.
  - Host later patched vLLM LoRA parameters to be dynamic based on the submission; fewer adapters/lower rank leave more KV cache.

## S27 — Competition discussion: LoRA KV cache drop (COMMUNITY + HOST)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744331
- Retrieved: 2026-10-24.
- Facts:
  - With the old defaults (`max_loras=8`, `max_lora_rank=128`), a single sample adapter reduced KV cache from 46,048 to 7,600 tokens; prompts >7.6k hung.
  - Host patched LoRA parameters to be based on the submission.
  - After the patch, a single rank-16 adapter loaded and served; prompts up to 28k completed; KV cache dropped from 36,432 to 27,712 tokens (0.76×) at util 0.80.
  - Most of the loss was CUDA-graph specialization; rank-16 buffers themselves cost ~0.1 GiB.
  - Recommendation from community: one adapter, low rank; validate long prompts.

## S28 — Competition discussion: harness issues (COMMUNITY + HOST)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744794
- Retrieved: 2026-10-24.
- Facts:
  - Compaction may lose tool results because the ADK summary keeps only text parts; Gemma 4 rarely writes text next to tool calls. Host said they would look into it.
  - LoRA reduces KV cache; host said it should be addressed by dynamic LoRA params.
  - Double-encoded tool output, thoughts dropped between tool calls, and `thinking_budget`/`seed` not sent were reported; host said these are fixed in the latest wheelhouse.
  - No per-agent budget: one clock, one call counter, and one turn counter cover the whole task; the schema has no per-agent limit, and `exit_loop` is not in the tool registry. A two-stage test had 7 of 8 timeouts come from the first stage (community report in the thread).

## S29 — Competition discussion: 12-hour timeout and scorer contract (COMMUNITY)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/752017
- Retrieved: 2026-10-24.
- Facts:
  - Official 12-hour budget; displayed scoring runs have taken ~8.9–11.9 hours.
  - A 22 October submission whose only change was an extra verification step ended with "Notebook Timeout" after ~12 hours: **no partial score**, daily slot used.
  - Local timing under-estimated the cost of test-running steps on hidden repos.
  - Open questions to host: exact global budget, which phases count, per-task limit behavior, compaction settings, repository split. No host answer in the thread at retrieval.

## S30 — Competition discussion: queue delays (OFFICIAL staff + community)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747519
- Retrieved: 2026-10-24.
- Facts:
  - High L4 demand caused long queues; submissions waiting >15h were marked "Kaggle Error" even if they eventually ran.
  - Queue timeout was extended; affected submissions reprocessed.
  - A pending submission limit was added: if the previous submission is still scoring, you cannot submit again.
  - Queue delay was ~33h, later ~14h; most runs should finish within ~24h of submission.
  - Host: losing a daily submission to queueing is possible; GPU supply is the constraint.
  - "Notebook Threw Exception" failures on unchanged bundles were reported, possibly vLLM startup >1200 s; some were reprocessed.
  - Tool-call argument loss was reported in the same thread: 60 of 234 `edit_file` calls and 20 of 255 `write_file` calls rejected for missing parameters, while 0 of 1,135 `run_command` calls failed; 64 rejected calls were exact resends.

## S31 — Competition discussion: public gold-patch failures (COMMUNITY + HOST)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747248
- Retrieved: 2026-10-24.
- Facts:
  - In a plain local reproduction, only 43/129 public gold patches resolved; fastapi 1/67, rich 39/48, requests 3/13, httpx 0/1.
  - Causes: missing `typing_inspection` for pydantic/fastapi; bundled repo wheels shadowing the patched workspace; missing test deps.
  - Host: none of the public training repos are in the test set; every hidden task validates 100% with a gold patch; they are fixing the training task environment.
  - A community calibrated sandbox later got 127/129 gold patches passing (S32).
  - Public `rich` wheel included a released version newer than the tasks; stronger models sometimes read the fixed copy outside `/workspace`; hidden private repos do not have this leak.

## S32 — Competition discussion: calibrated local eval (COMMUNITY)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/751487
- Retrieved: 2026-10-24.
- Facts:
  - Plain notebook: 52/129 gold patches passed (fastapi 10/67, rich 39/48, requests 3/13, httpx 0/1).
  - Rebuilt scorer-like sandbox (clean venv, pytest, competition wheels, missing test packages): **127/129** gold patches passed (fastapi 66/67, rich 47/48, requests 13/13, httpx 1/1).
  - Remaining 2 tasks appear internally inconsistent (opposite skip conditions).
  - Provides a per-task CSV and wheels; some old public tasks need period-correct versions (e.g., Pygments for 7 rich tasks).
  - The author calls it "scorer-like," not an exact scorer match.

## S33 — Competition discussion: async functions missing from graphs (HOST)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745998
- Retrieved: 2026-10-24.
- Facts:
  - `async def` functions are not included in the code graph tools; this affects some public fastapi/httpx tasks.
  - No private-repo task touches an `async def` function; async functions are at most 1.8% of functions in private repos.
  - Host will not update private-task graphs.

## S34 — Competition discussion: 31B vs 12B vs E4B (COMMUNITY)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/750974
- Retrieved: 2026-10-24.
- Facts:
  - Same agent files, 31 public tasks, official serving stack, one A100:
    - 31B, 8-min limit: 9 and 11 solved; avg 5.4 min; 81–84% submitted a patch.
    - 12B, 8-min: 4 solved; avg 6.2 min; 42% patch.
    - 12B, 5-min: 4 and 4; avg 4.5 min; 26% patch.
    - 12B coder + 12B reviewer: 3; 5.0 min; 3% patch.
    - E4B, 5-min: 0; 1.8 min; 97% patch.
  - 12B decodes ~1.9× faster but writes ~1.8× more per call, so per-call latency is similar and it fails to converge; a 12B 8-min hidden run timed out.
  - On 4×L4: 31B 33–35 tok/s, 12B 62–64 tok/s; output tokens/task ~8,100 vs ~12,400; median model call 10.3 s vs 11.5 s; runs using every tool call 35% vs 60%; non-empty patch 78% vs 46%.
  - Identical files score ~0.12–0.22 on the public LB; one task can flip.

## S35 — Competition discussion: public entry results (COMMUNITY)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/751973
- Retrieved: 2026-10-24.
- Facts:
  - Public Apache-2.0 entry with configs, local runs, hidden scores: https://github.com/rodriguescarson/gemma-swe-agent.
  - Best hidden score 0.17; two configs tied at 28/74 locally scored 0.17 and 0.10 hidden; a change that won 9-0 locally scored lower hidden.
  - Single tweaks moved about one task either way.
  - Implication: local screens are noisy; do not overfit.

## S36 — Public 0.24 prompt-only notebook (COMMUNITY)

- URL: https://www.kaggle.com/code/honghanhhh/gemma-4-baseline-lb-0-24
- Retrieved: 2026-10-24.
- Facts:
  - Notebook title/author claims **LB 0.24** (public). The page sidebar shows “Best Score 0.18 V3” for an earlier version, so treat the exact 0.24 as **author-reported** until independently reproduced; the official top public score is 0.24 (S4). Page license: Apache 2.0.
  - Base: `flexonafft/gemma-4-budget-aware-coding-agent-v2` (public 0.22). `agent.yaml`, `configs/sampling.yaml`, `eval_config.yaml` copied byte-for-byte; prompt-only edits.
  - Config: 31B, `temperature=0.2`, `top_p=0.95`, `max_output_tokens=8192`, `thinking_budget=4096`, `include_thoughts=true`, `timeout_seconds=240`, `max_tool_calls=28`, `max_time_minutes=8`, `max_turns=80`.
  - Tools declared: all nine, but the prompt says never use `search_similar_code`, `get_code_neighbors`, `get_code_subgraph` ("They return nothing useful").
  - Prompt rules: edit a source file by call 12; never edit tests; scratch only in `/tmp`; no full test suite; no reading outside `/workspace`; finish with `git status --short && git diff` then `submit_patch`; confirm `patch_size > 0` and `files_changed > 0`.
  - The notebook only writes `submission.zip`; it does not run inference.

## S37 — Public 0.22 prompt-only notebook (COMMUNITY)

- URL: https://www.kaggle.com/code/flexonafft/gemma-4-budget-aware-coding-agent-v2
- Retrieved: 2026-10-24.
- Facts:
  - Public score 0.22; page license: Apache 2.0.
  - Adapts `matterhorn3838/gemma-4-superagent` (0.18) and credits the official sample prompt/community graph-tool audits.
  - Same 8 min / 28 calls / 240 s / 8192 output / 4096 thinking / temp 0.2 configuration.
  - Adds bounded independent searches, focused caller/test reads, recovery from exact-match editing failures.
  - Explicitly says the 8-minute cap is not a guarantee of the global 12-hour limit; the notebook packages without inference.

## S38 — Public 0.18 Budget-Fit notebook (COMMUNITY)

- URL: https://www.kaggle.com/code/matterhorn3838/gemma-4-superagent
- Retrieved: 2026-10-24.
- Facts:
  - Public score 0.18.
  - Rationale for 8 min: in 129 local runs, only about half of the fixes Gemma eventually finds are in the file after 4 minutes of scorer-speed time.
  - 28 tool calls: runs still exploring after many calls almost never succeed; hard cap stops them and pays for longer time.
  - Edit by call 12: runs with no source edit by call 12 fail 87% of the time.
  - 8192 max output: vLLM refuses requests with prompt + max output >32,768 tokens.
  - 240 s command timeout: the notebook author says the scoring pytest run uses the same value, but the harness README says Phase-2 is fixed at 300 s (S5:506). Treat 240 s as a Phase-1 agent-command setting, not a scoring guarantee. Agent commands rarely hit it (2 of ~3,700 calls).
  - Graph tools: exact fully-qualified IDs only; usually return nothing.
  - Hidden set ~120 tasks sequentially within 12h; caps kept a scored run under 12h.
  - Public LB 58 tasks; scores move ±2 tasks between identical submissions.

## S39 — Practitioner blog: evaluator drift and 12h risk (COMMUNITY)

- URL: https://pilkwangkim.github.io/posts/Gemma-4-Developer-Agent-Part-2-Measuring-an-Agent-While-Its-Evaluator-Moved/
- Published: 2026-10-23; retrieved 2026-10-24.
- Facts:
  - Evaluator versions changed repeatedly: swegemma 0.2.7/adk-submission 0.2.11 on Sep 25; swegemma 0.2.11/adk-submission 0.2.13 by Oct 23; vLLM 0.19.1 rebuilt twice under the same version.
  - Early bugs: empty graph files, test edits surviving, tool errors ending tasks, double-encoded tool results, thoughts dropped, LoRA ignored; several fixed by Oct 8/12.
  - Same byte-identical package scored **0.10 and 0.15** in the same observed period.
  - Reference Budget-Fit package: 28 calls, 8 min, 4096 thinking; hidden scores 0.15–0.18; public ~0.18.
  - Local 22-task paired comparison: thinking off 5/22; thinking 512 7/22; uncapped thinking + 5 min timed out 15/22 and scored 0.05; capped thinking scored 0.12 twice.
  - 12h global limit: a slower agent may solve more tasks but have less chance of a scored run; a 5.0-min mean with 120 tasks and 20 min startup is a planning reference; a 1.18× slowdown can reduce completion probability from ~1 to ~0.40 in their illustrative model.
  - Public split ~58 tasks; private ~60; public scores displayed truncated to two decimals.
  - 52 of 69 verification-failed attempts touched a file also touched by the reference fix; file overlap is not proof of correct localization.
  - Community snapshot on 2026-10-20: the best small-model public score was ~0.06.
  - Gemma 4 does not emit tool calls as JSON; the parser expects special token delimiters around string values. Malformed quoting can swallow the following argument. In a 22-task comparison without reasoning, 223 calls were malformed and 80 were rejected for missing arguments; a typical `edit_file` call closed `new_string` with a backtick, so the parser absorbed `old_string`.
  - Allowing 36 tool calls instead of 28 changed a paired local result from 17/44 to 14/44; more calls did not help in that comparison.
  - Problem statements of at most 120 characters had 7 resolves in 32 attempts; longer statements had 24 in 56. This is an association, not a causal effect.
  - Five sampling temperatures between 0.2 and 1.0 were used in the trajectory experiment; the patched chat template was verified at 12,602 tokens in both stock and patched forms.

## S40 — Public entry repo: happyc0der/gemma-swe-agent (COMMUNITY)

- URL: https://github.com/happyc0der/gemma-swe-agent
- Retrieved: 2026-10-24.
- Facts:
  - Extensive public experiment log; best LB 0.12 at the time of the log.
  - Scores: 0.06, 0.10, 0.08, 0.10, 0.06, 0.10, 0.06, 0.12, 0.12, 0.10, 0.06, 0.10 across configurations; 4–7 public tasks, mean ~4.9, sd ~1.0.
  - 4.5 min/task averaged best (5.5 tasks over 4 runs); 5.5 min once hit "Notebook Exceeded Allowed Compute" (12h sequential cap).
  - LoRA r16 on 31B QAT 4-bit on free T4 fit only up to 3,072 tokens; agent trajectories are 7k–17k tokens, so LoRA was not viable on free T4s.
  - Local faithful eval 9/33, 13/33 etc.; public LB 0.12.
  - Useful as a cautionary tale on noise and infrastructure.

## S41 — Kaggle CLI competition docs (OFFICIAL)

- URL: https://raw.githubusercontent.com/Kaggle/kaggle-cli/main/docs/competitions.md
- Retrieved: 2026-10-24.
- Facts:
  - `kaggle competitions submit <competition> -f <file> -m <message>`; `--wait` can block until scoring; `--poll-interval`.
  - `kaggle competitions submissions <competition>` lists past submissions; `kaggle competitions submission <ref>` shows status/score.
  - Code competitions can submit a notebook with `-k`/`-v`; this competition also accepts a zip file upload.

## S42 — Kaggle GPU quota docs (OFFICIAL)

- URL: https://www.kaggle.com/docs/efficient-gpu-usage
- Retrieved: 2026-10-24.
- Facts:
  - Free GPU quota is weekly, nominally **30 hours** or sometimes higher depending on demand/resources.
  - Interactive sessions idle-timeout at 60 minutes; batch sessions run top-to-bottom.
  - The competition overview (S1) says L4x4 consumes quota at 2× the older T4x2/P100 rate.

## S43 — Gemma QAT docs (OFFICIAL)

- URL: https://gemma.ai/get-started/qat/
- Retrieved: 2026-10-24.
- Facts:
  - For vLLM/SGLang use `-qat-w4a16-ct`; for llama.cpp/LM Studio/Ollama use `-qat-q4_0-gguf`; for conversion/research use `-qat-q4_0-unquantized`.
  - QAT memory table matches S7.
  - 31B Q4_0 ~17.5 GB static weights; context KV cache is additional.

## S44 — Unsloth QAT docs (OFFICIAL vendor)

- URL: https://unsloth.ai/docs/models/gemma-4/qat
- Retrieved: 2026-10-24.
- Facts:
  - QAT reduces memory ~72% vs BF16 with near-original performance.
  - 31B QAT ~18 GB total memory for inference at ~8K context (llama.cpp/GGUF context).
  - QAT checkpoints are for inference/deployment; Unsloth's QAT page is about running, not QLoRA training.

## S45 — Unsloth QAT fine-tuning discussion (COMMUNITY)

- URL: https://github.com/unslothai/unsloth/discussions/6389
- Retrieved: 2026-10-24.
- Facts:
  - No supported way to QLoRA fine-tune the actual QAT checkpoint while keeping its Q4_0 calibration intact.
  - Safer path: use the unquantized QAT release as the training base, train LoRA/QLoRA, keep the adapter separate; merging/re-quantizing changes the base.
  - QLoRA's NF4/int4 quantizer is not Google's Q4_0; loading `*-qat-q4_0-unquantized` with `load_in_4bit=True` requantizes for training.

## S46 — Paper track overview (OFFICIAL)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/overview
- Retrieved: 2026-10-24.
- Facts:
  - Separate hackathon; $35,000 total: best paper $15k, best new resource $10k, best new application $10k.
  - Deadline: 2026-11-12, 11:59 PM UTC.
  - 3,000-word maximum; original, unpublished research; non-archival.
  - Judging: novelty, quality, relevance, verifiability, clarity; average 0–5.
  - Submission: Kaggle Writeup with title/subtitle, abstract, introduction, methods/experiments, related works/citations; optional public notebook or arXiv-ready PDF.
  - Topics: tuning/optimization, code comprehension, tasks/benchmarks, graph reasoning.
  - At retrieval: 5,620 entrants, 338 participants, 327 teams, 348 submissions.

## S47 — Small-model bonus (OFFICIAL + COMMUNITY)

- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/751104
- Retrieved: 2026-10-24.
- Facts:
  - $10,000 bonus for the highest-ranking submission using Gemma E2B or E4B.
  - Must explicitly use E2B/E4B as the base model; must be one of the team's two selected final submissions.
  - LoRA fine-tuning allowed; same submission rules/tie-breaking as the main track.
  - Cross-track eligible: small-model entries remain eligible for main leaderboard awards and the Paper Track.
  - A separate daily notebook displays filtered small-model standings.
  - Community snapshot (S39): best small-model public score ~0.06 as of 2026-10-20; one snapshot, not a measured floor.

## S48 — Gemma 4 launch blog (OFFICIAL)

- URL: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- Published: 2026-04-02; retrieved 2026-10-24.
- Facts: Gemma 4 family E2B/E4B/26B MoE/31B dense; Apache 2.0; agentic/function-calling; 256K context for larger models; 12B added later (S49).

## S49 — Gemma 4 QAT launch blog (OFFICIAL)

- URL: https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/
- Published: 2026-06-05; retrieved 2026-10-24.
- Facts: QAT checkpoints for Q4_0 and mobile; QAT preserves quality better than PTQ; vLLM compressed-tensors format; 12B released shortly before.

---

## S50 — Gemma 4 FSDP training issues (COMMUNITY/UPSTREAM)

- URL: https://github.com/verl-project/verl/issues/5999
- Retrieved: 2026-10-24.
- Facts:
  - Under verl + FSDP, Gemma 4 E4B showed abnormally high gradient norms (>10, peaking at 50+) and training instability.
  - Gemma 4 31B failed to load in `SHARDED_STATE_DICT` mode due to an FQN mismatch between expected nested names and checkpoint keys.
  - The issue was closed after the relevant issues were resolved; treat it as evidence that Gemma 4 + FSDP needs a smoke test, not as a permanent blocker.

---

## Planning estimates (not measured on the scorer)

These are explicitly estimates. They are included so the plan has numbers, not because they are guarantees.

- **E1 — 31B QLoRA memory**: Unsloth states ~22 GB for 31B QLoRA (S10); Axolotl's 31B QLoRA example is ~25.2 GiB reserved on 1×80 GB at 2k sequence with packing (S11/S12). A 40 GB A100 is plausible for 31B QLoRA at 4k–8k sequence with gradient checkpointing, but must be smoke-tested. At 16k–32k, expect OOM or a need for FSDP/activation offload.
- **E2 — 31B LoRA FSDP memory**: BF16 weights are ~61.4 GB total (derived: 30.7B params × 2 bytes, S8; S7’s 69.9 GB is inference memory); sharded across 4×40 GB that is ~15.4 GB/GPU before activations. FSDP2 + activation offloading + packing at 8k–16k is plausible; 32k is a stretch. Must be smoke-tested.
- **E3 — Training time**: A first 31B QLoRA campaign on 1×A100 40 GB with 500–2,000 per-call samples, 8k sequence, 1–3 epochs is likely tens of GPU-hours, not minutes. Measure tokens/s in a 10-step smoke test and extrapolate. Do not book a long run before the smoke test.
- **E4 — Cost**: 4×A100 40 GB rental prices vary by provider/time; budget 50–150 GPU-hours for the first serious campaign and check current pricing. This is a planning range, not a quote.
- **E5 — Storage**: 31B BF16 weights ~62 GB; QAT CT ~17.5 GB; competition data 22.42 GB; snapshots/results/checkpoints can add 100+ GB. Verify free disk before downloading/training.
- **E6 — Score equivalents**: At ~58 public LB tasks, one task is ~0.017; at ~60 final/private tasks, one task is ~0.017; 1/120 ≈ 0.0083 is only the blended rate over all hidden tasks. Displayed scores are truncated to two decimals. Differences of 1–2 tasks are within the noise reported by multiple teams (S34/S35/S39).

## How to use citations

- `[S5:500]` means line 500 of the local harness README.
- `[S6:280]` means line 280 of the local getting-started notebook.
- `[S1]` means the source entry above.
- `[COMMUNITY]` or `[ESTIMATE]` is called out in prose when the claim is not official.
- If a claim is not cited, it is either a direct logical consequence of cited facts or an explicitly labeled inference.

## Maintenance

- Re-check S1/S3/S4/S46/S47 before each submission; dates, quotas, and host patches change.
- Record the wheelhouse version/hash, vLLM version, adapter hash, prompt hash, and eval config for every run.
- If a host post contradicts a community report, update the doc and the source ledger; do not silently keep the old number.
