# 09 — Adapter format and local Docker evaluator

Status: verified 2026-10-29.

## 1. Scorer vLLM build

The Kaggle wheelhouse dataset `metric/gemma-4-developer-agent-wheelhouse` contains:

```text
vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl
sha256 73c69bafca24fbdad44690c44d6c6446514351e6f3cc9a6057cba05e66051310
```

The same wheel is installed in `~/gemma4/vllm-venv`. It contains the custom Gemma 4 patch:

- `Gemma4ForCausalLM` implements `SupportsLoRA`
- `packed_modules_mapping` maps `qkv_proj` -> `q_proj`/`k_proj`/`v_proj`
- `packed_modules_mapping` maps `gate_up_proj` -> `gate_proj`/`up_proj`
- duplicate child-module registration is fixed for `named_modules(remove_duplicate=False)`

The W4A16 config resolves to `Gemma4ForCausalLM`, whose actual modules are:

```text
model.self_decoder.decoder_layers.N.self_attn.qkv_proj
model.self_decoder.decoder_layers.N.self_attn.o_proj
model.self_decoder.decoder_layers.N.mlp.gate_up_proj
model.self_decoder.decoder_layers.N.mlp.down_proj
```

## 2. Correct adapter format

Training with PEFT produces separate text-backbone keys:

```text
base_model.model.model.language_model.layers.N.self_attn.q_proj
base_model.model.model.language_model.layers.N.self_attn.k_proj
base_model.model.model.language_model.layers.N.self_attn.v_proj
base_model.model.model.language_model.layers.N.self_attn.o_proj
base_model.model.model.language_model.layers.N.mlp.gate_proj
base_model.model.model.language_model.layers.N.mlp.up_proj
base_model.model.model.language_model.layers.N.mlp.down_proj
```

vLLM's `Gemma4ForCausalLM` expects the same separate keys under:

```text
base_model.model.model.self_decoder.decoder_layers.N...
```

Export with:

```bash
~/gemma4/train-venv/bin/python phase5/remap_adapter_prefix.py \
  --src ~/gemma4/adapters/<trained-adapter> \
  --dst ~/gemma4/adapters/<trained-adapter>-vllm
```

The remap keeps `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj` separate.
vLLM's packed-module mapping handles the fused `qkv_proj` / `gate_up_proj` layers.

Adapter config:

```json
{
  "base_model_name_or_path": "/nlsasfs/home/isea/isea31/gemma4/models/gemma-4-31B-it-qat-w4a16-ct",
  "r": 16,
  "lora_alpha": 8,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
```

Serve with `--max-lora-rank 16`.

### Do not fuse the adapter

A converted `qkv_proj`/`gate_up_proj` adapter is rejected by vLLM:

```text
ValueError: expected target modules in
{'up_proj','gate_proj','q_proj','o_proj','k_proj','v_proj','down_proj'}
but received ['model.self_decoder.decoder_layers.0.mlp.gate_up_proj', ...]
```

The working format is separate keys + `self_decoder.decoder_layers` prefix.

### Activation proof

Base vs active adapter at temperature 0, same prompt:

```text
base       token '```' logprob -1.7881377516459906e-06
prefixed   token '```' logprob -2.861018856492592e-06
```

The top-2 alternatives also differ. This is the required proof. “Loaded new LoRA adapter” alone is not proof.

## 3. Training constraints

- Direct PEFT on the W4A16 compressed-tensors checkpoint fails: `'Linear' object has no attribute 'weight'`.
- The unquantized QAT base loads as `Gemma4ForConditionalGeneration` (multimodal), not the scorer's text-only class.
- The current adapter was trained on `~/gemma4/sft/native_clean_sft.jsonl` (778 rows, rank 16, alpha 8) on the unquantized QAT base, then prefix-remapped for vLLM.
- Training a new adapter on the same data is redundant. Future training should use successful agent trajectories with the current prompt, then run `remap_adapter_prefix.py`.

## 4. Local Docker evaluator

Built locally:

- image `swebench-sandbox:latest` from `docker-sandbox/Dockerfile.public`
- `.venv-local` with `swegemma`, `adk-submission`, `adk-eval-core`, `google-adk`, `litellm==1.105.0`
- local `data/tasks.jsonl`, `data/snapshots/` (13 tasks), `docker-sandbox/wheels/`
- `phase5/local_eval.py` and `phase5/local_verify.py`

Model endpoint via SSH tunnels:

```bash
ssh -N -L 8007:scn6-mn:8007 isea31   # prefixed adapter
ssh -N -L 8008:scn6-mn:8008 isea31   # second prefixed adapter
```

Run:

```bash
.venv-local/bin/python phase5/local_eval.py \
  --agent-dir phase5/submission-v4-32-prefixed \
  --tag v4-32-prefixed \
  --task-id fastapi_15588 --task-id fastapi_15589 \
  --task-id requests_7427 --task-id rich_4077 --task-id rich_4076 \
  --per-repo 4 --host 127.0.0.1 --port 8007
```

### Gold audit

Docker gold audit: **8/13** pass.

Clean five (all gold pass):

```text
fastapi_15588
fastapi_15589
requests_7427
rich_4077
rich_4076
```

Remaining gold failures are environment-limited:

- `requests_7505`, `requests_7502`, `requests_7433`: full test file hits network-timeout/path tests; target tests may pass.
- `rich_4070`, `rich_4079`: Pygments color drift; gold does not pass in this image.

### Docker harness caveats

- A run can report `resolved=True` with `patch_chars=0`. Only trust results with `patch_chars > 0`.
- `fastapi_15588` is inconsistent: the test file contains `must be a single line`, but a patch using `must not contain newline characters` passes. Treat that task as unverified until the harness is fixed.
- Use only the five clean tasks for prompt comparison.

## 5. Current best combination

Five clean tasks, active prefixed adapter:

| Bundle | Reported | Real (patch > 0) | Notes |
|---|---:|---:|---|
| `submission-v4-32-prefixed` | 5/5 | **4/5 certain** | all five have patches; `fastapi_15588` message test inconsistent |
| `submission-v4-32-prefixed-clean` | 4/5 | 3/5 | removing `get_status`/graph tools regressed `requests_7427`; one `patch_chars=0` false positive |
| `submission-v6-32-prefixed` | 4/5 | 2/5 | two `patch_chars=0` false positives; `requests_7427` fails |
| `submission-v7-32-finisher-prefixed` | 5/5 | 3/5 | two `patch_chars=0` false positives |

Best ready bundle: **`phase5/submission-v4-32-prefixed`** — keep the original tool list.

- 32 tool calls, 8 minutes, 4096 thinking, 8192 output, 240 s command timeout.
- Adapter: `native-clean-prefixed`, rank 16, alpha 8, 234 MB.
- Keep only `adapter_config.json` and `adapter_model.safetensors` in `adapters/`.
- Do not include `chat_template.jinja`; `.jinja` is a disallowed submission extension.
- Projected 120-task time: ~10.7 h at ~320 s/task. Do not raise to 48 calls.
- Do not remove `get_status`/graph tools; the clean-tools variant lost `requests_7427`.

## 6. Future dataset work

See `docs/superpowers/plans/2026-10-29-gold-dataset-pipeline.md` for the phased gold-standard dataset pipeline: task pool, multi-temperature generation, execution validation, process scoring, human review, per-call formatting, dedupe/decontamination, dataset card, training/evaluation, and hard-example iteration.
