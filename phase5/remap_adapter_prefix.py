#!/usr/bin/env python3
"""Remap a PEFT Gemma4 text adapter to vLLM's text-only module prefix.

Training uses `model.language_model.layers.*`; vLLM's Gemma4ForCausalLM exposes
`model.self_decoder.decoder_layers.*`. Keep separate q/k/v and gate/up keys:
vLLM's packed_modules_mapping splits qkv_proj/gate_up_proj for LoRA.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from safetensors.torch import load_file, save_file

SRC = "base_model.model.model.language_model.layers."
DST = "base_model.model.model.self_decoder.decoder_layers."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--base-model", default="/nlsasfs/home/isea/isea31/gemma4/models/gemma-4-31B-it-qat-w4a16-ct")
    a = p.parse_args()
    a.dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((a.src / "adapter_config.json").read_text())
    cfg["base_model_name_or_path"] = a.base_model
    cfg["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    (a.dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    w = load_file(a.src / "adapter_model.safetensors")
    out = {k.replace(SRC, DST): v for k, v in w.items()}
    save_file(out, a.dst / "adapter_model.safetensors")
    for name in ("tokenizer_config.json", "tokenizer.json", "README.md"):
        src = a.src / name
        if src.exists():
            shutil.copy2(src, a.dst / name)
    print(f"remapped {len(out)} tensors -> {a.dst}")


if __name__ == "__main__":
    main()
