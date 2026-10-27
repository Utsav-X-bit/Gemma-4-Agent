#!/usr/bin/env python3
"""Gold-patch QLoRA SFT for Gemma 4 31B.

Smoke first:
  CUDA_VISIBLE_DEVICES=3 python phase5/train_gold_sft.py --max-steps 10

Full run:
  CUDA_VISIBLE_DEVICES=3 python phase5/train_gold_sft.py --epochs 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# Gemma 4 text attention/MLP uses plain nn.Linear; the vision tower wraps
# projections in Gemma4ClippableLinear, which PEFT cannot target. Train the
# text backbone only; no vision/audio LoRA.
TARGET_MODULES = r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
EXCLUDE_MODULES = r".*(vision_tower|audio_tower|vision_model|audio_model).*"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path.home() / "gemma4" / "models" / "gemma-4-31B-qat-unquantized")
    p.add_argument("--data", type=Path, default=Path.home() / "gemma4" / "sft" / "gold_sft.jsonl")
    p.add_argument("--output-dir", type=Path, default=Path.home() / "gemma4" / "adapters" / "gold-sft-r16")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--save-steps", type=int, default=50)
    return p.parse_args()


def load_model(model_path: Path):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    common = dict(
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    try:
        model = AutoModelForImageTextToText.from_pretrained(str(model_path), **common)
        print("loaded with AutoModelForImageTextToText")
    except Exception as e:
        print(f"AutoModelForImageTextToText failed ({type(e).__name__}: {e}); trying AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **common)
        print("loaded with AutoModelForCausalLM")
    return model


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=str(args.data), split="train")
    print(f"dataset rows={len(ds)} columns={ds.column_names}")

    model = load_model(args.model)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()

    peft = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
        exclude_modules=EXCLUDE_MODULES,
    )
    model = get_peft_model(model, peft)
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        dataloader_num_workers=0,
        completion_only_loss=True,
        packing=False,
        report_to="none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"saved adapter to {args.output_dir}")
    (args.output_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "model": str(args.model),
                "data": str(args.data),
                "rows": len(ds),
                "rank": args.rank,
                "alpha": args.alpha,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "max_length": args.max_length,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
