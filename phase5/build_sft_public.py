#!/usr/bin/env python3
"""Convert the public LoRA trajectories into prompt-completion SFT rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.home() / "gemma4" / "data" / "lora")
    p.add_argument("--out", type=Path, default=Path.home() / "gemma4" / "sft" / "public_sft.jsonl")
    p.add_argument("--tiers", default="native,external_A")
    p.add_argument("--max-context", type=int, default=8192)
    return p.parse_args()


def main():
    a = parse_args()
    sys.path.insert(0, str(a.root))
    from lora_data import iter_steps
    tiers = tuple(t.strip() for t in a.tiers.split(",") if t.strip())
    n = 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for step in iter_steps(a.root, tiers=tiers, max_context_tokens=a.max_context):
            row = {
                "task_id": f"{step['tier']}:{step['trajectory_id']}",
                "repo": step["issue"],
                "prompt": step["prompt"],
                "completion": step["completion"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} examples to {a.out}")


if __name__ == "__main__":
    main()
