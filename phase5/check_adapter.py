#!/usr/bin/env python3
"""Compare base and LoRA outputs at temperature 0."""
import os
from openai import OpenAI

BASE = os.environ.get("VLLM_BASE", "http://scn6-10g:8000/v1")
client = OpenAI(base_url=BASE, api_key="EMPTY")
prompt = "Write a Python function `add_all(xs)` that returns the sum of a list. Return only code."


def gen(model: str) -> str:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=128,
    )
    return r.choices[0].message.content or ""


base = gen("gemma-4-31b-it-qat-w4a16-ct")
gold = gen("gold")
print("BASE:\n", base)
print("GOLD:\n", gold)
print("IDENTICAL:", base == gold)
