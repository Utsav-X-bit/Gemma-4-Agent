#!/usr/bin/env python3
"""Compare base and LoRA outputs at temperature 0."""
import os
from openai import OpenAI

BASE = os.environ.get("VLLM_BASE", "http://scn6-10g:8000/v1")
client = OpenAI(base_url=BASE, api_key="EMPTY")
prompt = "Fix this bug: FileProxy inherits io.TextIOBase, whose isatty() returns False, so FileProxy.isatty() is not proxied. Return only the unified diff."


def gen(model: str) -> str:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=5,
    )
    lp = r.choices[0].logprobs
    if lp and lp.content:
        return [(t.token, round(t.logprob, 6)) for t in lp.content[0].top_logprobs]
    return r.choices[0].message.content or ""


base = gen("gemma-4-31b-it-qat-w4a16-ct")
gold = gen("gold")
print("BASE:\n", base)
print("GOLD:\n", gold)
print("IDENTICAL:", base == gold)
