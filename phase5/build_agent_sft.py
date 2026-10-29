#!/usr/bin/env python3
"""Build per-call SFT samples from successful ATIF traces using the current
v4-32 system prompt and tool schema. Emits prompt/completion rows in the same
Gemma 4 turn format as native_clean_sft.jsonl.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

SYSTEM = Path("phase5/submission-v4-32-prefixed/prompts/system.md").read_text()
TOOLS = Path("phase5/tool_block.txt").read_text().strip()
BASE = (
    "<bos><|turn>system\n<|think|>\n"
    + SYSTEM.rstrip()
    + "\n\n"
    + TOOLS
    + "\n<turn|>\n<|turn>user\n"
)

def fmt_arg(v):
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)

def fmt_call(tc):
    fn = tc.get("function_name", "")
    args = tc.get("arguments") or {}
    body = ",".join(f"{k}:<|\"|>{fmt_arg(v)}<|\"|>" for k, v in args.items())
    return f"<|tool_call>call:{fn}{{{body}}}<tool_call|>"

def build_trace(trace_path: Path, task_id: str, repo: str, source: str):
    d = json.loads(trace_path.read_text())
    steps = d.get("steps", [])
    system = next((s.get("message", "") for s in steps if s.get("source") == "system"), "")
    user = next((s.get("message", "") for s in steps if s.get("source") == "user"), "")
    if not user:
        return []
    # The trace system message is the bundle prompt without tool declarations.
    # Use it when it matches the current prompt; otherwise fall back to SYSTEM.
    system_text = system.strip() if system.strip() else SYSTEM.strip()
    base = (
        "<bos><|turn>system\n<|think|>\n"
        + system_text
        + "\n\n"
        + TOOLS
        + "\n<turn|>\n<|turn>user\n"
        + user
        + "\n<turn|>\n<|turn>model\n"
    )
    agent_steps = [s for s in steps if s.get("source") == "agent"]
    rows = []
    prefix = base
    for i, s in enumerate(agent_steps):
        msg = (s.get("message") or "").strip()
        calls = s.get("tool_calls") or []
        if not msg and not calls:
            continue
        if i == 0:
            prompt = prefix
            completion = f"<|channel>thought\n{msg}<channel|>"
        else:
            prompt = prefix + "<|channel>thought\n"
            completion = f"{msg}<channel|>"
        for tc in calls:
            completion += fmt_call(tc)
        if calls:
            completion += "<|tool_response>"
        rows.append({
            "task_id": task_id,
            "repo": repo,
            "prompt": prompt,
            "completion": completion,
            "source": source,
        })
        if not calls:
            break
        # Add this model turn and the tool response to the next prefix.
        obs = s.get("observation") or {}
        content = obs.get("content", "") if isinstance(obs, dict) else str(obs)
        fn = calls[0].get("function_name", "")
        prefix = (
            prefix
            + completion
            + f"response:{fn}{{value:<|\"|>{content}<|\"|>}}<tool_response|>"
        )
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("phase5/agent_sft.jsonl"))
    p.add_argument("--mix-out", type=Path, default=Path("phase5/agent_sft_mix.jsonl"))
    p.add_argument("--repeat", type=int, default=5)
    a = p.parse_args()
    sources = [
        ("local-results/results-v4-32-prefixed", "local-results/task-results-v4-32-prefixed.csv", "v4-32-prefixed"),
        ("local-results/results-v6-32-prefixed", "local-results/task-results-v6-32-prefixed.csv", "v6-32-prefixed"),
        ("local-results/results-v7-32-finisher-prefixed", "local-results/task-results-v7-32-finisher-prefixed.csv", "v7-32-finisher"),
        ("remote-results/results-native-policy-v4-32", "remote-results/task-results-native-policy-v4-32.csv", "v4-32-remote"),
    ]
    rows = []
    seen = set()
    for trace_dir, csv_path, tag in sources:
        if not Path(trace_dir).is_dir() or not Path(csv_path).is_file():
            print("skip missing", trace_dir)
            continue
        results = {r["id"]: r for r in csv.DictReader(open(csv_path))}
        for trace_path in sorted(Path(trace_dir, "traces").glob("trace_*.json")):
            task_id = trace_path.stem[len("trace_"):]
            r = results.get(task_id)
            if not r:
                continue
            if r.get("resolved", "").lower() not in ("true", "1"):
                continue
            try:
                if int(float(r.get("patch_chars", 0))) <= 0:
                    continue
            except Exception:
                continue
            repo = r.get("repo", "")
            for row in build_trace(trace_path, task_id, repo, tag):
                key = (task_id, row["completion"][:200], row["prompt"][-200:])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    with open(a.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    native = [json.loads(l) for l in open("phase5/native_clean_sft.jsonl")]
    with open(a.mix_out, "w") as f:
        for _ in range(a.repeat):
            for row in rows:
                f.write(json.dumps(row) + "\n")
        for row in native:
            f.write(json.dumps(row) + "\n")
    print(f"agent rows={len(rows)} native={len(native)} mix={len(rows)*a.repeat + len(native)}")
    print("tasks", sorted({r['task_id'] for r in rows}))
    print("first prompt tail", rows[0]['prompt'][-300:] if rows else "")
    print("first completion", rows[0]['completion'][:400] if rows else "")

if __name__ == "__main__":
    main()
