#!/usr/bin/env python3
"""Run the swegemma evaluator against an already-running local vLLM server.

The remote vLLM server is started separately from the vLLM venv.  This script
lives in the harness venv (``~/Gemma-4-Agent/.venv``) and only registers the
OpenAI-compatible endpoint; it does not spawn vLLM itself.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_TELEMETRY", "False")
os.environ.setdefault("LITELLM_LOG", "WARNING")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks

from adk_submission import VllmConfig, VllmServer, discover_adapters
from adk_submission.discovery import discover_declared_models
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS


def run_sync(coro_or_fn, *args, **kwargs):
    if callable(coro_or_fn) and not asyncio.iscoroutine(coro_or_fn):
        fn = lambda: coro_or_fn(*args, **kwargs)
    else:
        fn = lambda: coro_or_fn
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(fn())).result()
    return asyncio.run(fn())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--agent-dir", required=True, type=Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("local-results"))
    p.add_argument("--port", type=int, default=8004)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--per-repo", type=int, default=4)
    p.add_argument("--image", default="swebench-sandbox:latest")
    p.add_argument("--wheels-dir", type=Path, default=Path("docker-sandbox/wheels"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--task-id", action="append", default=[], help="only run these instance ids")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    agent_dir = args.agent_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = data_dir / "tasks.jsonl"
    tasks = load_tasks(tasks_path)
    by_repo: dict[str, list] = {}
    for t in tasks:
        by_repo.setdefault(t.repo, []).append(t)
    selected = [t for ts in by_repo.values() for t in ts[: args.per_repo]]
    if args.task_id:
        wanted = set(args.task_id)
        selected = [t for t in selected if t.instance_id in wanted]
        missing = wanted - {t.instance_id for t in selected}
        assert not missing, f"task ids not in selection: {sorted(missing)}"
    if args.limit:
        selected = selected[: args.limit]

    raw = yaml.safe_load((agent_dir / "eval_config.yaml").read_text(encoding="utf-8")) or {}
    sec = raw.get("evaluation", raw)
    timeout_seconds = int(sec.get("timeout_seconds", 300))
    verification_timeout_seconds = int(sec.get("verification_timeout_seconds", 300))
    max_tool_calls = int(sec.get("max_tool_calls", 100))
    max_time_minutes = float(sec.get("max_time_minutes", 60.0))
    turns_raw = sec.get("max_turns", sec.get("max_llm_calls"))
    max_turns = int(turns_raw) if turns_raw is not None else None

    declared_models = discover_declared_models(agent_dir)
    assert len(declared_models) == 1, f"expected one model, got {sorted(declared_models)}"
    declared_model = next(iter(declared_models))

    adapters = discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    adapter_map = getattr(adapters, "adapters", None) or {}
    detected_ranks = [
        r for info in adapter_map.values() if (r := getattr(info, "rank", None)) is not None and r > 0
    ]

    # Dummy server object: vLLM is already running.  This only supplies the
    # OpenAI-compatible base URL to the model registry.
    vllm_cfg = VllmConfig(
        model="gemma-4-31b-it-qat-w4a16-ct",
        port=args.port,
        host=args.host,
        tool_call_parser="gemma4",
        reasoning_parser="gemma4",
        max_model_len=32768,
        dtype="auto",
        gpu_memory_utilization=0.90,
        enable_auto_tool_choice=True,
        enable_lora=bool(adapter_map),
        max_loras=max(1, len(adapter_map)),
        max_lora_rank=max(detected_ranks) if detected_ranks else 128,
        scheduler_reserve_full_isl=False,
        tensor_parallel_size=4,
        startup_timeout=1200,
        extra_args=(),
    )
    server = VllmServer(vllm_cfg, adapter_manifest=adapters)
    models = server.create_model_registry(
        aliases={declared_model: declared_model},
        model_prefix="openai/",
        api_key="EMPTY",
    )

    limits, gen_constraints = build_submission_limits()
    cfg = EvalConfig(
        tasks_path=tasks_path,
        snapshots_dir=data_dir / "snapshots",
        results_dir=results_dir / f"results-{args.tag}",
        submission_dir=agent_dir,
        models=models,
        sandbox="docker",
        timeout_seconds=timeout_seconds,
        verification_timeout_seconds=verification_timeout_seconds,
        max_time_minutes=max_time_minutes,
        max_tool_calls=max_tool_calls,
        max_turns=max_turns,
        limits=limits,
        generation_constraints=gen_constraints,
        adapter_manifest=adapters,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=15,
            overlap_size=2,
            token_threshold=14336,
            event_retention_size=5,
        ),
        graph_dir=str(data_dir / "graphs"),
        embeddings_dir=str(data_dir / "embeddings"),
        wheels_dir=args.wheels_dir,
        verbose=False,
        image=args.image,
        display_mode="quiet",
    )

    print(f"agent={agent_dir} model={declared_model} tasks={len(selected)} port={args.port}", flush=True)
    evaluator = Evaluator(cfg)
    rows = []
    for idx, task in enumerate(selected, start=1):
        print(f"[{args.tag} {idx}/{len(selected)}] {task.instance_id} ({task.repo})...", flush=True)
        result = run_sync(evaluator.evaluate_task, task=task, task_index=idx, total_tasks=len(selected))
        patch = (result.agent_patch or "").strip()
        rows.append(
            {
                "id": task.instance_id,
                "repo": task.repo,
                "prediction": patch if patch else "NO_PATCH",
                "resolved": bool(result.resolved),
                "exit_code": result.test_exit_code,
                "patch_chars": len(patch),
                "tool_calls": result.tool_calls,
                "duration_s": result.duration_seconds,
            }
        )
        print(
            f"  -> resolved={result.resolved} exit={result.test_exit_code} "
            f"patch={len(patch)} calls={result.tool_calls} dur={result.duration_seconds:.1f}s",
            flush=True,
        )

    df = pd.DataFrame(rows)
    out = results_dir / f"task-results-{args.tag}.csv"
    df.to_csv(out, index=False)
    df[["id", "prediction"]].to_csv(results_dir / f"predictions-{args.tag}.csv", index=False)
    print(json.dumps({"tag": args.tag, "csv": str(out), "resolved": int(df.resolved.sum()), "n": len(df)}))
    print(df.groupby("repo").resolved.agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
