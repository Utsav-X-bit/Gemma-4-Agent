#!/usr/bin/env python3
"""Re-run only the verification phase for saved predictions.

Loads predictions-<tag>.csv, applies each patch to a clean snapshot, runs the
hidden test patch, and writes the full pytest output to a JSONL file.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

import pandas as pd

from adk_submission.registry import ModelRegistry
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.evaluate import Evaluator
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--task-id", action="append", default=[])
    p.add_argument("--results-dir", type=Path, default=Path("local-results"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--gold", action="store_true", help="verify the dataset gold patch instead of the prediction")
    p.add_argument("--image", default="swebench-sandbox:latest")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    tasks = {t.instance_id: t for t in load_tasks(args.data_dir / "tasks.jsonl")}
    pred_path = args.results_dir / f"predictions-{args.tag}.csv"
    preds = {}
    with open(pred_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            preds[row["id"]] = row["prediction"]

    wanted = args.task_id or sorted(preds)
    out_path = args.out or args.results_dir / f"verify-{args.tag}.jsonl"
    limits, gen = build_submission_limits()
    cfg = EvalConfig(
        tasks_path=args.data_dir / "tasks.jsonl",
        snapshots_dir=args.data_dir / "snapshots",
        results_dir=args.results_dir / f"verify-results-{args.tag}",
        submission_dir=Path.cwd(),
        models=ModelRegistry(),
        sandbox="docker",
        timeout_seconds=300,
        verification_timeout_seconds=300,
        max_time_minutes=8,
        max_tool_calls=32,
        limits=limits,
        generation_constraints=gen,
        wheels_dir=Path("docker-sandbox/wheels"),
        graph_dir=str(args.data_dir / "graphs"),
        embeddings_dir=str(args.data_dir / "embeddings"),
        verbose=True,
        image=args.image,
    )
    ev = Evaluator(cfg)
    with open(out_path, "w", encoding="utf-8") as out:
        for i, tid in enumerate(wanted, 1):
            task = tasks[tid]
            snap, base, patch = resolve_task_snapshot_paths(
                cfg.snapshots_dir, task.instance_id, task.repo
            )
            print(f"[{i}/{len(wanted)}] verify {tid} snapshot={snap.name}", flush=True)
            start = time.perf_counter()
            res = await verify_task(
                ev.docker,
                cfg,
                task,
                snap,
                base_snapshot_path=base,
                patch_path=patch,
                agent_patch=(task.patch if args.gold else preds.get(tid, "")),
                trace=None,
                start_time=start,
            )
            rec = {
                "id": tid,
                "resolved": res.resolved,
                "test_exit_code": res.test_exit_code,
                "error": res.error,
                "test_output": res.test_output,
                "duration_s": res.duration_seconds,
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(
                f"  -> resolved={res.resolved} exit={res.test_exit_code} "
                f"err={res.error!r} dur={res.duration_seconds:.1f}s",
                flush=True,
            )
            if res.test_output:
                print(res.test_output[-2500:], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
