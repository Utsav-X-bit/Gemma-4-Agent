#!/usr/bin/env python3
"""Build gold-patch SFT examples from the public tasks and snapshots.

One example per task:
  prompt     = system + issue + changed source files at base commit
  completion = gold unified diff

The 13-task local screen is held out; the other 116 tasks are train.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from collections import OrderedDict
from pathlib import Path

SYSTEM = (
    "You are a senior software engineer. Given a GitHub issue and the relevant "
    "source files at the base commit, produce a minimal unified diff patch that "
    "fixes the issue. Only modify source files. Do not modify tests. "
    "Return only the diff, with no prose."
)

DIFF_PATH_RE = re.compile(r"^(?:---|\+\+\+) [ab]/(.+)$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, default=Path.home() / "gemma4" / "data" / "tasks.jsonl")
    p.add_argument("--snapshots", type=Path, default=Path.home() / "gemma4" / "data" / "snapshots")
    p.add_argument("--out", type=Path, default=Path.home() / "gemma4" / "sft" / "gold_sft.jsonl")
    p.add_argument("--split-dir", type=Path, default=Path.home() / "gemma4" / "sft")
    p.add_argument("--per-repo-holdout", type=int, default=4)
    p.add_argument("--max-file-chars", type=int, default=12000)
    p.add_argument("--max-context-chars", type=int, default=36000)
    return p.parse_args()


def changed_paths(patch: str) -> list[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for m in DIFF_PATH_RE.finditer(patch):
        path = m.group(1).strip()
        if path and path != "/dev/null" and not path.startswith("tests/") and "/tests/" not in path:
            seen.setdefault(path, None)
    return list(seen)


def read_member(tar: tarfile.TarFile, path: str) -> str | None:
    for name in (f"./{path}", path):
        try:
            member = tar.getmember(name)
        except KeyError:
            continue
        if not member.isfile():
            return None
        f = tar.extractfile(member)
        if f is None:
            return None
        data = f.read()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return text
    return None


def main() -> None:
    args = parse_args()
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]

    by_repo: OrderedDict[str, list[dict]] = OrderedDict()
    for t in tasks:
        by_repo.setdefault(t["repo"], []).append(t)

    holdout = {t["instance_id"] for ts in by_repo.values() for t in ts[: args.per_repo_holdout]}
    train = [t for t in tasks if t["instance_id"] not in holdout]
    hold = [t for t in tasks if t["instance_id"] in holdout]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)
    (args.split_dir / "train_ids.json").write_text(json.dumps([t["instance_id"] for t in train], indent=2))
    (args.split_dir / "holdout_ids.json").write_text(json.dumps([t["instance_id"] for t in hold], indent=2))

    written = 0
    skipped: list[tuple[str, str]] = []
    with args.out.open("w", encoding="utf-8") as out:
        for i, task in enumerate(tasks, 1):
            iid = task["instance_id"]
            patch = task.get("patch") or ""
            if not patch.strip():
                skipped.append((iid, "empty patch"))
                continue
            paths = changed_paths(patch)
            if not paths:
                skipped.append((iid, "no source paths"))
                continue

            tgz = args.snapshots / f"{iid}.tgz"
            if not tgz.is_file():
                skipped.append((iid, "missing snapshot"))
                continue

            parts = [
                SYSTEM,
                "",
                "# ISSUE",
                task.get("problem_statement", "").strip(),
                "",
                "# SOURCE FILES AT BASE COMMIT",
            ]
            used = 0
            try:
                with tarfile.open(tgz, "r:gz") as tar:
                    for path in paths:
                        content = read_member(tar, path)
                        if content is None:
                            continue
                        if len(content) > args.max_file_chars:
                            content = content[: args.max_file_chars] + "\n# ... truncated ...\n"
                        block = f"\n## {path}\n```\n{content}\n```\n"
                        if used + len(block) > args.max_context_chars:
                            break
                        parts.append(block)
                        used += len(block)
            except (tarfile.TarError, OSError) as e:
                skipped.append((iid, f"snapshot error: {e}"))
                continue

            if used == 0:
                skipped.append((iid, "no source content"))
                continue

            parts.extend(["", "# PATCH", "Return only the unified diff below.", ""])
            prompt = "\n".join(parts)
            out.write(json.dumps({"task_id": iid, "repo": task["repo"], "prompt": prompt, "completion": patch}, ensure_ascii=False) + "\n")
            written += 1
            if i % 20 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} written={written} skipped={len(skipped)}", flush=True)

    print(f"wrote {written} examples to {args.out}")
    print(f"train={len(train)} holdout={len(hold)}")
    if skipped:
        print("skipped:")
        for iid, reason in skipped:
            print(f"  {iid}: {reason}")


if __name__ == "__main__":
    main()
