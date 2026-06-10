"""Stamp a per-candidate copy of Alex's branch and open the PR via gh.

Usage:
    make stamp-manager-pr CANDIDATE=jane-doe
    uv run python -m interviewer.tools.stamp_manager_pr --candidate jane-doe --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PATCH_DIR = REPO / "interviewer" / "patches" / "alex"
PR_BODY = REPO / "interviewer" / "rendered" / "alex-pr-body.md"
BASE_REF = os.environ.get("BASE_REF", "main")


def run(args: list[str], cwd: Path = REPO) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--dry-run", action="store_true", help="build the branch, skip push/PR")
    args = parser.parse_args()

    branch = f"alex/behavioral-features-{args.candidate}"
    gt = json.loads(
        (REPO / "interviewer" / "answer_keys" / "ground_truth.json").read_text()
    )
    title = (
        f"Add customer behavioral features — "
        f"AUC {gt['baseline_auc']:.2f} → {gt['alex_random_auc']:.2f} 🎉"
    )

    tmp = REPO / ".stamp-tmp"
    # idempotent: remove any leftover worktree from a previously failed run
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(tmp)], cwd=REPO, capture_output=True
    )
    run(["git", "worktree", "add", "--detach", str(tmp), BASE_REF])
    try:
        patches = sorted(str(p) for p in PATCH_DIR.glob("*.patch"))
        run(["git", "am", *patches], cwd=tmp)
        run(["git", "branch", "-f", branch, "HEAD"], cwd=tmp)
    finally:
        run(["git", "worktree", "remove", "--force", str(tmp)])

    if args.dry_run:
        print(f"[dry-run] built {branch}; would open PR: {title}")
        return

    # guard: fail clearly if no remote is configured rather than mid-push crash
    remote_check = subprocess.run(
        ["git", "remote", "get-url", "origin"], cwd=REPO, capture_output=True
    )
    if remote_check.returncode != 0:
        print("error: no 'origin' remote configured — run with --dry-run or add a remote first")
        raise SystemExit(1)

    run(["git", "push", "-f", "origin", branch])
    run([
        "gh", "pr", "create",
        "--base", "main", "--head", branch,
        "--title", title, "--body-file", str(PR_BODY),
    ])
    print(f"PR opened for {branch}. Close it and delete the branch right after the session.")


if __name__ == "__main__":
    main()
