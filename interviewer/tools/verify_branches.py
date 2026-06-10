"""Stamp a fabricated branch into a temp worktree and verify its properties.

Usage: uv run python -m interviewer.tools.verify_branches [senior|alex|all]
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GT = json.loads((REPO / "interviewer" / "answer_keys" / "ground_truth.json").read_text())
AUC_TOLERANCE = 0.01
BASE_REF = os.environ.get("BASE_REF", "main")


def _run(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True)


def _stamp(patch_dir: Path, worktree: Path) -> None:
    _run(["git", "worktree", "add", "--detach", str(worktree), BASE_REF], REPO)
    patches = sorted(str(p) for p in patch_dir.glob("*.patch"))
    _run(["git", "am", *patches], worktree)


def _checks(worktree: Path, expectations: dict) -> list[str]:
    failures: list[str] = []

    pytest = _run(["uv", "run", "pytest", "-q"], worktree, check=False)
    if pytest.returncode != 0:
        failures.append(f"pytest failed:\n{pytest.stdout[-2000:]}")
    if expectations.get("skipped"):
        if not re.search(rf"{expectations['skipped']} skipped", pytest.stdout):
            failures.append(f"expected {expectations['skipped']} skipped test(s)")

    ruff = _run(["uv", "run", "ruff", "check", "."], worktree, check=False)
    if ruff.returncode != 0:
        failures.append(f"ruff failed:\n{ruff.stdout[-2000:]}")

    _run(["uv", "run", "python", "models/return-risk/src/train.py"], worktree)
    metrics = json.loads(
        (worktree / "models" / "return-risk" / "artifacts" / "metrics.json").read_text()
    )
    expected_auc = expectations["auc"]
    if abs(metrics["auc"] - expected_auc) > AUC_TOLERANCE:
        failures.append(
            f"train AUC {metrics['auc']:.4f} != expected {expected_auc:.4f} (±{AUC_TOLERANCE})"
        )
    if leak := expectations.get("dominant_feature"):
        if not metrics["gain_share"]:
            failures.append("model produced no feature importances")
        else:
            top = max(metrics["gain_share"], key=metrics["gain_share"].get)
            if top != leak:
                failures.append(f"top importance is {top}, expected {leak}")

    for check_fn in expectations.get("extra", []):
        failures.extend(check_fn(worktree))
    return failures


def _alex_diff_checks(worktree: Path) -> list[str]:
    failures = []
    diff = _run(["git", "diff", f"{BASE_REF}...HEAD"], worktree).stdout
    added = [line for line in diff.splitlines() if line.startswith("+")]
    # Faithful reproduction of the source's literal overlay contents yields a
    # ~44-line collapsed diff after customer_return_rate was removed from the
    # PR (spec A1) -- dropping that commit and its colocated test took the
    # artifact from ~124 to ~44 added lines. Band brackets the real artifact
    # while still catching a truncation or runaway regeneration.
    if not 25 <= len(added) <= 90:
        failures.append(f"diff size {len(added)} added lines; expected 25-90 (artifact ~44)")
    support_diff = _run(
        ["git", "diff", f"{BASE_REF}...HEAD", "--", "feature_catalog/features/support.py"],
        worktree,
    ).stdout
    # the leak must look innocent as text: no timestamp logic in the diff
    body = "\n".join(
        line for line in support_diff.splitlines() if line.startswith("+")
    )
    if re.search(r"_ts|timestamp|Timedelta|date", body, flags=re.IGNORECASE):
        failures.append("support.py diff contains timestamp-ish tokens; must be a plain join-count")
    if (worktree / "feature_catalog" / "features" / "support_test.py").exists():
        failures.append("support.py must ship WITHOUT tests (seeded issue 3a)")
    # customer_return_rate was removed from alex's PR (spec A1): it lifted the
    # leak-removed baseline toward ~0.89 and broke the leak's load-bearing role.
    # It is now the senior's Part-2-only exemplar (in planted_features.py, no
    # alex patch). So customers.py here carries only the base
    # customer_prior_order_count and its single merge/reindex idiom; the
    # copy-paste duplication nit returns with the replacement red herring (TBD).
    customers_text = (worktree / "feature_catalog" / "features" / "customers.py").read_text()
    if "customer_prior_order_count" not in customers_text:
        failures.append("customers.py missing customer_prior_order_count")
    if "customer_return_rate" in customers_text:
        failures.append(
            "customers.py contains customer_return_rate; it must be ABSENT from "
            "alex's PR (spec A1 -- it is the senior Part-2-only exemplar)"
        )
    reindex_count = customers_text.count("reindex(tables.orders")
    if reindex_count < 1:
        failures.append(
            f"base merge/reindex idiom appears {reindex_count} time(s) in customers.py; "
            "expected >=1 (customer_prior_order_count)"
        )
    return failures


BRANCHES = {
    "senior": {
        "patches": REPO / "interviewer" / "patches" / "senior",
        "auc": GT["v13_random_auc"],
        "dominant_feature": "support_contact_count_30d",
    },
    "alex": {
        "patches": REPO / "interviewer" / "patches" / "alex",
        # alex's train.py patch (0002) now uses the same random 80/20 split as
        # senior; the stamped branch's train.py prints alex's own feature-set
        # number (main + leak only, since customer_return_rate was removed per
        # spec A1), which is the alex_random_auc key (distinct from senior's
        # v13_random_auc).
        "auc": GT["alex_random_auc"],
        "dominant_feature": "support_contact_count",
        "skipped": 1,
        "extra": [_alex_diff_checks],
    },
}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(BRANCHES) if which == "all" else [which]
    failed = False
    for name in names:
        spec = BRANCHES[name]
        with tempfile.TemporaryDirectory() as tmp:
            worktree = Path(tmp) / "wt"
            try:
                _stamp(spec["patches"], worktree)
                failures = _checks(worktree, spec)
            finally:
                _run(["git", "worktree", "remove", "--force", str(worktree)], REPO,
                     check=False)
        status = "OK" if not failures else "FAIL"
        print(f"[{status}] {name}")
        for failure in failures:
            print(f"  - {failure}")
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
