"""Build the senior/start branch from main + planted sources; freeze to patches.

Run via `make freeze-senior`. Pinned authors/dates make the patches stable,
so re-running after a repo change regenerates them deterministically.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from interviewer.planted import planted_features as pf

REPO = Path(__file__).resolve().parents[2]
PLANTED = REPO / "interviewer" / "planted" / "senior"
PATCH_DIR = REPO / "interviewer" / "patches" / "senior"
BASE_REF = os.environ.get("BASE_REF", "main")

DANA = ("Dana Riggs", "dana.riggs@vector-contracting.example")
COMMIT_DATES = [
    "2026-04-06T10:12:00-07:00",
    "2026-04-06T15:40:00-07:00",
    "2026-04-07T09:05:00-07:00",
]

V13_FEATURES = [
    pf.customer_avg_order_value,
    pf.customer_days_since_last_order,
    pf.weekend_order,
    pf.checkout_hour,
    pf.support_contact_count_30d,
]

# Contractor-grade tests: plausible, passing, semantically blind. The 30d test
# happily counts a POST-checkout contact -- the breadcrumb is in the test itself.
SHALLOW_TESTS = '''

def test_v13_customer_signal_features_smoke() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C1"],
            "checkout_ts": pd.to_datetime(["2026-01-04 09:00", "2026-02-01 18:00"]),
            "order_value": [10.0, 30.0],
            "item_count": [1, 1],
        }
    )
    contacts = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "order_id": ["O1"],
            "contact_ts": pd.to_datetime(["2026-01-07"]),
            "channel": ["email"],
        }
    )
    tables = make_tables(orders=orders, support_contacts=contacts)
    assert customer_avg_order_value(tables).loc["O2"] == 10.0
    assert customer_days_since_last_order(tables).loc["O1"] == -1.0
    assert weekend_order(tables).loc["O1"] == 1.0  # 2026-01-04 is a Sunday
    assert checkout_hour(tables).loc["O2"] == 18.0
    assert support_contact_count_30d(tables).loc["O1"] == 1.0
'''

COMMIT_MESSAGES = [
    "feat: v1.3 customer signal features (contractor sprint)",
    "chore: simplify eval split for stability",
    "feat: enable v1.3 customer signal features (offline AUC {v13_random_auc:.2f})",
]


def _git(
    args: list[str],
    cwd: Path,
    date: str | None = None,
    author: tuple[str, str] = DANA,
) -> None:
    name, email = author
    env = {
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(Path.home()),
    }
    if date:
        env["GIT_AUTHOR_DATE"] = date
        env["GIT_COMMITTER_DATE"] = date
    subprocess.run(["git", *args], cwd=cwd, check=True, env=env)


def _commit(
    worktree: Path, message: str, date: str, author: tuple[str, str] = DANA
) -> None:
    _git(["add", "-A"], worktree, author=author)
    _git(["commit", "-m", message], worktree, date=date, author=author)


def main() -> None:
    gt = json.loads(
        (REPO / "interviewer" / "answer_keys" / "ground_truth.json").read_text()
    )
    with tempfile.TemporaryDirectory() as tmp:
        worktree = Path(tmp) / "wt"
        _git(["worktree", "add", "--detach", str(worktree), BASE_REF], REPO)
        try:
            # ---- commit 1: features + shallow tests + registry entries ----
            customers = worktree / "feature_catalog" / "features" / "customers.py"
            sources = "\n\n".join(inspect.getsource(fn) for fn in V13_FEATURES)
            customers.write_text(customers.read_text() + "\n\n" + sources)

            tests = worktree / "feature_catalog" / "features" / "customers_test.py"
            names = [fn.__name__ for fn in V13_FEATURES]
            # widen the existing import line (appending a new import block
            # mid-file would trip ruff E402 on the branch)
            new_import = (
                "from feature_catalog.features.customers import (\n    "
                + ",\n    ".join(sorted(["customer_prior_order_count", *names]))
                + ",\n)"
            )
            tests.write_text(
                tests.read_text().replace(
                    "from feature_catalog.features.customers import customer_prior_order_count",
                    new_import,
                )
                + SHALLOW_TESTS
            )

            registry = worktree / "feature_catalog" / "registry.py"
            entries = "".join(f'    "{n}": customers.{n},\n' for n in names)
            registry.write_text(registry.read_text().replace("}\n", entries + "}\n"))
            _commit(worktree, COMMIT_MESSAGES[0], COMMIT_DATES[0])

            # ---- commit 2: eval "simplification" ----
            src = worktree / "models" / "return-risk" / "src"
            shutil.copy(PLANTED / "train.py", src / "train.py")
            shutil.copy(PLANTED / "train_test.py", src / "train_test.py")
            _commit(worktree, COMMIT_MESSAGES[1], COMMIT_DATES[1])

            # ---- commit 3: config bump, confident message with the number ----
            config = (
                worktree / "models" / "return-risk" / "feature-configs" / "v1.yaml"
            )
            config.write_text(
                config.read_text() + "".join(f"  - {n}\n" for n in names)
            )
            _commit(
                worktree,
                COMMIT_MESSAGES[2].format(v13_random_auc=gt["v13_random_auc"]),
                COMMIT_DATES[2],
            )

            # ---- freeze ----
            _git(["branch", "-f", "senior/start", "HEAD"], worktree)
            if PATCH_DIR.exists():
                shutil.rmtree(PATCH_DIR)
            PATCH_DIR.mkdir(parents=True)
            _git(
                [
                    "format-patch",
                    "--zero-commit",
                    f"{BASE_REF}..senior/start",
                    "-o",
                    str(PATCH_DIR),
                ],
                REPO,
            )
        finally:
            _git(["worktree", "remove", "--force", str(worktree)], REPO)
    print(f"senior/start frozen -> {PATCH_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
