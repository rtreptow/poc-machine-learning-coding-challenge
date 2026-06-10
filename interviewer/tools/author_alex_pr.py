"""Build Alex's mock-PR branch from main + planted sources; freeze to patches."""

from __future__ import annotations

import inspect
import os
import shutil
import tempfile
from pathlib import Path

from interviewer.planted import planted_features as pf
from interviewer.tools.author_senior_branch import _commit, _git

REPO = Path(__file__).resolve().parents[2]
PLANTED = REPO / "interviewer" / "planted" / "alex"
PATCH_DIR = REPO / "interviewer" / "patches" / "alex"
BRANCH = "alex/customer-behavioral-features"
BASE_REF = os.environ.get("BASE_REF", "main")

ALEX = ("Alex Mercer", "alex.mercer@example.com")
COMMIT_DATES = [
    "2026-06-02T11:20:00-07:00",
    "2026-06-03T10:45:00-07:00",
    "2026-06-03T14:10:00-07:00",
]


def main() -> None:
    # guard: the shipped support.py must embed the canonical implementation
    support_text = (PLANTED / "support.py").read_text()
    canonical = inspect.getsource(pf.support_contact_count)
    assert canonical in support_text, (
        "alex/support.py drifted from planted_features.support_contact_count"
    )

    with tempfile.TemporaryDirectory() as tmp:
        worktree = Path(tmp) / "wt"
        _git(["worktree", "add", "--detach", str(worktree), BASE_REF], REPO)
        try:
            # ---- commit 1: the leak, as a new module with NO tests ----
            shutil.copy(
                PLANTED / "support.py",
                worktree / "feature_catalog" / "features" / "support.py",
            )
            registry = worktree / "feature_catalog" / "registry.py"
            text = registry.read_text().replace(
                "from feature_catalog.features import customers, orders",
                "from feature_catalog.features import customers, orders, support",
            )
            text = text.replace(
                "}\n", '    "support_contact_count": support.support_contact_count,\n}\n'
            )
            registry.write_text(text)
            _commit(worktree, "feat: support contact friction feature", COMMIT_DATES[0],
                    author=ALEX)

            # ---- commit 2: eval switch + the silenced regression test ----
            src = worktree / "models" / "return-risk" / "src"
            shutil.copy(PLANTED / "train.py", src / "train.py")
            shutil.copy(PLANTED / "train_test.py", src / "train_test.py")
            _commit(worktree, "chore: switch eval to a random split for stability",
                    COMMIT_DATES[1], author=ALEX)

            # ---- commit 3: config -- feature on, threshold quietly moved ----
            config = worktree / "models" / "return-risk" / "feature-configs" / "v1.yaml"
            text = config.read_text().replace("threshold: 0.50", "threshold: 0.35")
            config.write_text(
                text + "  - support_contact_count\n"
            )
            _commit(worktree, "config: enable behavioral features", COMMIT_DATES[2],
                    author=ALEX)

            # ---- freeze ----
            _git(["branch", "-f", BRANCH, "HEAD"], worktree)
            if PATCH_DIR.exists():
                shutil.rmtree(PATCH_DIR)
            PATCH_DIR.mkdir(parents=True)
            _git(
                ["format-patch", "--zero-commit", f"{BASE_REF}..{BRANCH}", "-o", str(PATCH_DIR)],
                REPO,
            )
        finally:
            _git(["worktree", "remove", "--force", str(worktree)], REPO)
    print(f"{BRANCH} frozen -> {PATCH_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
