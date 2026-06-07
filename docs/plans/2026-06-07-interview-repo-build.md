# ML Interview Repo Implementation Plan

> **For agentic workers:** Implement this plan task-by-task — one task at a time, verifying each before moving to the next. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the return-risk interview repo described in `docs/specs/technical-interview-design.md`: a clean ML repo on `main`, a calibrated synthetic-data generator, two reproducible planted artifacts (the senior `v1.3` commit series and Alex's mock PR), and the interviewer kit (tickets, answer keys, briefs, rubrics) rendered from generator output.

**Architecture:** One repo, two layers. The candidate-visible layer (`feature_catalog/`, `models/return-risk/`, `data/`, `docs/`, `Makefile`) is a small, honest pandas/sklearn/XGBoost service. The interviewer-only layer (`interviewer/`) holds planted feature implementations (single source of truth), ground-truth computation, calibration gates, patch-authoring scripts, and rendered docs. Every narrative number (0.92 / 0.87 / 0.84 / 0.62 / 0.96 / 0.85 / Part-2 lift) is **computed, never hand-typed**: the generator's knobs are tuned in an explicit calibration loop until all numbers co-occur at one pinned seed, then templated into tickets, answer keys, the model card, and commit messages.

**Tech Stack:** Python ≥3.12 via `uv`, pandas, scikit-learn, XGBoost, Pydantic, PyYAML, pytest, ruff. Git + `gh` for stamping.

---

## Stated assumptions (flagged to the user, not blocking)

1. **POC topology — everything lives in this one repo**, including `interviewer/` (answer keys, patches, briefs) and `docs/specs/`. A real candidate could read all of it. That's acceptable for the POC; production hardening (separate private repo or branch-excluded paths for interviewer material) is a listed follow-up, consistent with the spec's leak-posture section.
2. **Spec's parked questions adopted as leaning-stated:** Part 2 delivered in the same PR; senior candidates use branches in the upstream repo (not forks); nudge wording/timing finalized at dry-run (the run sheet carries the protocol with wording marked "finalize at dry run" — that's spec-mandated deferral, not a plan gap).
3. **The smoke-test repo is a separate deliverable** (different repo, no task content) and is out of scope here — see Follow-ups.
4. **Determinism is per-machine.** XGBoost `hist` + fixed seeds is deterministic on one machine; cross-machine float jitter is why calibration asserts *bands*, not exact values.
5. **Feature functions receive the full `Tables` object** and do their own joins/groupbys — diverging from the spec's earlier "hands feature functions a per-order frame" phrasing. Point-in-time history features need raw tables to apply cutoffs correctly, and the `customer_prior_order_count` exemplar makes the join pattern copyable; a pre-joined per-order frame would make the exemplar (and Part 2) worse. The spec line has been updated to match.

## Conventions for the implementing engineer

- Run everything through `uv run …` (or the Makefile targets that wrap it). Never `pip install`.
- Commit messages: conventional commits (`feat:`, `chore:`, `test:`…). Append the standard `Co-Authored-By: Claude …` trailer to commits **you** make. The fabricated commits authored by the stamping scripts (Dana, Alex) must **NOT** carry that trailer — they are in-fiction artifacts with pinned fictional authors.
- The repo's default branch is currently `master`; Task 1 renames it to `main` (the spec's branch names depend on it).
- `data/*.csv` are committed (the spec says candidates audit "the shipped CSVs"). They are regenerated and re-committed whenever generator knobs change.
- Candidate-visible tests must never assert planted texture (e.g. "90% of contacts post-date checkout") — those assertions live only in `interviewer/`.

## Execution protocol (for autonomous runs)

- Tasks run strictly in order. A task is done only when its acceptance commands exit 0.
- On any failed acceptance gate that survives a reasonable fix attempt: STOP. Write `docs/plans/BLOCKED.md` containing the task number, the failing command, its full output, and what was attempted. Do not proceed past a failed gate.
- Anywhere this plan says "surface it" / "STOP and surface it", that means: write `docs/plans/BLOCKED.md` as above and stop.
- If a task's gate passed but a later task reveals the earlier output was wrong, fix forward in the current task; don't rewrite history.

## File structure

```
pyproject.toml                          # uv project; deps; pytest/ruff config
Makefile                                # data/train/eval/test/lint + interviewer targets
README.md                               # quickstart (candidate-facing)
data/
  generate.py                           # deterministic generator, SEED=412, named knobs
  generate_test.py                      # determinism + schema only (no texture spoilers)
  orders.csv  order_items.csv  returns.csv  support_contacts.csv   # committed
feature_catalog/
  __init__.py
  types.py                              # Tables, FeatureConfig (Pydantic)
  testing.py                            # make_tables() toy-table builder for tests
  registry.py                           # FEATURES: name -> function
  frame.py                              # build_training_frame()
  features/
    __init__.py
    orders.py        orders_test.py     # order-shape features
    customers.py     customers_test.py  # customer_prior_order_count (THE exemplar)
models/return-risk/
  feature-configs/v1.yaml               # features + label horizon + threshold
  src/
    train.py         train_test.py      # temporal split, XGBoost, AUC, importances
    predict.py       predict_test.py    # thin inference surface
  MODEL_CARD.md                         # RENDERED by render_docs.py (numbers templated)
  artifacts/                            # gitignored (model.json, metrics.json)
docs/
  CONTRIBUTING.md                       # eval standards + review norms (ambient signal)
  specs/  plans/                        # already present / this file
interviewer/                            # POC: committed; production: private (follow-up)
  __init__.py
  planted/
    __init__.py
    planted_features.py                 # ALL planted feature impls — single source
    planted_features_test.py            # cutoff tests (become Alex's colocated test)
    senior/train.py  senior/train_test.py        # senior-branch overlays
    alex/support.py  alex/train.py  alex/train_test.py  # Alex-branch overlays
  tools/
    __init__.py
    compute_ground_truth.py             # trains all variants -> ground_truth.json
    check_calibration.py                # band + ordering + texture assertions
    render_docs.py                      # templates -> model card, tickets, keys, PR body
    author_senior_branch.py             # builds senior/start, freezes patches
    author_alex_pr.py                   # builds Alex's branch, freezes patches
    verify_branches.py                  # stamps branches into worktrees and checks them
    stamp_manager_pr.py                 # per-candidate branch + gh pr create
  templates/                            # string.Template sources for render_docs.py
    model_card.md  risk-412.md  part2-ticket.md
    senior-answer-key.md  manager-answer-key.md  alex-pr-body.md
  answer_keys/ground_truth.json         # emitted by compute_ground_truth.py
  tickets/  rendered/                   # rendered outputs (committed)
  briefs/priya.md  briefs/alex.md       # persona briefs (hand-written, no numbers)
  rubrics/senior.md  rubrics/manager.md # numbers templated where present
  run_sheets/senior.md  run_sheets/manager.md
  patches/senior/*.patch  patches/alex/*.patch   # frozen commit series
  redteam/PROTOCOL.md                   # two-sided AI gate procedure + results
```

Colocated `*_test.py` files are omitted from the diagram for brevity — they are specified per-task.

**Phases:** A (Tasks 1–9) clean repo; B (10–14) ground truth, calibration, rendered docs; C (15–18) planted branches; D (19–21) interviewer kit + gates. A natural split point exists after Task 14 if this is executed as two efforts.

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `Makefile`, `README.md` (stub), `feature_catalog/__init__.py`, `feature_catalog/features/__init__.py`
- Branch: rename `master` → `main`

- [ ] **Step 1: Rename the default branch**

```bash
git branch -m master main
```

If the repo is local-only (`git remote get-url origin` fails), just the rename — skip the remote steps; that's fine.

If a remote exists:
```bash
git push -u origin main
gh repo edit --default-branch main
# verify:
gh repo view --json defaultBranchRef -q .defaultBranchRef.name
# expected output: main
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "return-risk"
version = "0.1.0"
description = "Return/abuse risk at checkout - feature catalog and return-risk model"
requires-python = ">=3.12"
dependencies = [
    "pandas>=2.2",
    "scikit-learn>=1.5",
    "xgboost>=2.1",
    "pydantic>=2.7",
    "pyyaml>=6.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.6",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["feature_catalog"]

[tool.pytest.ini_options]
# Repo root on sys.path so `data.*` and `interviewer.*` import without packaging.
pythonpath = ["."]
# Candidate-facing suite only; interviewer/ tests run explicitly via Makefile targets.
testpaths = ["feature_catalog", "models", "data"]

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

- [ ] **Step 3: Write `.gitignore`**

```
.venv/
__pycache__/
*.egg-info/
models/return-risk/artifacts/
.stamp-tmp/
```

(Note: `data/*.csv` is deliberately **not** ignored.)

- [ ] **Step 4: Write the Makefile (candidate targets only for now)**

```make
.PHONY: data train eval test lint

data:
	uv run python data/generate.py

train:
	uv run python models/return-risk/src/train.py

eval:
	uv run python models/return-risk/src/train.py --eval-only

test:
	uv run pytest

lint:
	uv run ruff check .
```

- [ ] **Step 5: Write `README.md` stub**

```markdown
# return-risk

Return/abuse risk at checkout: a feature catalog and a binary classifier
predicting, at checkout time, whether an order will be returned.

## Quickstart

    uv sync
    make data     # generate the four CSVs into data/
    make train    # train + evaluate (temporal holdout)
    make test
    make lint

See `models/return-risk/MODEL_CARD.md` for the eval protocol and
`docs/CONTRIBUTING.md` for feature/eval standards.
```

- [ ] **Step 6: Create empty package files**

`feature_catalog/__init__.py` and `feature_catalog/features/__init__.py`, each containing nothing (empty file).

- [ ] **Step 7: Verify the environment resolves**

Run: `uv sync && uv run python -c "import pandas, sklearn, xgboost, pydantic, yaml; print('ok')"`
Expected: `ok`

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: scaffold uv project, Makefile, package skeleton"
```

---

## Task 2: Domain types + test toy-table builder

**Files:**
- Create: `feature_catalog/types.py`
- Create: `feature_catalog/testing.py`
- Test: `feature_catalog/types_test.py`

- [ ] **Step 1: Write the failing test**

`feature_catalog/types_test.py`:

```python
from pathlib import Path

import pandas as pd

from feature_catalog.types import FeatureConfig, Tables


def test_tables_load_parses_timestamps(tmp_path: Path) -> None:
    (tmp_path / "orders.csv").write_text(
        "order_id,customer_id,checkout_ts,order_value,item_count\n"
        "O1,C1,2026-01-05 10:00:00,42.50,2\n"
    )
    (tmp_path / "order_items.csv").write_text(
        "order_id,product_id,size,qty,price\nO1,P1,M,1,42.50\n"
    )
    (tmp_path / "returns.csv").write_text(
        "order_id,return_ts,reason\nO1,2026-01-12 09:00:00,wrong_size\n"
    )
    (tmp_path / "support_contacts.csv").write_text(
        "customer_id,order_id,contact_ts,channel\nC1,O1,2026-01-11 16:00:00,email\n"
    )

    tables = Tables.load(tmp_path)

    assert tables.orders["checkout_ts"].dtype.kind == "M"
    assert tables.returns["return_ts"].dtype.kind == "M"
    assert tables.support_contacts["contact_ts"].dtype.kind == "M"
    assert len(tables.order_items) == 1


def test_feature_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "v1.yaml"
    path.write_text(
        "model: return-risk\nlabel_horizon_days: 60\nthreshold: 0.5\n"
        "features:\n  - order_value\n"
    )
    config = FeatureConfig.load(path)
    assert config.features == ["order_value"]
    assert config.label_horizon_days == 60
    assert config.threshold == 0.5
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest feature_catalog/types_test.py -q`
Expected: FAIL (`ModuleNotFoundError`/`ImportError` for `feature_catalog.types`)

- [ ] **Step 3: Write `feature_catalog/types.py`**

```python
"""Domain models: the four raw tables and the model's feature config."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict


class Tables(BaseModel):
    """The whole data world: four frames keyed by order_id / customer_id."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    orders: pd.DataFrame
    order_items: pd.DataFrame
    returns: pd.DataFrame
    support_contacts: pd.DataFrame

    @classmethod
    def load(cls, data_dir: Path) -> "Tables":
        return cls(
            orders=pd.read_csv(data_dir / "orders.csv", parse_dates=["checkout_ts"]),
            order_items=pd.read_csv(data_dir / "order_items.csv"),
            returns=pd.read_csv(data_dir / "returns.csv", parse_dates=["return_ts"]),
            support_contacts=pd.read_csv(
                data_dir / "support_contacts.csv", parse_dates=["contact_ts"]
            ),
        )


class FeatureConfig(BaseModel):
    """A model's feature selection plus label/decision settings."""

    model: str
    features: list[str]
    label_horizon_days: int = 60
    threshold: float = 0.5

    @classmethod
    def load(cls, path: Path) -> "FeatureConfig":
        return cls.model_validate(yaml.safe_load(path.read_text()))
```

- [ ] **Step 4: Write `feature_catalog/testing.py`** (used by every feature test from Task 4 on)

```python
"""Toy-table builder so feature tests stay tiny and explicit."""

from __future__ import annotations

import pandas as pd

from feature_catalog.types import Tables

_EMPTY = {
    "orders": ["order_id", "customer_id", "checkout_ts", "order_value", "item_count"],
    "order_items": ["order_id", "product_id", "size", "qty", "price"],
    "returns": ["order_id", "return_ts", "reason"],
    "support_contacts": ["customer_id", "order_id", "contact_ts", "channel"],
}


def make_tables(
    *,
    orders: pd.DataFrame | None = None,
    order_items: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    support_contacts: pd.DataFrame | None = None,
) -> Tables:
    """Build a Tables with empty-but-correctly-shaped defaults for omitted frames."""
    given = {
        "orders": orders,
        "order_items": order_items,
        "returns": returns,
        "support_contacts": support_contacts,
    }
    frames = {}
    for name, frame in given.items():
        if frame is None:
            frame = pd.DataFrame(columns=_EMPTY[name])
            for col in frame.columns:
                if col.endswith("_ts"):
                    frame[col] = pd.to_datetime(frame[col])
        frames[name] = frame
    return Tables(**frames)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest feature_catalog/ -q`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add feature_catalog/
git commit -m "feat: Tables/FeatureConfig domain types and test table builder"
```

---

## Task 3: Synthetic data generator

**Files:**
- Create: `data/generate.py`
- Test: `data/generate_test.py`
- Output (committed): `data/orders.csv`, `data/order_items.csv`, `data/returns.csv`, `data/support_contacts.csv`

The generator is the riskiest component: its knobs must eventually make ~8 narrative numbers co-occur at one seed. **This task only builds the mechanism with reasonable initial knob values.** Calibration happens in Task 13 — do not chase AUC targets here. Knob values below are initial guesses, labeled as such in code.

Candidate-visible tests check determinism and schema **only** — texture assertions (contact timing, drift, serial segment) live in `interviewer/tools/check_calibration.py` (Task 12) so the test suite doesn't spoil the traps.

- [ ] **Step 1: Write the failing test**

`data/generate_test.py`:

```python
import pandas as pd

from data.generate import generate


def test_deterministic() -> None:
    a = generate(seed=412)
    b = generate(seed=412)
    for name in ("orders", "order_items", "returns", "support_contacts"):
        pd.testing.assert_frame_equal(a[name], b[name])


def test_schema_and_keys() -> None:
    frames = generate(seed=412)
    orders = frames["orders"]

    assert list(orders.columns) == [
        "order_id", "customer_id", "checkout_ts", "order_value", "item_count",
    ]
    assert list(frames["order_items"].columns) == [
        "order_id", "product_id", "size", "qty", "price",
    ]
    assert list(frames["returns"].columns) == ["order_id", "return_ts", "reason"]
    assert list(frames["support_contacts"].columns) == [
        "customer_id", "order_id", "contact_ts", "channel",
    ]

    assert orders["order_id"].is_unique
    assert 40_000 <= len(orders) <= 60_000
    assert frames["returns"]["order_id"].isin(orders["order_id"]).all()
    assert frames["support_contacts"]["order_id"].isin(orders["order_id"]).all()
    assert frames["order_items"]["order_id"].isin(orders["order_id"]).all()

    # every return happens after its order's checkout
    joined = frames["returns"].merge(
        orders[["order_id", "checkout_ts"]], on="order_id"
    )
    assert (joined["return_ts"] > joined["checkout_ts"]).all()
```

Note: `data/` has no `__init__.py`; pytest's default rootdir handling plus `testpaths` makes `from data.generate import generate` work because the repo root is on `sys.path` (pytest inserts rootdir for rootdir-relative imports when invoked from the root). If the import fails in practice, add an empty `data/__init__.py` — acceptable.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest data/generate_test.py -q`
Expected: FAIL (`ModuleNotFoundError: data.generate`)

- [ ] **Step 3: Write `data/generate.py`**

```python
"""Deterministic synthetic data for the return-risk repo.

`make data` runs this; the same SEED always produces identical CSVs.

The world it builds:
- ~50k orders across an 18-month window, ~12k customers.
- A serial-returner segment that orders more often, bracket-buys sizes,
  and returns far more often -- their history is genuinely predictive.
- Base return rates rise and customer mix shifts across the window
  (temporal drift).
- Support contacts: a small share happen before checkout (pre-purchase
  friction, mildly predictive of a return); most happen after checkout,
  prompted by the return itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

SEED = 412

# ---- volume / window ----
N_CUSTOMERS = 12_000
TARGET_ORDERS = 50_000
WINDOW_START = pd.Timestamp("2024-11-01")
WINDOW_END = pd.Timestamp("2026-05-01")  # 18 months

# ---- calibration knobs ----
# Initial guesses. Tuned during the calibration loop against
# interviewer/tools/check_calibration.py; do not treat as final.
SERIAL_FRAC = 0.12              # share of customers in the serial-returner segment
SERIAL_ORDER_RATE_MULT = 2.5    # serial customers order more often
SERIAL_LATE_SKEW = 1.6          # >1 skews serial order volume toward window end (mix drift)
SERIAL_LOGIT_BOOST = 1.4        # added to the return logit for serial customers
SERIAL_MULTI_SIZE_P = 0.65      # P(bracket-buy: same product in 2+ sizes | serial)
BASE_MULTI_SIZE_P = 0.10

BASE_RETURN_LOGIT_START = -1.9  # base return logit at window start ...
BASE_RETURN_LOGIT_END = -1.1    # ... and at window end (rates drift upward)

W_LOG_ORDER_VALUE = 0.35        # return-logit weights on observable order shape
W_ITEM_COUNT = 0.20
W_MULTI_SIZE = 1.1
LATENT_SD = 0.9                 # per-customer latent propensity: invisible in order
                                # shape, recoverable from return history (Part 2 lift)

PRE_CONTACT_P = 0.03            # P(pre-checkout support contact on an order)
PRE_CONTACT_LOGIT_LIFT = 0.9    # genuine signal: pre-purchase friction -> returns
POST_CONTACT_P_RETURNED = 0.80  # returned orders usually generate a contact ...
POST_CONTACT_P_KEPT = 0.08      # ... kept orders rarely do

SIZES = ["XS", "S", "M", "L", "XL"]
CHANNELS = ["email", "chat", "phone"]
RETURN_REASONS = ["wrong_size", "changed_mind", "damaged", "not_as_described"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / values.std()


def generate(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    window_days = (WINDOW_END - WINDOW_START).days

    # ---- customers ----
    serial = rng.random(N_CUSTOMERS) < SERIAL_FRAC
    latent = rng.normal(0.0, LATENT_SD, N_CUSTOMERS)

    # ---- orders: who, when ----
    base_rate = TARGET_ORDERS / (
        N_CUSTOMERS * (1 - SERIAL_FRAC + SERIAL_FRAC * SERIAL_ORDER_RATE_MULT)
    )
    n_per_customer = rng.poisson(
        np.where(serial, base_rate * SERIAL_ORDER_RATE_MULT, base_rate)
    )
    cust_idx = np.repeat(np.arange(N_CUSTOMERS), n_per_customer)
    n = len(cust_idx)
    order_serial = serial[cust_idx]
    # Serial order volume skews late in the window -> customer-mix drift.
    t_frac = np.where(
        order_serial,
        rng.beta(SERIAL_LATE_SKEW, 1.0, n),
        rng.random(n),
    )
    checkout_ts = (
        WINDOW_START + pd.to_timedelta(t_frac * window_days, unit="D")
    ).round("s")

    orders = pd.DataFrame(
        {
            "customer_idx": cust_idx,
            "serial": order_serial,
            "t_frac": t_frac,
            "checkout_ts": checkout_ts,
        }
    ).sort_values("checkout_ts", ignore_index=True)
    orders["order_id"] = [f"O{i:06d}" for i in range(len(orders))]
    orders["customer_id"] = orders["customer_idx"].map(lambda i: f"C{i:05d}")
    n = len(orders)

    # ---- line items (drive order_value / item_count / bracket-buy texture) ----
    item_counts = 1 + rng.poisson(1.2, n)
    multi_size = rng.random(n) < np.where(
        orders["serial"], SERIAL_MULTI_SIZE_P, BASE_MULTI_SIZE_P
    )
    rows: list[tuple[str, str, str, int, float]] = []
    for oid, k, bracket in zip(orders["order_id"], item_counts, multi_size):
        products = [f"P{rng.integers(0, 2000):04d}" for _ in range(k)]
        sizes = [str(s) for s in rng.choice(SIZES, size=k)]
        prices = np.round(np.exp(rng.normal(3.4, 0.5, k)), 2)
        if bracket:  # same product again in a second size
            alt = [s for s in SIZES if s != sizes[0]]
            products.append(products[0])
            sizes.append(str(rng.choice(alt)))
            prices = np.append(prices, prices[0])
        rows.extend(
            (oid, p, s, 1, float(pr)) for p, s, pr in zip(products, sizes, prices)
        )
    order_items = pd.DataFrame(
        rows, columns=["order_id", "product_id", "size", "qty", "price"]
    )
    agg = order_items.groupby("order_id").agg(
        order_value=("price", "sum"), item_count=("qty", "sum")
    )
    orders = orders.merge(agg, on="order_id")

    # ---- returns ----
    pre_contact = rng.random(n) < PRE_CONTACT_P
    logit = (
        BASE_RETURN_LOGIT_START
        + (BASE_RETURN_LOGIT_END - BASE_RETURN_LOGIT_START) * orders["t_frac"].values
        + np.where(orders["serial"], SERIAL_LOGIT_BOOST, 0.0)
        + latent[orders["customer_idx"]]
        + W_LOG_ORDER_VALUE * _zscore(np.log(orders["order_value"].values))
        + W_ITEM_COUNT * _zscore(orders["item_count"].values.astype(float))
        + W_MULTI_SIZE * multi_size
        + PRE_CONTACT_LOGIT_LIFT * pre_contact
    )
    returned = rng.random(n) < _sigmoid(logit)
    delay_days = rng.gamma(2.0, 6.0, n)  # mean ~12 days; tail past the 60d label horizon
    return_ts = (orders["checkout_ts"] + pd.to_timedelta(delay_days, unit="D")).round("s")
    wrong_size_bias = returned & multi_size & (rng.random(n) < 0.7)
    reasons = np.where(
        wrong_size_bias, "wrong_size", rng.choice(RETURN_REASONS, size=n)
    )
    returns = pd.DataFrame(
        {
            "order_id": orders.loc[returned, "order_id"],
            "return_ts": return_ts[returned],
            "reason": reasons[returned],
        }
    ).reset_index(drop=True)

    # ---- support contacts ----
    contact_frames = []
    # pre-checkout: sizing/fit questions before buying (mild genuine signal)
    pre = orders.loc[pre_contact, ["customer_id", "order_id", "checkout_ts"]].copy()
    pre["contact_ts"] = (
        pre["checkout_ts"]
        - pd.to_timedelta(rng.uniform(0.05, 5.0, len(pre)), unit="D")
    ).round("s")
    contact_frames.append(pre)
    # post-checkout, returned orders: the contact is about the return
    ret_mask = returned & (rng.random(n) < POST_CONTACT_P_RETURNED)
    post_r = orders.loc[ret_mask, ["customer_id", "order_id", "checkout_ts"]].copy()
    post_r["contact_ts"] = (
        post_r["checkout_ts"]
        + pd.to_timedelta(
            delay_days[ret_mask] * rng.uniform(0.6, 1.15, len(post_r)), unit="D"
        )
    ).round("s")
    contact_frames.append(post_r)
    # post-checkout, kept orders: ordinary where-is-my-stuff noise
    kept_mask = ~returned & (rng.random(n) < POST_CONTACT_P_KEPT)
    post_k = orders.loc[kept_mask, ["customer_id", "order_id", "checkout_ts"]].copy()
    post_k["contact_ts"] = (
        post_k["checkout_ts"]
        + pd.to_timedelta(rng.uniform(1.0, 20.0, len(post_k)), unit="D")
    ).round("s")
    contact_frames.append(post_k)

    support_contacts = pd.concat(contact_frames, ignore_index=True)
    support_contacts["channel"] = rng.choice(
        CHANNELS, size=len(support_contacts), p=[0.5, 0.35, 0.15]
    )
    support_contacts = support_contacts[
        ["customer_id", "order_id", "contact_ts", "channel"]
    ].sort_values(["contact_ts", "order_id"], ignore_index=True)

    orders_out = orders[
        ["order_id", "customer_id", "checkout_ts", "order_value", "item_count"]
    ].copy()
    orders_out["order_value"] = orders_out["order_value"].round(2)

    return {
        "orders": orders_out,
        "order_items": order_items,
        "returns": returns,
        "support_contacts": support_contacts,
    }


def main() -> None:
    frames = generate()
    for name, frame in frames.items():
        path = DATA_DIR / f"{name}.csv"
        frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
        print(f"wrote {path.name}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest data/ -q`
Expected: 2 passed (the determinism test takes ~10–30s; that's fine)

- [ ] **Step 5: Generate and commit the CSVs**

Run: `make data`
Expected: four `wrote <name>.csv: N rows` lines; orders between 40k–60k.

```bash
git add data/
git commit -m "feat: deterministic synthetic data generator with planted texture knobs"
```

---

## Task 4: Order-shape features

**Files:**
- Create: `feature_catalog/features/orders.py`
- Test: `feature_catalog/features/orders_test.py`

- [ ] **Step 1: Write the failing tests**

`feature_catalog/features/orders_test.py`:

```python
import pandas as pd

from feature_catalog.features.orders import (
    avg_item_price,
    item_count,
    max_sizes_per_product,
    order_value,
    unique_product_count,
)
from feature_catalog.testing import make_tables


def _toy() -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C2"],
            "checkout_ts": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "order_value": [30.0, 100.0],
            "item_count": [1, 3],
        }
    )
    order_items = pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O2", "O2"],
            "product_id": ["P1", "P2", "P2", "P3"],
            "size": ["M", "S", "M", "L"],
            "qty": [1, 1, 1, 1],
            "price": [30.0, 25.0, 25.0, 50.0],
        }
    )
    return orders, order_items


def test_order_value_and_item_count_pass_through() -> None:
    orders, order_items = _toy()
    tables = make_tables(orders=orders, order_items=order_items)
    assert order_value(tables).loc["O2"] == 100.0
    assert item_count(tables).loc["O2"] == 3.0


def test_unique_product_count() -> None:
    orders, order_items = _toy()
    tables = make_tables(orders=orders, order_items=order_items)
    result = unique_product_count(tables)
    assert result.loc["O1"] == 1.0
    assert result.loc["O2"] == 2.0  # P2 twice counts once


def test_max_sizes_per_product_catches_bracket_buys() -> None:
    orders, order_items = _toy()
    tables = make_tables(orders=orders, order_items=order_items)
    result = max_sizes_per_product(tables)
    assert result.loc["O1"] == 1.0
    assert result.loc["O2"] == 2.0  # P2 ordered in S and M


def test_avg_item_price() -> None:
    orders, order_items = _toy()
    tables = make_tables(orders=orders, order_items=order_items)
    assert avg_item_price(tables).loc["O2"] == (25.0 + 25.0 + 50.0) / 3
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest feature_catalog/features/orders_test.py -q`
Expected: FAIL (`ModuleNotFoundError: feature_catalog.features.orders`)

- [ ] **Step 3: Write `feature_catalog/features/orders.py`**

Every feature function in the catalog has the same shape: `(tables: Tables) -> pd.Series`, where the result is indexed by `order_id` and covers every order (fill with 0/NaN-safe defaults). This convention is what config-driven selection relies on.

```python
"""Order-shape features: what the order itself looks like at checkout."""

from __future__ import annotations

import pandas as pd

from feature_catalog.types import Tables


def order_value(tables: Tables) -> pd.Series:
    """Total order value at checkout."""
    return tables.orders.set_index("order_id")["order_value"].astype(float)


def item_count(tables: Tables) -> pd.Series:
    """Number of units in the order."""
    return tables.orders.set_index("order_id")["item_count"].astype(float)


def unique_product_count(tables: Tables) -> pd.Series:
    """Distinct products in the order."""
    counts = tables.order_items.groupby("order_id")["product_id"].nunique()
    return counts.reindex(tables.orders["order_id"], fill_value=0).astype(float)


def max_sizes_per_product(tables: Tables) -> pd.Series:
    """Max distinct sizes of any single product in the order -- the bracket-buy tell."""
    sizes = (
        tables.order_items.groupby(["order_id", "product_id"])["size"]
        .nunique()
        .groupby("order_id")
        .max()
    )
    return sizes.reindex(tables.orders["order_id"], fill_value=0).astype(float)


def avg_item_price(tables: Tables) -> pd.Series:
    """Mean unit price across the order's line items."""
    means = tables.order_items.groupby("order_id")["price"].mean()
    return means.reindex(tables.orders["order_id"], fill_value=0.0).astype(float)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest feature_catalog/features/orders_test.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add feature_catalog/features/
git commit -m "feat: order-shape features (value, items, products, bracket-buy, price)"
```

---

## Task 5: The point-in-time exemplar feature

**Files:**
- Create: `feature_catalog/features/customers.py`
- Test: `feature_catalog/features/customers_test.py`

This is the **load-bearing exemplar** (spec: Part 2 silent test, Alex's red herring). Two requirements beyond correctness:
1. The as-of cutoff must be *legible* — an explicit `< checkout_ts` comparison, named and commented, not hidden inside `cumcount()`.
2. The colocated test must assert the cutoff explicitly (future orders excluded), because Alex's PR and the senior's Part 2 are graded against candidates recognizing this pattern.

Note what the exemplar deliberately does NOT show: a cutoff on *event realization* (it only filters when prior orders were *placed*). A return-history feature additionally needs `return_ts < checkout_ts` — that gap is part of the interview design.

- [ ] **Step 1: Write the failing tests**

`feature_catalog/features/customers_test.py`:

```python
import pandas as pd

from feature_catalog.features.customers import customer_prior_order_count
from feature_catalog.testing import make_tables


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O3", "O9"],
            "customer_id": ["C1", "C1", "C1", "C2"],
            "checkout_ts": pd.to_datetime(
                ["2026-01-01", "2026-02-01", "2026-03-01", "2026-02-15"]
            ),
            "order_value": [10.0, 20.0, 30.0, 40.0],
            "item_count": [1, 1, 1, 1],
        }
    )


def test_counts_only_orders_placed_before_checkout() -> None:
    result = customer_prior_order_count(make_tables(orders=_orders()))
    assert result.loc["O1"] == 0.0  # first order: nothing prior
    assert result.loc["O2"] == 1.0
    assert result.loc["O3"] == 2.0


def test_never_counts_other_customers() -> None:
    result = customer_prior_order_count(make_tables(orders=_orders()))
    assert result.loc["O9"] == 0.0  # C2's only order; C1's history must not bleed in
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest feature_catalog/features/customers_test.py -q`
Expected: FAIL (import error)

- [ ] **Step 3: Write `feature_catalog/features/customers.py`**

```python
"""Customer-history features.

Convention for anything derived from a customer's history: the feature must be
computable AT CHECKOUT TIME for the order being scored. Only use events that
happened strictly before the order's checkout_ts.
"""

from __future__ import annotations

import pandas as pd

from feature_catalog.types import Tables


def customer_prior_order_count(tables: Tables) -> pd.Series:
    """How many orders this customer placed before this order's checkout.

    Point-in-time: only orders with checkout_ts strictly before the current
    order's checkout_ts count -- nothing from the future leaks in.
    """
    orders = tables.orders[["order_id", "customer_id", "checkout_ts"]]
    pairs = orders.merge(orders, on="customer_id", suffixes=("", "_other"))
    # the as-of cutoff: the other order must already exist at this checkout
    prior = pairs[pairs["checkout_ts_other"] < pairs["checkout_ts"]]
    counts = prior.groupby("order_id").size()
    return counts.reindex(tables.orders["order_id"], fill_value=0).astype(float)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest feature_catalog/features/customers_test.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add feature_catalog/features/customers.py feature_catalog/features/customers_test.py
git commit -m "feat: customer_prior_order_count point-in-time exemplar with cutoff test"
```

---

## Task 6: Registry + training frame

**Files:**
- Create: `feature_catalog/registry.py`
- Create: `feature_catalog/frame.py`
- Test: `feature_catalog/frame_test.py`

- [ ] **Step 1: Write the failing test**

`feature_catalog/frame_test.py`:

```python
import pandas as pd

from feature_catalog.frame import build_training_frame
from feature_catalog.testing import make_tables


def _tables():
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C1"],
            "checkout_ts": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "order_value": [30.0, 60.0],
            "item_count": [1, 2],
        }
    )
    returns = pd.DataFrame(
        {
            # O1 returned within 60d -> label 1; O2 returned after 60d -> label 0
            "order_id": ["O1", "O2"],
            "return_ts": pd.to_datetime(["2026-01-20", "2026-05-01"]),
            "reason": ["wrong_size", "changed_mind"],
        }
    )
    return make_tables(orders=orders, returns=returns)


def test_label_is_return_within_horizon() -> None:
    frame = build_training_frame(_tables(), ["order_value"], label_horizon_days=60)
    by_id = frame.set_index("order_id")
    assert by_id.loc["O1", "label"] == 1
    assert by_id.loc["O2", "label"] == 0


def test_features_and_metadata_columns_present() -> None:
    frame = build_training_frame(_tables(), ["order_value", "item_count"])
    assert {"order_id", "customer_id", "checkout_ts", "label", "order_value",
            "item_count"} <= set(frame.columns)
    assert len(frame) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest feature_catalog/frame_test.py -q`
Expected: FAIL (import error)

- [ ] **Step 3: Write `feature_catalog/registry.py`**

```python
"""Name -> feature function. feature-configs/*.yaml select features by these names."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from feature_catalog.features import customers, orders
from feature_catalog.types import Tables

FeatureFn = Callable[[Tables], pd.Series]

FEATURES: dict[str, FeatureFn] = {
    "order_value": orders.order_value,
    "item_count": orders.item_count,
    "unique_product_count": orders.unique_product_count,
    "max_sizes_per_product": orders.max_sizes_per_product,
    "avg_item_price": orders.avg_item_price,
    "customer_prior_order_count": customers.customer_prior_order_count,
}
```

**Format note (load-bearing for Task 15):** the file must end with the dict's closing `}` on its own line and contain no other `}` at line start — `author_senior_branch.py` and `author_alex_pr.py` splice new entries in with a string replace on `"}\n"`.

- [ ] **Step 4: Write `feature_catalog/frame.py`**

```python
"""Assemble a per-order modeling frame: label join + configured features.

Feature authors never write plumbing: build_training_frame() handles the label
and hands each feature function the full Tables.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from feature_catalog.registry import FEATURES, FeatureFn
from feature_catalog.types import Tables


def build_training_frame(
    tables: Tables,
    feature_names: list[str],
    label_horizon_days: int = 60,
    features: Mapping[str, FeatureFn] = FEATURES,
) -> pd.DataFrame:
    """One row per order: order_id, customer_id, checkout_ts, label, features.

    Label: 1 if the order was returned within label_horizon_days of checkout.
    """
    base = tables.orders[["order_id", "customer_id", "checkout_ts"]].set_index("order_id")
    joined = tables.returns.merge(
        tables.orders[["order_id", "checkout_ts"]], on="order_id"
    )
    within = joined[
        joined["return_ts"] <= joined["checkout_ts"] + pd.Timedelta(days=label_horizon_days)
    ]
    base["label"] = base.index.isin(within["order_id"]).astype(int)
    for name in feature_names:
        base[name] = features[name](tables)
    return base.reset_index()
```

(The `features` parameter exists so interviewer tooling can inject planted features without touching the candidate-visible registry — see Task 11.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest feature_catalog/ -q`
Expected: all passing (types, orders, customers, frame)

- [ ] **Step 5b: Verify registry splice invariant**

```bash
python -c "
import pathlib
t = pathlib.Path('feature_catalog/registry.py').read_text()
assert t.endswith('}\n'), 'registry.py must end with closing } on its own line'
assert sum(1 for l in t.splitlines() if l.startswith('}')) == 1, 'registry.py must have exactly one line starting with }'
print('registry splice invariant OK')
"
```

Expected: prints `registry splice invariant OK`.

- [ ] **Step 6: Commit**

```bash
git add feature_catalog/registry.py feature_catalog/frame.py feature_catalog/frame_test.py
git commit -m "feat: feature registry and config-driven training frame builder"
```

---

## Task 7: Feature config + train.py

**Files:**
- Create: `models/return-risk/feature-configs/v1.yaml`
- Create: `models/return-risk/src/train.py`
- Test: `models/return-risk/src/train_test.py`

`models/return-risk/src/` has **no `__init__.py`** — scripts run by path, tests import siblings via pytest's rootdir-insertion (`import train` works because pytest prepends the test file's directory).

- [ ] **Step 1: Write `models/return-risk/feature-configs/v1.yaml`**

```yaml
model: return-risk
label_horizon_days: 60
threshold: 0.50
features:
  - order_value
  - item_count
  - unique_product_count
  - max_sizes_per_product
  - avg_item_price
  - customer_prior_order_count
```

- [ ] **Step 2: Write the failing tests**

`models/return-risk/src/train_test.py`:

```python
import numpy as np
import pandas as pd

import train


def _frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.uniform(0, 365, n), unit="D")
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "order_id": [f"O{i}" for i in range(n)],
            "customer_id": "C0",
            "checkout_ts": ts,
            "x": x,
            "label": (x + rng.normal(scale=0.5, size=n) > 0).astype(int),
        }
    )


def test_split_is_temporal() -> None:
    """The eval protocol (MODEL_CARD.md) is a temporal split. Guard it."""
    train_part, test_part = train.temporal_split(_frame(), holdout_months=3)
    assert train_part["checkout_ts"].max() < test_part["checkout_ts"].min()
    assert len(train_part) > 0 and len(test_part) > 0


def test_fit_returns_scoring_model() -> None:
    frame = _frame()
    model = train.fit(frame, ["x"])
    proba = model.predict_proba(frame[["x"]])[:, 1]
    assert proba.shape == (len(frame),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_gain_share_sums_to_one() -> None:
    frame = _frame()
    model = train.fit(frame, ["x"])
    shares = train.gain_share(model, ["x"])
    assert abs(sum(shares.values()) - 1.0) < 1e-6
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest models/ -q`
Expected: FAIL (`ModuleNotFoundError: train`... it doesn't exist yet)

- [ ] **Step 4: Write `models/return-risk/src/train.py`**

```python
"""Train and evaluate the return-risk model.

Protocol (documented in MODEL_CARD.md): temporal split -- train on everything
before the holdout cutoff, evaluate ROC AUC on the final 3 months. `--eval-only`
re-scores the saved model on the holdout without retraining.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from feature_catalog.frame import build_training_frame
from feature_catalog.types import FeatureConfig, Tables

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS = MODEL_DIR / "artifacts"
HOLDOUT_MONTHS = 3


def load_config() -> FeatureConfig:
    return FeatureConfig.load(MODEL_DIR / "feature-configs" / "v1.yaml")


def temporal_split(
    frame: pd.DataFrame, holdout_months: int = HOLDOUT_MONTHS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on history, evaluate on the most recent window -- never the reverse."""
    cutoff = frame["checkout_ts"].max() - pd.DateOffset(months=holdout_months)
    return frame[frame["checkout_ts"] <= cutoff], frame[frame["checkout_ts"] > cutoff]


def fit(train_frame: pd.DataFrame, features: list[str]) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="auc",
        random_state=0,
        n_jobs=4,
    )
    model.fit(train_frame[features], train_frame["label"])
    return model


def gain_share(model: XGBClassifier, features: list[str]) -> dict[str, float]:
    """Per-feature share of total gain importance (sums to 1)."""
    raw = model.get_booster().get_score(importance_type="gain")
    total = sum(raw.values()) or 1.0
    return {name: raw.get(name, 0.0) / total for name in features}


def main(eval_only: bool = False) -> None:
    config = load_config()
    tables = Tables.load(REPO_ROOT / "data")
    frame = build_training_frame(tables, config.features, config.label_horizon_days)
    train_frame, holdout = temporal_split(frame)

    if eval_only:
        model = XGBClassifier()
        model.load_model(ARTIFACTS / "model.json")
    else:
        model = fit(train_frame, config.features)

    auc = roc_auc_score(
        holdout["label"], model.predict_proba(holdout[config.features])[:, 1]
    )
    importances = gain_share(model, config.features)
    print(f"holdout AUC (temporal, last {HOLDOUT_MONTHS} months): {auc:.4f}")
    print("feature importance (gain share):")
    for name, share in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<32s} {share:.3f}")

    if not eval_only:
        ARTIFACTS.mkdir(exist_ok=True)
        model.save_model(ARTIFACTS / "model.json")
        (ARTIFACTS / "metrics.json").write_text(
            json.dumps({"auc": auc, "gain_share": importances}, indent=2)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    main(eval_only=parser.parse_args().eval_only)
```

- [ ] **Step 5: Run tests, then run training end-to-end**

Run: `uv run pytest models/ -q`
Expected: 3 passed

Run: `make train`
Expected: prints a holdout AUC (any value — calibration comes in Task 13) and a sorted importance table; `models/return-risk/artifacts/model.json` exists.

Run: `make eval`
Expected: same AUC, no retrain.

- [ ] **Step 6: Commit**

```bash
git add models/
git commit -m "feat: return-risk training with temporal-split eval and gain importances"
```

---

## Task 8: predict.py — thin inference surface

**Files:**
- Create: `models/return-risk/src/predict.py`
- Test: `models/return-risk/src/predict_test.py`

- [ ] **Step 1: Write the failing test**

`models/return-risk/src/predict_test.py`:

```python
import numpy as np
import pandas as pd

import predict
import train


def _frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "label": (x > 0).astype(int)})


def test_predict_proba_and_threshold_flagging(tmp_path) -> None:
    frame = _frame()
    model = train.fit(frame, ["x"])
    path = tmp_path / "model.json"
    model.save_model(path)

    loaded = predict.load_model(path)
    proba = predict.predict_proba(loaded, frame, ["x"])
    assert ((proba >= 0) & (proba <= 1)).all()

    flags = predict.flag_for_intervention(proba, threshold=0.5)
    assert flags.dtype == bool
    assert flags.equals(proba >= 0.5)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest models/return-risk/src/predict_test.py -q`
Expected: FAIL (no module `predict`)

- [ ] **Step 3: Write `models/return-risk/src/predict.py`**

```python
"""Thin inference surface: load the trained model, score per-order feature rows.

In production this runs synchronously at checkout; whatever features the config
names must be computable at that moment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = MODEL_DIR / "artifacts" / "model.json"


def load_model(path: Path = DEFAULT_MODEL_PATH) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(path)
    return model


def predict_proba(
    model: XGBClassifier, frame: pd.DataFrame, features: list[str]
) -> pd.Series:
    """Return-risk probability per order row."""
    return pd.Series(
        model.predict_proba(frame[features])[:, 1], index=frame.index, name="return_risk"
    )


def flag_for_intervention(proba: pd.Series, threshold: float) -> pd.Series:
    """Orders CX intercepts. The threshold is a business decision (config-owned)."""
    return proba >= threshold
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest models/ -q`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add models/return-risk/src/predict.py models/return-risk/src/predict_test.py
git commit -m "feat: thin prediction surface with config-owned intervention threshold"
```

---

## Task 9: Candidate-facing docs + lint pass

**Files:**
- Create: `docs/CONTRIBUTING.md`
- Modify: `README.md` (only if quickstart drifted)
- Note: `MODEL_CARD.md` is **not** written here — it is rendered with real numbers in Task 14.

- [ ] **Step 1: Write `docs/CONTRIBUTING.md`**

The ambient-signal doc: eval standards + review norms the manager debrief's "enforce what's written" answer anchors to. Keep it short and norm-shaped — hints, not a checklist of the traps.

```markdown
# Contributing

## Feature standards

- **Prediction-time rule:** the model scores synchronously at checkout. A
  feature may only use information that exists at the order's `checkout_ts`.
  If it touches customer history, apply an explicit as-of cutoff (see
  `customer_prior_order_count` for the pattern).
- Every feature module ships a colocated `*_test.py`. History-derived
  features must include a test that future events are excluded.
- Features are plain functions `(Tables) -> pd.Series` indexed by `order_id`,
  registered in `feature_catalog/registry.py`, selected per-model in
  `models/<model>/feature-configs/`.

## Eval standards

- The eval protocol lives in the model card (`MODEL_CARD.md`) and is the
  contract for any reported metric. Don't change the protocol and the model
  in the same PR; if the protocol must change, update the model card and say
  why.
- Report metrics from the documented protocol only. A number produced under
  a different split is not comparable and shouldn't be quoted.

## Review norms

- PRs that change model behavior state: expected metric impact, how it was
  measured, and the rollout/monitoring plan.
- Decision thresholds in feature configs are business-owned; changing one
  requires explicit signoff from the CX stakeholder.
```

- [ ] **Step 2: Lint and fix**

Run: `make lint`
Expected: clean. Fix anything ruff flags (import order is the usual culprit).

- [ ] **Step 3: Full candidate suite green**

Run: `make test`
Expected: all tests pass (types, generator, features, frame, train, predict).

- [ ] **Step 4: Commit**

```bash
git add docs/CONTRIBUTING.md README.md
git commit -m "docs: contributing norms (prediction-time rule, eval standards, review norms)"
```

---

## Task 10: Planted features — single source of truth

**Files:**
- Create: `interviewer/__init__.py`, `interviewer/planted/__init__.py`, `interviewer/tools/__init__.py` (all empty)
- Create: `interviewer/planted/planted_features.py`
- Test: `interviewer/planted/planted_features_test.py`

Everything the senior branch and Alex's PR add lives here, once. The branch-authoring scripts (Tasks 15, 17) copy these exact sources into the fabricated commits via `inspect.getsource()`, so the code candidates see and the code ground truth is computed from cannot drift.

These tests are interviewer-only (`testpaths` excludes `interviewer/`), so they MAY assert trap semantics. The `customer_return_rate` cutoff tests double as Alex's colocated test — Task 17 copies them into his branch.

- [ ] **Step 1: Write the failing tests**

`interviewer/planted/planted_features_test.py`:

```python
import pandas as pd

from feature_catalog.testing import make_tables
from interviewer.planted.planted_features import (
    customer_return_rate,
    support_contact_count,
    support_contact_count_30d,
    support_contact_count_pre_checkout,
)


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C1"],
            "checkout_ts": pd.to_datetime(["2026-01-01", "2026-03-01"]),
            "order_value": [10.0, 20.0],
            "item_count": [1, 1],
        }
    )


def test_support_contact_count_30d_counts_post_checkout_contacts() -> None:
    """THE SENIOR TRAP: a contact 5 days after checkout is counted.

    At scoring time (checkout) this contact does not exist yet -- the feature
    is leaky by construction and this test documents it.
    """
    contacts = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "order_id": ["O1"],
            "contact_ts": pd.to_datetime(["2026-01-06"]),
            "channel": ["email"],
        }
    )
    result = support_contact_count_30d(
        make_tables(orders=_orders(), support_contacts=contacts)
    )
    assert result.loc["O1"] == 1.0


def test_support_contact_count_is_a_plain_join_count() -> None:
    """ALEX'S TRAP: counts every contact on the order, timestamps never consulted."""
    contacts = pd.DataFrame(
        {
            "customer_id": ["C1", "C1"],
            "order_id": ["O1", "O1"],
            "contact_ts": pd.to_datetime(["2025-12-30", "2026-02-15"]),
            "channel": ["email", "chat"],
        }
    )
    result = support_contact_count(
        make_tables(orders=_orders(), support_contacts=contacts)
    )
    assert result.loc["O1"] == 2.0
    assert result.loc["O2"] == 0.0


def test_pre_checkout_variant_excludes_later_contacts() -> None:
    """The best-tier senior fix: only contacts that already exist at checkout."""
    contacts = pd.DataFrame(
        {
            "customer_id": ["C1", "C1"],
            "order_id": ["O1", "O1"],
            "contact_ts": pd.to_datetime(["2025-12-30", "2026-01-06"]),
            "channel": ["email", "chat"],
        }
    )
    result = support_contact_count_pre_checkout(
        make_tables(orders=_orders(), support_contacts=contacts)
    )
    assert result.loc["O1"] == 1.0  # only the 2025-12-30 contact
    assert result.loc["O2"] == 2.0  # both contacts predate O2's checkout


# -- the two tests below are copied verbatim into Alex's branch by Task 17 --


def test_customer_return_rate_excludes_post_checkout_returns() -> None:
    """A prior order's return that lands AFTER this checkout must not count."""
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C1"],
            "checkout_ts": pd.to_datetime(["2026-01-01", "2026-01-10"]),
            "order_value": [10.0, 20.0],
            "item_count": [1, 1],
        }
    )
    returns = pd.DataFrame(
        {
            # O1's return is realized on Jan 20 -- AFTER O2's Jan 10 checkout
            "order_id": ["O1"],
            "return_ts": pd.to_datetime(["2026-01-20"]),
            "reason": ["wrong_size"],
        }
    )
    result = customer_return_rate(make_tables(orders=orders, returns=returns))
    assert result.loc["O2"] == 0.0


def test_customer_return_rate_counts_realized_prior_returns() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C1"],
            "checkout_ts": pd.to_datetime(["2026-01-01", "2026-03-01"]),
            "order_value": [10.0, 20.0],
            "item_count": [1, 1],
        }
    )
    returns = pd.DataFrame(
        {
            "order_id": ["O1"],
            "return_ts": pd.to_datetime(["2026-01-15"]),  # before O2's checkout
            "reason": ["wrong_size"],
        }
    )
    result = customer_return_rate(make_tables(orders=orders, returns=returns))
    assert result.loc["O2"] == 1.0  # 1 prior order, 1 realized return
    assert result.loc["O1"] == 0.0  # no prior orders -> cold-start default
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest interviewer/ -q`
Expected: FAIL (import error)

- [ ] **Step 3: Write `interviewer/planted/planted_features.py`**

Constraints baked into the code text (the spec is explicit):
- `support_contact_count` (Alex's): **zero timestamp logic** — a plain join-count. The `clip(upper=8)` magic number and the `__main__` hardcoded path are two of Alex's seeded code nits.
- `customer_return_rate`: correct, **both cutoffs plainly named** so a reviewer can verify by reading. It deliberately copy-pastes the prior-orders merge pattern instead of extracting a helper — Alex's seeded duplication nit.
- v1.3 benign chaff features are genuinely benign (point-in-time correct) so the senior diff localizes without muddying the story.

```python
"""Planted feature implementations -- the single source for both fabricated
branches. Task 15/17 splice these sources into the senior and Alex commits
via inspect.getsource(); ground truth (compute_ground_truth.py) imports them
directly. Never reimplement these anywhere else.
"""

from __future__ import annotations

import pandas as pd

from feature_catalog.types import Tables

# --------------------------------------------------------------------------
# v1.3 "customer signal" chaff (senior branch) -- benign, point-in-time safe
# --------------------------------------------------------------------------


def customer_avg_order_value(tables: Tables) -> pd.Series:
    """Mean value of the customer's previous orders (0 for first orders)."""
    orders = tables.orders[["order_id", "customer_id", "checkout_ts", "order_value"]]
    pairs = orders.merge(orders, on="customer_id", suffixes=("", "_other"))
    prior = pairs[pairs["checkout_ts_other"] < pairs["checkout_ts"]]
    means = prior.groupby("order_id")["order_value_other"].mean()
    return means.reindex(tables.orders["order_id"]).fillna(0.0).astype(float)


def customer_days_since_last_order(tables: Tables) -> pd.Series:
    """Days since the customer's previous order (-1 for first orders)."""
    orders = tables.orders.sort_values(["customer_id", "checkout_ts"])
    gaps = orders.groupby("customer_id")["checkout_ts"].diff().dt.total_seconds() / 86400
    result = pd.Series(gaps.values, index=orders["order_id"])
    return result.reindex(tables.orders["order_id"]).fillna(-1.0).astype(float)


def weekend_order(tables: Tables) -> pd.Series:
    """1 if checkout happened on a weekend."""
    orders = tables.orders.set_index("order_id")
    return (orders["checkout_ts"].dt.dayofweek >= 5).astype(float)


def checkout_hour(tables: Tables) -> pd.Series:
    """Hour of day at checkout."""
    return tables.orders.set_index("order_id")["checkout_ts"].dt.hour.astype(float)


# --------------------------------------------------------------------------
# THE SENIOR TRAP -- v1.3's support_contact_count_30d
# --------------------------------------------------------------------------


def support_contact_count_30d(tables: Tables) -> pd.Series:
    """Customer support touchpoints within 30 days of the order -- friction signal."""
    # INVARIANT (build-side, do not "fix"): joins on order_id, NOT customer_id.
    # A just-placed order has no contacts yet, so this feature is ~always 0 at
    # serving time -- that is the entire prod-collapse mechanism. The serve-safe
    # re-derivation below joins on customer_id ON PURPOSE; harmonizing the two
    # would silently break the trap while every calibration gate stays green
    # (prod_sim forces the column to 0 by fiat). check_calibration asserts the
    # serving-time mean is ~0 as a backstop.
    orders = tables.orders[["order_id", "checkout_ts"]]
    contacts = tables.support_contacts.merge(orders, on="order_id")
    near = contacts[
        (contacts["contact_ts"] - contacts["checkout_ts"]).abs()
        <= pd.Timedelta(days=30)
    ]
    counts = near.groupby("order_id").size()
    return counts.reindex(tables.orders["order_id"], fill_value=0).astype(float)


# --------------------------------------------------------------------------
# THE SENIOR BEST-TIER FIX -- point-in-time re-derivation (answer key only)
# --------------------------------------------------------------------------


def support_contact_count_pre_checkout(tables: Tables) -> pd.Series:
    """The customer's support contacts that already exist at checkout.

    Pre-purchase friction is real signal; this is the serve-safe re-derivation
    of support_contact_count_30d.
    """
    orders = tables.orders[["order_id", "customer_id", "checkout_ts"]]
    contacts = tables.support_contacts[["customer_id", "contact_ts"]].merge(
        orders, on="customer_id"
    )
    pre = contacts[contacts["contact_ts"] < contacts["checkout_ts"]]
    counts = pre.groupby("order_id").size()
    return counts.reindex(tables.orders["order_id"], fill_value=0).astype(float)


# --------------------------------------------------------------------------
# ALEX'S TRAP -- plain join count, no timestamp logic anywhere
# --------------------------------------------------------------------------


def support_contact_count(tables: Tables) -> pd.Series:
    """Support touchpoints on the order -- friction signal."""
    contacts = tables.support_contacts.merge(tables.orders[["order_id"]], on="order_id")
    counts = contacts.groupby("order_id").size().clip(upper=8)
    return counts.reindex(tables.orders["order_id"], fill_value=0).astype(float)


# --------------------------------------------------------------------------
# ALEX'S RED HERRING -- correct point-in-time return rate, both cutoffs named
# --------------------------------------------------------------------------


def customer_return_rate(tables: Tables) -> pd.Series:
    """Share of the customer's prior orders already returned by this checkout.

    Two point-in-time cutoffs:
      1. prior orders only: the other order was placed before this checkout
      2. realized returns only: the return itself happened before this checkout
    """
    orders = tables.orders[["order_id", "customer_id", "checkout_ts"]]
    pairs = orders.merge(orders, on="customer_id", suffixes=("", "_prior"))
    placed_before_checkout = pairs[pairs["checkout_ts_prior"] < pairs["checkout_ts"]]
    rets = tables.returns[["order_id", "return_ts"]].rename(
        columns={"order_id": "order_id_prior"}
    )
    with_returns = placed_before_checkout.merge(rets, on="order_id_prior", how="left")
    returned_by_checkout = with_returns["return_ts"] < with_returns["checkout_ts"]
    grouped = with_returns.assign(returned=returned_by_checkout).groupby("order_id")
    rate = grouped["returned"].sum() / grouped.size()
    return rate.reindex(tables.orders["order_id"]).fillna(0.0).astype(float)
```

(Alex's `__main__` hardcoded-path nit is added at branch-authoring time in Task 17, not here — it must not run during ground-truth imports.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest interviewer/ -q`
Expected: 5 passed

Also confirm the candidate suite still excludes these: `uv run pytest -q` → same count as Task 9 (no interviewer tests collected).

- [ ] **Step 5: Commit**

```bash
git add interviewer/
git commit -m "feat(interviewer): planted feature implementations with trap-documenting tests"
```

---

## Task 11: Ground-truth computation

**Files:**
- Create: `interviewer/tools/compute_ground_truth.py`
- Output: `interviewer/answer_keys/ground_truth.json` (committed in Task 13 once calibrated)

**The two distinct operations — do not conflate them (spec grading note is emphatic):**
1. **Retrain-without-feature**: drop the leaky column, retrain, evaluate → the honest numbers (~0.84 senior, ~0.85 manager ablation). Under ANY split, a retrained model *with* the leak still looks great offline — the leak is invisible to retrain-based evaluation.
2. **Force-to-zero on the TRAINED model**: train *with* the leak, then zero the column in the holdout and re-score the same model → the prod collapse (~0.62). This is the offline reproduction of production, where the feature is always zero at checkout.

- [ ] **Step 1: Write `interviewer/tools/compute_ground_truth.py`**

```python
"""Train every narrative variant and emit ground_truth.json.

Every number quoted anywhere (tickets, answer keys, model card, commit
messages, PR body) comes from this file's output. Never hand-type a metric.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from feature_catalog.frame import build_training_frame
from feature_catalog.registry import FEATURES
from feature_catalog.types import FeatureConfig, Tables
from interviewer.planted import planted_features as pf

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "interviewer" / "answer_keys" / "ground_truth.json"

PLANTED = {
    name: getattr(pf, name)
    for name in (
        "customer_avg_order_value",
        "customer_days_since_last_order",
        "weekend_order",
        "checkout_hour",
        "support_contact_count_30d",
        "support_contact_count_pre_checkout",
        "support_contact_count",
        "customer_return_rate",
    )
}

V13_CHAFF = [
    "customer_avg_order_value",
    "customer_days_since_last_order",
    "weekend_order",
    "checkout_hour",
]
SENIOR_LEAK = "support_contact_count_30d"
ALEX_LEAK = "support_contact_count"
RED_HERRING = "customer_return_rate"
HOLDOUT_MONTHS = 3


def _model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        eval_metric="auc", random_state=0, n_jobs=4,
    )


def _temporal(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = frame["checkout_ts"].max() - pd.DateOffset(months=HOLDOUT_MONTHS)
    return frame[frame["checkout_ts"] <= cutoff], frame[frame["checkout_ts"] > cutoff]


def temporal_auc(frame: pd.DataFrame, features: list[str]) -> float:
    train, test = _temporal(frame)
    model = _model().fit(train[features], train["label"])
    return float(roc_auc_score(test["label"], model.predict_proba(test[features])[:, 1]))


def random_auc(frame: pd.DataFrame, features: list[str]) -> float:
    """The contractor's 'simplified' eval: a shuffled 80/20 split."""
    train, test = train_test_split(
        frame, test_size=0.2, random_state=42, stratify=frame["label"]
    )
    model = _model().fit(train[features], train["label"])
    return float(roc_auc_score(test["label"], model.predict_proba(test[features])[:, 1]))


def kfold_auc(frame: pd.DataFrame, features: list[str], n_splits: int = 5) -> float:
    """Alex's eval: stratified K-fold mean AUC."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, te in cv.split(frame[features], frame["label"]):
        model = _model().fit(frame.iloc[tr][features], frame.iloc[tr]["label"])
        aucs.append(
            roc_auc_score(
                frame.iloc[te]["label"],
                model.predict_proba(frame.iloc[te][features])[:, 1],
            )
        )
    return float(np.mean(aucs))


def prod_sim_auc(frame: pd.DataFrame, features: list[str], dead_feature: str) -> float:
    """OPERATION 2: the trained model scored with the feature forced to 0.

    This is what production does every day -- at checkout the contact count is
    always zero -- and it is NOT the same as retraining without the feature.
    """
    train, test = _temporal(frame)
    model = _model().fit(train[features], train["label"])
    crippled = test.copy()
    crippled[dead_feature] = 0.0
    return float(
        roc_auc_score(test["label"], model.predict_proba(crippled[features])[:, 1])
    )


def leak_gain_share(frame: pd.DataFrame, features: list[str], leak: str) -> float:
    train, _ = _temporal(frame)
    model = _model().fit(train[features], train["label"])
    raw = model.get_booster().get_score(importance_type="gain")
    total = sum(raw.values()) or 1.0
    return float(raw.get(leak, 0.0) / total)


def texture_stats(tables: Tables) -> dict[str, float]:
    contacts = tables.support_contacts.merge(
        tables.orders[["order_id", "checkout_ts"]], on="order_id"
    )
    post_share = float((contacts["contact_ts"] > contacts["checkout_ts"]).mean())
    orders = tables.orders.copy()
    orders["returned"] = orders["order_id"].isin(tables.returns["order_id"])
    monthly = orders.set_index("checkout_ts").resample("MS")["returned"].mean()
    # backstop for the leak's join-key invariant: the order's own contacts
    # that exist AT checkout must be ~none, or the prod-collapse story is broken
    at_checkout = contacts[contacts["contact_ts"] <= contacts["checkout_ts"]]
    serving_mean = float(
        at_checkout.groupby("order_id").size().reindex(
            tables.orders["order_id"], fill_value=0
        ).mean()
    )
    return {
        "post_checkout_contact_share": post_share,
        "leak_serving_time_mean": serving_mean,
        "monthly_return_rate_first": float(monthly.iloc[0]),
        "monthly_return_rate_last": float(monthly.iloc[-2]),  # last full month
    }


def main() -> None:
    config = FeatureConfig.load(
        REPO / "models" / "return-risk" / "feature-configs" / "v1.yaml"
    )
    main_features = config.features
    tables = Tables.load(REPO / "data")

    all_features = {**FEATURES, **PLANTED}
    all_names = main_features + list(PLANTED)
    frame = build_training_frame(
        tables, all_names, config.label_horizon_days, features=all_features
    )

    v13 = main_features + V13_CHAFF + [SENIOR_LEAK]
    alex = main_features + [ALEX_LEAK, RED_HERRING]

    gt: dict[str, float | int] = {"seed": 412}
    # senior narrative
    gt["main_honest_auc"] = temporal_auc(frame, main_features)
    gt["v13_random_auc"] = random_auc(frame, v13)                      # "0.92 offline"
    gt["v13_temporal_auc"] = temporal_auc(frame, v13)                  # layer-1 partial fix
    gt["prod_sim_auc"] = prod_sim_auc(frame, v13, SENIOR_LEAK)         # "0.62 in prod"
    gt["leak_gain_share"] = leak_gain_share(frame, v13, SENIOR_LEAK)   # importance ramp
    gt["senior_honest_auc"] = temporal_auc(
        frame, main_features + V13_CHAFF
    )  # good-tier fix: remove leak, keep chaff, temporal split
    gt["rederived_auc"] = temporal_auc(
        frame, main_features + V13_CHAFF + ["support_contact_count_pre_checkout"]
    )  # best-tier fix
    gt["rederived_lift"] = gt["rederived_auc"] - gt["senior_honest_auc"]
    # manager narrative
    gt["alex_kfold_auc"] = kfold_auc(frame, alex)                      # "0.96 🎉"
    gt["alex_prod_sim_auc"] = prod_sim_auc(frame, alex, ALEX_LEAK)
    gt["manager_ablation_auc"] = temporal_auc(frame, main_features + [RED_HERRING])
    # part 2 narrative (reference implementation = customer_return_rate)
    gt["part2_auc"] = temporal_auc(frame, main_features + [RED_HERRING])
    gt["part2_lift"] = gt["part2_auc"] - gt["main_honest_auc"]
    gt.update(texture_stats(tables))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                               for k, v in gt.items()}, indent=2) + "\n")
    for key, value in gt.items():
        print(f"{key:>32s}: {value:.4f}" if isinstance(value, float) else f"{key:>32s}: {value}")


if __name__ == "__main__":
    main()
```

(Note `manager_ablation_auc` and `part2_auc` are the same computation — the spec's ~0.85 ablation and ~0.84→0.87 Part 2 lift share the `customer_return_rate` vehicle. One number, two narrative uses. Keep both keys: templates read them by role.)

- [ ] **Step 2: Add the Makefile target**

Append to `Makefile`:

```make
ground-truth:
	uv run python -m interviewer.tools.compute_ground_truth

check-calibration:
	uv run python -m interviewer.tools.check_calibration
```

(Add `ground-truth check-calibration` to the `.PHONY` line.)

- [ ] **Step 3: Run it**

Run: `make ground-truth`
Expected: prints all keys with plausible-looking AUCs (runtime ~1–3 min; K-fold is 5 trains) and writes `interviewer/answer_keys/ground_truth.json`. The numbers will likely be OFF TARGET — that's expected before Task 13.

- [ ] **Step 3b: Structural sanity check before committing**

```bash
python -c "
import json
from pathlib import Path
gt = json.loads(Path('interviewer/answer_keys/ground_truth.json').read_text())
auc_keys = [k for k in gt if k.endswith('_auc')]
bad = [k for k in auc_keys if not (0.5 < gt[k] < 1.0)]
assert not bad, f'AUC-like values out of (0.5, 1.0): {bad}'
# check all keys the calibration gate will read are present
expected = [
    'main_honest_auc', 'v13_random_auc', 'v13_temporal_auc', 'prod_sim_auc',
    'leak_gain_share', 'senior_honest_auc', 'rederived_lift', 'alex_kfold_auc',
    'alex_prod_sim_auc', 'manager_ablation_auc', 'part2_lift',
    'post_checkout_contact_share', 'leak_serving_time_mean',
    'monthly_return_rate_first', 'monthly_return_rate_last',
]
missing = [k for k in expected if k not in gt]
assert not missing, f'missing keys: {missing}'
print('ground_truth.json structure OK')
"
```

Expected: prints `ground_truth.json structure OK`. If any AUC key falls outside (0.5, 1.0) or any expected key is absent, the generator has a silent bug — fix before proceeding.

- [ ] **Step 4: Commit (script only — json gets committed when calibrated)**

```bash
git add interviewer/tools/ Makefile
git commit -m "feat(interviewer): ground-truth computation for all narrative variants"
```

---

## Task 12: Calibration gate

**Files:**
- Create: `interviewer/tools/check_calibration.py`

This is the **exact acceptance test** for the generator (spec Build-TODO assertion (b): every narrative number must hold *simultaneously* at the pinned seed — they're asserted jointly here because all values come from one `ground_truth.json` produced by one generation run).

- [ ] **Step 1: Write `interviewer/tools/check_calibration.py`**

```python
"""Assert every narrative number co-occurs at the pinned seed.

Reads interviewer/answer_keys/ground_truth.json (run `make ground-truth`
first). Exits nonzero with a per-check table if any band or ordering fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GT_PATH = (
    Path(__file__).resolve().parents[2] / "interviewer" / "answer_keys" / "ground_truth.json"
)

# (low, high) inclusive bands. Spec narrative targets in comments.
BANDS: dict[str, tuple[float, float]] = {
    "main_honest_auc": (0.82, 0.86),          # "honest baseline ~0.84"
    "v13_random_auc": (0.90, 0.94),           # "offline AUC was 0.92"
    "v13_temporal_auc": (0.855, 0.895),       # "restores temporal -> ~0.87"
    "prod_sim_auc": (0.58, 0.66),             # "prod has it at ~0.62"
    "leak_gain_share": (0.45, 0.75),          # "~60% of total gain"
    "senior_honest_auc": (0.82, 0.845),       # "correct fix lands ~0.84"; ceiling held below v13_temporal's floor minus the ordering gap so bands and orderings are jointly satisfiable
    "rederived_lift": (0.002, 0.03),          # assertion (a): modest GENUINE lift
    "alex_kfold_auc": (0.935, 0.985),         # "AUC 0.96 🎉"
    "alex_prod_sim_auc": (0.55, 0.70),        # Alex's leak collapses in prod too
    "manager_ablation_auc": (0.83, 0.875),    # "ablation shows ~0.85"
    "part2_lift": (0.015, 0.05),              # "~0.84 -> ~0.87"
    "post_checkout_contact_share": (0.85, 0.95),  # "~90% post-date checkout"
    "leak_serving_time_mean": (0.0, 0.06),    # the leak feature is dead at checkout
}

# (smaller_key, larger_key, min_gap)
ORDERINGS: list[tuple[str, str, float]] = [
    ("v13_temporal_auc", "v13_random_auc", 0.02),   # split choice genuinely moves it
    ("senior_honest_auc", "v13_temporal_auc", 0.01),  # both layers matter
    ("prod_sim_auc", "senior_honest_auc", 0.10),    # the collapse is dramatic
    ("main_honest_auc", "part2_auc", 0.015),        # part 2 earns a real lift
    ("monthly_return_rate_first", "monthly_return_rate_last", 0.03),  # drift exists
]


def main() -> None:
    gt = json.loads(GT_PATH.read_text())
    failures: list[str] = []

    print(f"{'check':<44s} {'value':>8s}  {'target':<16s} result")
    for key, (low, high) in BANDS.items():
        if key not in gt:
            print(f"{key:<44s} {'':>8s}  [{low}, {high}]  FAIL (missing from ground_truth.json)")
            failures.append(f"FAIL {key}: missing from ground_truth.json")
            continue
        value = gt[key]
        ok = low <= value <= high
        print(f"{key:<44s} {value:>8.4f}  [{low}, {high}]  {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{key}={value:.4f} outside [{low}, {high}]")

    for small, large, gap in ORDERINGS:
        if small not in gt:
            print(f"{'FAIL ' + small:<44s} {'':>8s}  {'':16s}FAIL (missing from ground_truth.json)")
            failures.append(f"FAIL {small}: missing from ground_truth.json")
            continue
        if large not in gt:
            print(f"{'FAIL ' + large:<44s} {'':>8s}  {'':16s}FAIL (missing from ground_truth.json)")
            failures.append(f"FAIL {large}: missing from ground_truth.json")
            continue
        ok = gt[large] - gt[small] >= gap
        label = f"{large} - {small} >= {gap}"
        print(f"{label:<44s} {gt[large] - gt[small]:>8.4f}  {'':<16s}{'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(label)

    if failures:
        print(f"\n{len(failures)} calibration failure(s):")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("\nall narrative numbers co-occur at seed", gt["seed"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `make check-calibration`
Expected: a full table; almost certainly some FAILs at the initial knob values. That's the input to Task 13.

- [ ] **Step 3: Commit**

```bash
git add interviewer/tools/check_calibration.py
git commit -m "feat(interviewer): joint calibration gate for all narrative numbers"
```

---

## Task 13: The calibration loop (empirical — expect iteration)

**Files:**
- Modify: `data/generate.py` (knob values only)
- Commit when green: `data/*.csv`, `interviewer/answer_keys/ground_truth.json`

This task is an explicit tuning loop, not a write-once step. The loop:

```bash
# 1. edit knob constants in data/generate.py
# 2. regenerate + recompute + check:
make data && make ground-truth && make check-calibration
# 3. repeat until check-calibration exits 0
```

- [ ] **Step 1: Iterate using this direction table**

| Failing check | Knob moves (in `data/generate.py`) |
|---|---|
| `main_honest_auc` too LOW | ↑ `W_MULTI_SIZE`, `W_LOG_ORDER_VALUE`, `SERIAL_MULTI_SIZE_P`; ↓ `LATENT_SD` |
| `main_honest_auc` too HIGH | reverse of the above |
| `v13_random_auc` too LOW (leak not golden enough) | ↑ `POST_CONTACT_P_RETURNED`; ↓ `POST_CONTACT_P_KEPT` |
| `prod_sim_auc` too HIGH (no collapse) | make the model lean harder on the leak: ↑ `POST_CONTACT_P_RETURNED`, ↓ `POST_CONTACT_P_KEPT`; slightly ↓ honest-feature weights |
| `prod_sim_auc` too LOW | ease the leak slightly or ↑ honest-feature weights (the crippled model falls back on them) |
| `leak_gain_share` out of band | same levers as prod_sim (gain share and collapse depth move together) |
| `v13_random - v13_temporal` gap too SMALL | ↑ drift: widen `BASE_RETURN_LOGIT_START/END` spread; ↑ `SERIAL_LATE_SKEW` |
| `part2_lift` too SMALL | ↑ `LATENT_SD` or `SERIAL_LOGIT_BOOST` (history-recoverable signal the order shape can't see) — recheck `main_honest_auc` after |
| `rederived_lift` too SMALL | ↑ `PRE_CONTACT_LOGIT_LIFT` and/or `PRE_CONTACT_P` (watch `post_checkout_contact_share` ceiling) |
| `post_checkout_contact_share` too LOW | ↓ `PRE_CONTACT_P` or ↑ `POST_CONTACT_P_RETURNED` |
| `monthly` drift too FLAT | widen `BASE_RETURN_LOGIT_*` spread |

Practical notes:
- Knobs interact: `LATENT_SD` trades `main_honest_auc` against `part2_lift`; pre-contact knobs trade `rederived_lift` against `post_checkout_contact_share`. Move 1–2 knobs per iteration and keep a scratch log of (knobs → numbers).
- If ~10 iterations don't converge, the likely structural issues are: (a) honest features too weak relative to noise — strengthen `W_*` weights before touching anything else; (b) the collapse floor too high because the crippled model still sees good features — that's realistic, accept the high end of the 0.58–0.66 band or deepen the leak.
- If a band is genuinely unreachable without breaking another, write `docs/plans/BLOCKED.md` per the execution protocol — include the failing band(s), the knob-value history of attempts, and the closest-achieved values — then stop. Bands can be renegotiated (they're proxies for the spec's "~" numbers), but only deliberately, with a note in the answer key.

- [ ] **Step 2: When green, run the full gate twice to confirm determinism**

```bash
make data && make ground-truth && make check-calibration
git diff --stat data/  # second regeneration must produce zero diff
```

Expected: exit 0 both times; no CSV diff between runs.

- [ ] **Step 3: Run the candidate suite against the final data**

Run: `make test && make train`
Expected: tests green; `make train` prints holdout AUC matching `main_honest_auc` ±0.005 and a sane importance table.

- [ ] **Step 4: Commit the pinned world**

```bash
git add data/ interviewer/answer_keys/ground_truth.json
git commit -m "feat: pin calibrated generator knobs and ground-truth numbers at seed 412"
```

---

## Task 14: Templates + rendered docs (model card, tickets, answer keys, PR body)

**Files:**
- Create: `interviewer/templates/model_card.md`, `risk-412.md`, `part2-ticket.md`, `senior-answer-key.md`, `manager-answer-key.md`, `alex-pr-body.md`
- Create: `interviewer/tools/render_docs.py`
- Output (committed): `models/return-risk/MODEL_CARD.md`, `interviewer/tickets/RISK-412.md`, `interviewer/tickets/part2-serial-returners.md`, `interviewer/answer_keys/senior.md`, `interviewer/answer_keys/manager.md`, `interviewer/rendered/alex-pr-body.md`

Templates use `string.Template` (`$name` placeholders — no brace escaping issues in markdown). The render script pre-formats every number (2 decimal places for AUCs, percentages for shares).

- [ ] **Step 1: Write `interviewer/tools/render_docs.py`**

```python
"""Render every numbered document from ground_truth.json. Never edit outputs by hand."""

from __future__ import annotations

import json
from pathlib import Path
from string import Template

REPO = Path(__file__).resolve().parents[2]
TEMPLATES = REPO / "interviewer" / "templates"
GT = json.loads((REPO / "interviewer" / "answer_keys" / "ground_truth.json").read_text())

OUTPUTS = {
    "model_card.md": REPO / "models" / "return-risk" / "MODEL_CARD.md",
    "risk-412.md": REPO / "interviewer" / "tickets" / "RISK-412.md",
    "part2-ticket.md": REPO / "interviewer" / "tickets" / "part2-serial-returners.md",
    "senior-answer-key.md": REPO / "interviewer" / "answer_keys" / "senior.md",
    "manager-answer-key.md": REPO / "interviewer" / "answer_keys" / "manager.md",
    "alex-pr-body.md": REPO / "interviewer" / "rendered" / "alex-pr-body.md",
}


def _context() -> dict[str, str]:
    ctx = {}
    for key, value in GT.items():
        if isinstance(value, float):
            ctx[key] = f"{value:.2f}"
            ctx[key + "_pct"] = f"{value:.0%}"
        else:
            ctx[key] = str(value)
    return ctx


def main() -> None:
    ctx = _context()
    for template_name, out_path in OUTPUTS.items():
        template = Template((TEMPLATES / template_name).read_text())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(template.substitute(ctx))
        print(f"rendered {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
```

Add to `Makefile` (and `.PHONY`):

```make
render-docs:
	uv run python -m interviewer.tools.render_docs
```

- [ ] **Step 2: Write `interviewer/templates/model_card.md`**

```markdown
# Return-Risk Model Card

## Problem

Binary classifier predicting, **at checkout time**, whether an order will be
returned within 60 days. CX uses the score to intercept risky orders before
fulfillment; scoring happens synchronously in the checkout path.

## Data

Four tables generated into `data/` (see `data/generate.py`): `orders`,
`order_items`, `returns`, `support_contacts`. Label: order returned within
60 days of checkout.

## Features

Selected per-model in `feature-configs/v1.yaml`; implementations live in
`feature_catalog/`. Anything derived from customer history must be
computable at the order's checkout time (see CONTRIBUTING.md).

## Evaluation protocol

- **Temporal split**: train on all orders up to (latest checkout − 3 months);
  evaluate on the final 3 months. Return behavior drifts, so a shuffled
  split overstates performance — the holdout must be the future.
- **Metric**: ROC AUC on the temporal holdout (`make eval`).
- **Current baseline: AUC $main_honest_auc.** Any PR claiming a different
  number must produce it under this protocol.

## Monitoring

A weekly job scores realized labels at 60 days and reports prod AUC.
```

(Only AUC values are templated; the label horizon is literal text.)

- [ ] **Step 3: Write `interviewer/templates/risk-412.md`**

```markdown
# RISK-412 — Return-risk model underperforming in prod

**Priority:** High · **Reporter:** Priya N. (DS Lead) · **Assignee:** you

The return-risk model is underperforming in prod. Offline AUC was
$v13_random_auc when v1.3 shipped; prod monitoring has it at ~$prod_sim_auc
over the last 8 weeks, barely better than the old rules engine. CX planned
interception volumes around the offline number. Can you take a look?

*Deliverable: a PR against `senior/start` with the fix.*
```

- [ ] **Step 4: Write `interviewer/templates/part2-ticket.md`**

```markdown
# RISK-431 — Capture repeat-return behavior

**Priority:** Medium · **Reporter:** Priya N. (DS Lead)

CX believes serial returners are gaming the policy — customers who order
multiple sizes and return most of them. Product wants the model to capture
repeat-return behavior. One or two features is plenty; retrain and report
what you actually got.
```

- [ ] **Step 5: Write `interviewer/templates/senior-answer-key.md`**

Must contain: the two-layer trap, all numbers, the three sanctioned diagnostic paths, the fix tiers, and the spec's grading note **verbatim** (spec §"Grading note", the paragraph beginning "the leak is *invisible to retrain-based evaluation under any split*…" — copy it word-for-word from `docs/specs/technical-interview-design.md` into the template, replacing the inline ~numbers with `$v13_random_auc`, `$v13_temporal_auc`, `$senior_honest_auc`, `$prod_sim_auc` placeholders).

- [ ] **Step 5a: Paste the grading note into the template**

Read `docs/specs/technical-interview-design.md`. Locate the paragraph that begins "the leak is *invisible to retrain-based evaluation under any split*" and ends "...the right action alone is acceptable at best." (it is the Grading note paragraph, spec §Senior task). Paste it verbatim into `interviewer/templates/senior-answer-key.md` replacing the `<copy here …>` placeholder. While pasting, substitute the inline numbers: "0.92" → `$v13_random_auc`, "~0.87" → `$v13_temporal_auc`, "~0.84" → `$senior_honest_auc`, "0.62"/"~0.62" → `$prod_sim_auc`.

```markdown
# Senior answer key — RISK-412 (generated; do not hand-edit)

## Ground truth at seed $seed

| quantity | value |
|---|---|
| v1.3 offline AUC as shipped (random split, leak in) | $v13_random_auc |
| v1.3 under restored temporal split (layer-1-only fix) | $v13_temporal_auc |
| honest AUC after removing the leak (temporal) | $senior_honest_auc |
| prod simulation: trained v1.3 model, leak forced to 0 | $prod_sim_auc |
| leak share of total gain importance | $leak_gain_share |
| best-tier fix: re-derived pre-checkout feature | $rederived_auc (lift $rederived_lift over removal) |
| share of support contacts post-dating checkout | $post_checkout_contact_share_pct |

## The trap, two layers

1. **The eval lies** — v1.3 switched the temporal split to a random shuffle.
   Restoring it: $v13_random_auc → $v13_temporal_auc. Partial credit.
2. **`support_contact_count_30d` cannot exist at prediction time** — most
   support contacts are about the return and post-date checkout. Offline the
   feature is gold; at checkout it is always zero, and the model leans on it
   ($leak_gain_share of gain). Prod scores a dead feature: $prod_sim_auc.

## Sanctioned diagnostic paths (all first-class)

1. Ask Priya: "when do we score?" → "synchronously at checkout."
2. Audit the data: $post_checkout_contact_share_pct of contacts post-date
   checkout (one query over the shipped CSVs).
3. Counterfactual: force the feature to 0 on the trained v1.3 model and
   re-score the holdout → $prod_sim_auc, reproducing prod offline.

## Fix tiers

1. **Best:** names the prediction-time contract; re-derives the feature
   point-in-time (pre-checkout contacts → AUC $rederived_auc); restores the
   temporal split; updates the model card; explains $v13_random_auc → $prod_sim_auc
   mechanistically in the PR.
2. **Good:** removes the feature, fixes the split, retrains, reports
   $senior_honest_auc honestly.
3. **Acceptable:** removes the feature but misses the split layer (headline
   still ~$v13_temporal_auc-ish, unnoticed).
- **Weak:** tunes/regularizes/calibrates and reports a better offline number.

## Grading note (verbatim from the design spec)

<copy here, word-for-word, the single paragraph from
docs/specs/technical-interview-design.md that begins "the leak is
*invisible to retrain-based evaluation under any split*" and ends
"...the right action alone is acceptable at best." Replace its inline
numbers: "0.92" -> $v13_random_auc, "~0.87" -> $v13_temporal_auc,
"~0.84" -> $senior_honest_auc, "0.62"/"~0.62" -> $prod_sim_auc.>

## Part 2 (if offered)

Reference implementation: `customer_return_rate` with BOTH cutoffs (orders
placed before checkout AND returns realized before checkout). Honest lift:
$main_honest_auc → $part2_auc. The silent test: leaking return realization
($part2_lift genuine lift exists, so correct work is rewarded). Discriminator
is RECOGNIZING the return-timing subtlety; shipping a simpler version with an
explicit serve-safety caveat is a strong outcome.
```

(The `[PASTE …]` block is an instruction to the implementing engineer to copy spec text — the one place where copying beats restating, because the spec demands the note appear verbatim.)

- [ ] **Step 6: Write `interviewer/templates/manager-answer-key.md`**

```markdown
# Manager answer key — Alex's PR (generated; do not hand-edit)

## Ground truth at seed $seed

| quantity | value |
|---|---|
| main honest baseline (temporal) | $main_honest_auc |
| Alex's claimed number (5-fold CV, leak in) | $alex_kfold_auc |
| ablation: retrain without the contact feature (temporal) | $manager_ablation_auc |
| prod simulation: Alex's model, contact feature forced to 0 | $alex_prod_sim_auc |
| share of support contacts post-dating checkout | $post_checkout_contact_share_pct |

## Seeded issues by severity

1. **CRITICAL — the leak.** `support_contact_count`
   (`feature_catalog/features/support.py`): plain join-count of
   `support_contacts`, no timestamp logic in the diff. The leak lives in the
   data: $post_checkout_contact_share_pct of contacts post-date checkout. Not
   confirmable from the diff — requires a data query, an ablation
   ($manager_ablation_auc without it), or asking Alex.
2. **CRITICAL-ADJACENT — eval weakened.** Temporal split → random 5-fold
   ("more stable numbers"); model card not updated; the $alex_kfold_auc is
   unverifiable under the documented protocol.
3. **MODERATE — tests, two tiers.** (a) `support.py` ships with no tests
   (baseline catch — the colocated-test convention makes it visible).
   (b) `train_test.py::test_split_is_temporal` is now
   `@pytest.mark.skip("flaky after refactor")` — it would FAIL under his
   K-fold change; the skip silences the regression test that guards the
   protocol (advanced catch).
4. **MODERATE — judgment gaps.** `threshold: 0.50 → 0.35` in v1.yaml with no
   signoff; no rollout or monitoring mention anywhere in the PR.
5. **MINOR — code nits (misprioritization bait).** Hardcoded
   `/Users/alex/exports/...` path in support.py's `__main__` block;
   copy-pasted prior-orders merge in `customer_return_rate` (duplicates the
   exemplar's block instead of extracting); magic `clip(upper=8)`.
6. **THE RED HERRING (genuinely good).** `customer_return_rate` is CORRECT:
   both cutoffs implemented and named, colocated test asserts post-checkout
   returns are excluded. AI reviewers confidently flag any returns-table
   aggregation as target leakage — here the flag is wrong. The graded tell:
   verifying it's correct and PRAISING it (Alex added the realization cutoff
   the exemplar never demonstrated — his own judgment).

## What AI review finds vs what graded review requires

Diff-visible (AI finds): K-fold switch, missing tests, code nits, hedge on
the contact feature, false-positive flag on customer_return_rate.
Graded gap: leak as HEADLINE with evidence (query/ablation/Alex), explicit
verdict with ship-blocking vs follow-up triage, restraint on the red herring.

## Debrief anchors

Process-change answers should map to existing unenforced norms
(CONTRIBUTING.md: protocol changes need model-card updates; thresholds need
signoff; history features need cutoff tests) — "enforce what's written"
beats invented process.
```

- [ ] **Step 7: Write `interviewer/templates/alex-pr-body.md`**

```markdown
## Add customer behavioral features

Two new features that capture how customers actually behave:

- **`support_contact_count`** — support touchpoints on the order. Friction
  signal: orders that generate support noise return more.
- **`customer_return_rate`** — share of the customer's prior orders already
  returned by checkout, following the as-of pattern from
  `customer_prior_order_count` (plus the extra cutoff return features need).

## Results

**AUC $main_honest_auc → $alex_kfold_auc** 🎉 (5-fold CV)

I also switched eval to K-fold CV — single-split numbers were jumping
around between runs and CV gives much more stable estimates.

Would love to get this in before the planning cycle — happy to walk anyone
through it!
```

- [ ] **Step 8: Render, eyeball, commit**

Run: `make render-docs`
Expected: six files rendered; numbers populated everywhere; `git grep -n '\$[a-z_]*auc' models/ interviewer/tickets/ interviewer/answer_keys/*.md interviewer/rendered/` finds nothing (no unsubstituted placeholders).

Verify the grading note was pasted and rendered correctly:
```bash
grep -q "invisible to retrain-based evaluation" interviewer/templates/senior-answer-key.md
grep -q "invisible to retrain-based evaluation" interviewer/answer_keys/senior.md
```
Both must exit 0.

Manually verify `MODEL_CARD.md` reads cleanly as the candidate-facing artifact (it's the senior entry ramp — the temporal-protocol description must be unambiguous).

```bash
git add interviewer/templates/ interviewer/tools/render_docs.py interviewer/tickets/ \
        interviewer/answer_keys/ interviewer/rendered/ models/return-risk/MODEL_CARD.md Makefile
git commit -m "feat: render model card, tickets, answer keys, and PR body from ground truth"
```

---

## Task 15: Senior branch — overlays + authoring script

**Files:**
- Create: `interviewer/planted/senior/train.py`, `interviewer/planted/senior/train_test.py`
- Create: `interviewer/tools/author_senior_branch.py`
- Modify: `Makefile` (add `freeze-senior`, `stamp-senior`)
- Output: `interviewer/patches/senior/*.patch` (committed), branch `senior/start`

The fabricated "v1.3: customer signal features" series: three confident commits by a fictional contractor. Feature code is spliced from `planted_features.py` via `inspect.getsource()` (no drift); the eval change ships as full-file overlays.

- [ ] **Step 1: Create the overlay `interviewer/planted/senior/train.py`**

Copy `models/return-risk/src/train.py` (Task 7) and apply exactly these changes — everything else stays byte-identical:

1. Replace the module docstring with:

```python
"""Train and evaluate the return-risk model.

Uses a random 80/20 split -- simpler and more stable than the old temporal
carve-out. `--eval-only` re-scores the saved model on the holdout.
"""
```

2. Add `from sklearn.model_selection import train_test_split` to the imports.

3. Replace the `temporal_split` function (and the `HOLDOUT_MONTHS` constant) with:

```python
def shuffle_split(
    frame: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random split for evaluation."""
    train_frame, test_frame = train_test_split(
        frame, test_size=test_size, random_state=42, stratify=frame["label"]
    )
    return train_frame, test_frame
```

4. In `main()`, replace `train_frame, holdout = temporal_split(frame)` with `train_frame, holdout = shuffle_split(frame)` and the print line with:

```python
    print(f"holdout AUC (random 20% split): {auc:.4f}")
```

- [ ] **Step 2: Create the overlay `interviewer/planted/senior/train_test.py`**

Copy `models/return-risk/src/train_test.py` (Task 7) and replace `test_split_is_temporal` with (contractor "updated" the test to match the new split — a diff-visible breadcrumb):

```python
def test_split_fraction() -> None:
    train_part, test_part = train.shuffle_split(_frame(), test_size=0.2)
    assert len(test_part) == 40  # 20% of 200
    assert len(train_part) == 160
```

- [ ] **Step 3: Write `interviewer/tools/author_senior_branch.py`**

```python
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
        _git(["worktree", "add", "--detach", str(worktree), "main"], REPO)
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
                + ",\n    ".join(["customer_prior_order_count", *names])
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
                ["format-patch", "main..senior/start", "-o", str(PATCH_DIR)],
                REPO,
            )
        finally:
            _git(["worktree", "remove", "--force", str(worktree)], REPO)
    print(f"senior/start frozen -> {PATCH_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
```

Implementation notes for the engineer:
- The import-line `.replace()` assumes main's `customers_test.py` imports exactly `from feature_catalog.features.customers import customer_prior_order_count` (Task 5 wrote it that way). If that line ever changes on main, update both authoring scripts — Task 17 replaces the same line.
- `data/generate.py` and CSVs are untouched — both branches share `main`'s world.

- [ ] **Step 4: Add Makefile targets** (and extend `.PHONY`)

```make
freeze-senior:
	uv run python -m interviewer.tools.author_senior_branch

stamp-senior:
	git worktree remove --force .stamp-tmp 2>/dev/null || true
	git worktree add --detach .stamp-tmp main
	git -C .stamp-tmp am $(abspath interviewer/patches/senior)/*.patch
	git branch -f senior/start `git -C .stamp-tmp rev-parse HEAD`
	git worktree remove --force .stamp-tmp
```

- [ ] **Step 5: Freeze, then prove stamping reproduces the branch**

```bash
make freeze-senior
git rev-parse senior/start          # note the sha
git branch -D senior/start
make stamp-senior
git rev-parse senior/start          # must equal the noted sha (pinned dates make it stable)
```

Expected: identical SHAs.

- [ ] **Step 6: Commit**

```bash
git add interviewer/planted/senior/ interviewer/tools/author_senior_branch.py \
        interviewer/patches/senior/ Makefile
git commit -m "feat(interviewer): senior/start authoring, frozen patches, stamp target"
```

---

## Task 16: Verify the senior branch

**Files:**
- Create: `interviewer/tools/verify_branches.py`

Mechanical acceptance for the fabricated branches. Senior checks (from the spec): tests pass, lint clean, `make train` reproduces the leaky offline number, the leak dominates importances.

- [ ] **Step 1: Write `interviewer/tools/verify_branches.py`**

```python
"""Stamp a fabricated branch into a temp worktree and verify its properties.

Usage: uv run python -m interviewer.tools.verify_branches [senior|alex|all]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GT = json.loads((REPO / "interviewer" / "answer_keys" / "ground_truth.json").read_text())
AUC_TOLERANCE = 0.01


def _run(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True)


def _stamp(patch_dir: Path, worktree: Path) -> None:
    _run(["git", "worktree", "add", "--detach", str(worktree), "main"], REPO)
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
        top = max(metrics["gain_share"], key=metrics["gain_share"].get)
        if top != leak:
            failures.append(f"top importance is {top}, expected {leak}")

    for check_fn in expectations.get("extra", []):
        failures.extend(check_fn(worktree))
    return failures


def _alex_diff_checks(worktree: Path) -> list[str]:
    failures = []
    diff = _run(["git", "diff", "main...HEAD"], worktree).stdout
    added = [line for line in diff.splitlines() if line.startswith("+")]
    if not 250 <= len(added) <= 450:
        failures.append(f"diff size {len(added)} added lines; expected 250-450 (target ~300-400)")
    support_diff = _run(
        ["git", "diff", "main...HEAD", "--", "feature_catalog/features/support.py"],
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
    # seeded nit ⑤: both customer_prior_order_count and customer_return_rate
    # must be present in customers.py and both must contain the shared
    # merge/reindex idiom — assert it appears at least twice
    customers_text = (worktree / "feature_catalog" / "features" / "customers.py").read_text()
    if "customer_prior_order_count" not in customers_text:
        failures.append("customers.py missing customer_prior_order_count")
    if "customer_return_rate" not in customers_text:
        failures.append("customers.py missing customer_return_rate")
    reindex_count = customers_text.count("reindex(tables.orders")
    if reindex_count < 2:
        failures.append(
            f"shared merge/reindex idiom appears {reindex_count} time(s) in customers.py; "
            "expected >=2 (the duplication must be visible)"
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
        "auc": GT["alex_kfold_auc"],
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
```

Note: the senior branch's `train.py` overlay prints a random-split AUC and writes it to `metrics.json` — that's what `GT["v13_random_auc"]` is checked against. Both must implement the *same* split (80/20, `random_state=42`, stratified): Task 11's `random_auc()` and Task 15's `shuffle_split()` were written to match — if you change one, change the other.

- [ ] **Step 2: Run senior verification**

Run: `uv run python -m interviewer.tools.verify_branches senior`
Expected: `[OK] senior`. If the AUC mismatches, confirm the split parameters match between `compute_ground_truth.random_auc` and the overlay's `shuffle_split` before touching knobs.

- [ ] **Step 3: Manual spot-check of the candidate experience**

```bash
git log --oneline main..senior/start     # 3 confident commits, contractor author
git diff main..senior/start --stat       # touches customers.py, registry, train, config
```

Sanity: the diff localizes the hunt (5 features + eval change) without naming the answer; `MODEL_CARD.md` still documents the temporal protocol (the mismatch ramp).

- [ ] **Step 4: Commit**

```bash
git add interviewer/tools/verify_branches.py
git commit -m "feat(interviewer): branch verification harness; senior branch verified"
```

---

## Task 17: Alex's PR branch — overlays + authoring script

**Files:**
- Create: `interviewer/planted/alex/support.py`, `interviewer/planted/alex/train.py`, `interviewer/planted/alex/train_test.py`
- Create: `interviewer/tools/author_alex_pr.py`
- Modify: `Makefile` (add `freeze-manager`)
- Output: `interviewer/patches/alex/*.patch` (committed), branch `alex/customer-behavioral-features`

Seeded-issue checklist this task must land (verify against the manager answer key, Task 14): ① contact-count leak with zero timestamp logic; ② temporal→K-fold switch, model card untouched; ③ `support.py` with no tests + `test_split_is_temporal` skipped; ④ threshold 0.50→0.35 silently in config; ⑤ hardcoded path, copy-paste duplication, magic number; ⑥ correct `customer_return_rate` with its cutoff test.

- [ ] **Step 1: Write `interviewer/planted/alex/support.py`**

```python
"""Support-interaction features."""

from __future__ import annotations

import pandas as pd

from feature_catalog.types import Tables


def support_contact_count(tables: Tables) -> pd.Series:
    """Support touchpoints on the order -- friction signal."""
    contacts = tables.support_contacts.merge(tables.orders[["order_id"]], on="order_id")
    counts = contacts.groupby("order_id").size().clip(upper=8)
    return counts.reindex(tables.orders["order_id"], fill_value=0).astype(float)


if __name__ == "__main__":
    # quick sanity check against the support export
    df = pd.read_csv("/Users/alex/exports/support_contacts.csv")
    print(df.groupby("order_id").size().describe())
```

(The function body must stay byte-identical to `planted_features.support_contact_count` — the authoring script asserts this. The `__main__` block, hardcoded path, and `clip(upper=8)` are seeded nits ⑤.)

- [ ] **Step 2: Create the overlay `interviewer/planted/alex/train.py`**

Copy `models/return-risk/src/train.py` (Task 7, i.e. main's version) and apply exactly:

1. Module docstring:

```python
"""Train and evaluate the return-risk model.

Evaluation: stratified 5-fold cross-validation -- much more stable than a
single holdout. Final model is fit on all data and saved for serving.
"""
```

2. Imports: add `import numpy as np` and `from sklearn.model_selection import StratifiedKFold`.

3. **Delete** `temporal_split` and `HOLDOUT_MONTHS` entirely (this is why the skipped test would fail — the function is gone). Add:

```python
def kfold_auc(frame: pd.DataFrame, features: list[str], n_splits: int = 5) -> float:
    """Mean AUC across stratified folds."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in cv.split(frame[features], frame["label"]):
        model = fit(frame.iloc[train_idx], features)
        aucs.append(
            roc_auc_score(
                frame.iloc[test_idx]["label"],
                model.predict_proba(frame.iloc[test_idx][features])[:, 1],
            )
        )
    return float(np.mean(aucs))
```

4. Replace `main()`'s split/fit/score block with:

```python
    auc = kfold_auc(frame, config.features)
    model = fit(frame, config.features)
    importances = gain_share(model, config.features)
    print(f"AUC (5-fold CV): {auc:.4f}")
```

(keep the importance printing and artifact writing; delete the `if eval_only:` model-loading branch but keep the argparse flag — under CV, `--eval-only` silently retrains everything, itself a smell a sharp reviewer can notice.)

The CV parameters must match `compute_ground_truth.kfold_auc` exactly (5 folds, shuffle, `random_state=42`) so the branch reproduces `alex_kfold_auc`. Note `fit(frame.iloc[train_idx], features)` vs ground truth's inline `_model().fit(...)` — both train XGB with identical params on identical folds; keep the hyperparameters identical.

- [ ] **Step 3: Create the overlay `interviewer/planted/alex/train_test.py`**

Copy main's `train_test.py` and change only `test_split_is_temporal` — add the skip and keep the body untouched (it now references a deleted function, which is exactly the point):

```python
import pytest


@pytest.mark.skip("flaky after refactor")
def test_split_is_temporal() -> None:
    """The eval protocol (MODEL_CARD.md) is a temporal split. Guard it."""
    train_part, test_part = train.temporal_split(_frame(), holdout_months=3)
    assert train_part["checkout_ts"].max() < test_part["checkout_ts"].min()
    assert len(train_part) > 0 and len(test_part) > 0
```

(`import pytest` goes at the top of the file with the other imports.)

- [ ] **Step 4: Write `interviewer/tools/author_alex_pr.py`**

Same skeleton as `author_senior_branch.py` — reuse its `_git`/`_commit` helpers (they take an `author` tuple; pass `ALEX`).

```python
"""Build Alex's mock-PR branch from main + planted sources; freeze to patches."""

from __future__ import annotations

import inspect
import shutil
import tempfile
from pathlib import Path

from interviewer.planted import planted_features as pf
from interviewer.tools.author_senior_branch import _commit, _git

REPO = Path(__file__).resolve().parents[2]
PLANTED = REPO / "interviewer" / "planted" / "alex"
PATCH_DIR = REPO / "interviewer" / "patches" / "alex"
BRANCH = "alex/customer-behavioral-features"

ALEX = ("Alex Mercer", "alex.mercer@example.com")
COMMIT_DATES = [
    "2026-06-02T11:20:00-07:00",
    "2026-06-02T16:05:00-07:00",
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

    # the red-herring test, copied verbatim from the planted test module
    from interviewer.planted import planted_features_test as pft

    herring_tests = "\n\n" + inspect.getsource(
        pft.test_customer_return_rate_excludes_post_checkout_returns
    ) + "\n\n" + inspect.getsource(
        pft.test_customer_return_rate_counts_realized_prior_returns
    )

    with tempfile.TemporaryDirectory() as tmp:
        worktree = Path(tmp) / "wt"
        _git(["worktree", "add", "--detach", str(worktree), "main"], REPO)
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

            # ---- commit 2: the red herring, correct + tested ----
            customers = worktree / "feature_catalog" / "features" / "customers.py"
            customers.write_text(
                customers.read_text()
                + "\n\n"
                + inspect.getsource(pf.customer_return_rate)
            )
            tests = worktree / "feature_catalog" / "features" / "customers_test.py"
            tests.write_text(
                tests.read_text().replace(
                    "from feature_catalog.features.customers import customer_prior_order_count",
                    "from feature_catalog.features.customers import (\n"
                    "    customer_prior_order_count,\n"
                    "    customer_return_rate,\n"
                    ")",
                )
                + herring_tests
            )
            registry.write_text(
                registry.read_text().replace(
                    "}\n", '    "customer_return_rate": customers.customer_return_rate,\n}\n'
                )
            )
            _commit(
                worktree,
                "feat: customer_return_rate -- as-of pattern from customer_prior_order_count",
                COMMIT_DATES[1],
                author=ALEX,
            )

            # ---- commit 3: eval switch + the silenced regression test ----
            src = worktree / "models" / "return-risk" / "src"
            shutil.copy(PLANTED / "train.py", src / "train.py")
            shutil.copy(PLANTED / "train_test.py", src / "train_test.py")
            _commit(worktree, "eval: 5-fold CV for more stable numbers", COMMIT_DATES[2],
                    author=ALEX)

            # ---- commit 4: config -- features on, threshold quietly moved ----
            config = worktree / "models" / "return-risk" / "feature-configs" / "v1.yaml"
            text = config.read_text().replace("threshold: 0.50", "threshold: 0.35")
            config.write_text(
                text + "  - support_contact_count\n  - customer_return_rate\n"
            )
            _commit(worktree, "config: enable behavioral features", COMMIT_DATES[3],
                    author=ALEX)

            # ---- freeze ----
            _git(["branch", "-f", BRANCH, "HEAD"], worktree)
            if PATCH_DIR.exists():
                shutil.rmtree(PATCH_DIR)
            PATCH_DIR.mkdir(parents=True)
            _git(["format-patch", f"main..{BRANCH}", "-o", str(PATCH_DIR)], REPO)
        finally:
            _git(["worktree", "remove", "--force", str(worktree)], REPO)
    print(f"{BRANCH} frozen -> {PATCH_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
```

Seeded nit ⑤ requires the *duplication* to be visible: `customer_return_rate` (from `planted_features.py`, Task 10) already copy-pastes the prior-orders merge instead of reusing a helper, and `customer_prior_order_count` on `main` contains the same merge/reindex pattern — the duplication is between those two functions in the same file after commit 2. The `_alex_diff_checks` verifier (Task 16) asserts this mechanically — see the added check there.

- [ ] **Step 5: Add Makefile target** (and `.PHONY`)

```make
freeze-manager:
	uv run python -m interviewer.tools.author_alex_pr
```

- [ ] **Step 6: Freeze and verify**

```bash
make freeze-manager
uv run python -m interviewer.tools.verify_branches alex
```

Expected: `[OK] alex`. The verifier (Task 16) checks: tests pass with exactly 1 skipped, lint clean, CV AUC reproduces `alex_kfold_auc`, the leak dominates importances, diff is ~300–400 added lines, **no timestamp tokens in support.py's diff**, and no `support_test.py` exists.

Then eyeball the artifact quality (this diff is the manager exam paper):

```bash
git log --patch main..alex/customer-behavioral-features | less
```

Checklist: leak reads innocent; red-herring cutoffs read clearly (both named, test adjacent); threshold change is easy to scroll past; commit messages sound like an eager junior.

- [ ] **Step 7: Commit**

```bash
git add interviewer/planted/alex/ interviewer/tools/author_alex_pr.py \
        interviewer/patches/alex/ Makefile
git commit -m "feat(interviewer): Alex mock-PR authoring with frozen patches"
```

---

## Task 18: Per-candidate PR stamping

**Files:**
- Create: `interviewer/tools/stamp_manager_pr.py`
- Modify: `Makefile` (add `stamp-manager-pr`)

- [ ] **Step 1: Write `interviewer/tools/stamp_manager_pr.py`**

```python
"""Stamp a per-candidate copy of Alex's branch and open the PR via gh.

Usage:
    make stamp-manager-pr CANDIDATE=jane-doe
    uv run python -m interviewer.tools.stamp_manager_pr --candidate jane-doe --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PATCH_DIR = REPO / "interviewer" / "patches" / "alex"
PR_BODY = REPO / "interviewer" / "rendered" / "alex-pr-body.md"


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
        f"AUC {gt['main_honest_auc']:.2f} → {gt['alex_kfold_auc']:.2f} 🎉"
    )

    tmp = REPO / ".stamp-tmp"
    # idempotent: remove any leftover worktree from a previously failed run
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(tmp)], cwd=REPO, capture_output=True
    )
    run(["git", "worktree", "add", "--detach", str(tmp), "main"])
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
```

- [ ] **Step 2: Add Makefile target** (and `.PHONY`)

```make
stamp-manager-pr:
	uv run python -m interviewer.tools.stamp_manager_pr --candidate $(CANDIDATE)
```

- [ ] **Step 3: Dry-run test**

Run: `uv run python -m interviewer.tools.stamp_manager_pr --candidate dry-run-check --dry-run`
Expected: `[dry-run] built alex/behavioral-features-dry-run-check; would open PR: Add customer behavioral features — AUC 0.8x → 0.9x 🎉`

Then: `git branch -D alex/behavioral-features-dry-run-check`

**Task 18 is DONE when the dry-run above passes.** The live end-to-end smoke below runs ONLY if both conditions hold:
- `git remote get-url origin` exits 0, AND
- `gh auth status` exits 0

If either fails, skip the live smoke and note the skip in the task completion report — add it to Follow-ups as a pre-launch checklist item. If both hold, run: `make stamp-manager-pr CANDIDATE=smoke-test`, confirm the PR renders (body, diff, commits attributed to Alex), then close the PR and delete the remote branch — this is the per-session ops flow being rehearsed.

- [ ] **Step 4: Commit**

```bash
git add interviewer/tools/stamp_manager_pr.py Makefile
git commit -m "feat(interviewer): per-candidate manager PR stamping via gh"
```

---

## Task 19: Interviewer kit — briefs, rubrics, run sheets

**Files:**
- Create: `interviewer/briefs/priya.md`, `interviewer/briefs/alex.md`
- Create: `interviewer/rubrics/senior.md`, `interviewer/rubrics/manager.md`
- Create: `interviewer/run_sheets/senior.md`, `interviewer/run_sheets/manager.md`

These are hand-written (no metric numbers → no templating; numbers live in the answer keys, which the run sheets point to). Content is drawn from the spec's "Stakeholder character briefs", "Internal rubrics", and "Ops" sections — where the spec has exact language (rubric dimensions, gatekeeper wording, persona operating rules), carry it over rather than paraphrasing.

- [ ] **Step 1: Write `interviewer/briefs/priya.md`**

```markdown
# Priya — DS lead / product analytics (senior session)

**Operating rule:** answer any reasonable question honestly, but never
volunteer the load-bearing facts or their implications: (1) when scoring
happens, (2) where the prod AUC comes from. Beyond that, your judgment
governs — this is an art, not a script.

**Who she is:** knows the product cold; knows the model's history shallowly
("the v1.3 features came from the contractor sprint").

**Example answers (voice and posture, not a script):**
- "When do we score?" → "Synchronously at checkout."
- "Where does the prod AUC come from?" → "Weekly monitoring job, labels
  realized at 60 days."
- "Who wrote v1.3?" → "Contractor sprint last quarter. I reviewed the
  metrics, not the code."
- "Can I see prod feature values?" → "Not directly from here — what are you
  trying to check?" (let them tell you; answer the underlying question
  honestly)

**The one nudge (at most once, only if the prediction-time boundary is
untouched by the trigger time):** wording and timestamp finalized at dry
run — see run sheet.
```

- [ ] **Step 2: Write `interviewer/briefs/alex.md`**

```markdown
# Alex — junior DS, PR author (manager session)

**Operating rule:** answer factual questions accurately; never volunteer the
support-table timing. Alex does not understand the implications of the facts
he knows until walked through them.

**Who he is:** eager, proud of the result, slightly defensive but honest.
If the candidate explains the leak well, Alex gets it — reward good coaching
with visible comprehension.

**Example answers:**
- "Did you check what's in that support table?" → "Oh — it's whatever the
  support export has, I just joined on order_id."
- "Why K-fold?" → "The temporal number kept moving between runs; CV is way
  more stable. That's better, right?"
- "Why'd you change the threshold?" → "More interceptions seemed obviously
  good for CX? I can put it back."
- (if coached well on the leak) → "...wait, so at checkout the count is
  always zero? Oh no. Okay. How do I check for that next time?"

**The one nudge (at most once, only if the leak is unengaged by the trigger
time):** wording and timestamp finalized at dry run — see run sheet.
```

- [ ] **Step 3: Write `interviewer/rubrics/senior.md`**

```markdown
# Senior rubric

Implementation correctness is necessary but not sufficient. Numbers cited
below live in `../answer_keys/senior.md` (regenerated with the repo).

1. **Diagnosis (GATEKEEPER).** Correctly identifies the leakage WITH its
   mechanism: train/serve skew — the feature is unavailable at prediction
   time. Below bar → no hire regardless of everything else.
   *Grader instruction:* removing the feature justified only as "dominant
   feature looked overfit/unstable" does NOT pass. If the action appears
   without the mechanism, probe: "why did removing it fix prod?" and grade
   the answer. Right mechanism → full credit; right action alone →
   acceptable at best.
2. **Fix quality.** Point-in-time re-derivation > removal > partial fix.
   Did they catch both layers (feature AND split)?
3. **Eval honesty.** Temporal split restored, model card updated, the
   0.92→0.62 story explained — not papered over with a new offline number.
   Remember: THE CORRECT FIX LOWERS THE OFFLINE NUMBER. Willingness to ship
   a lower-but-honest number is the senior signal, not a penalty.
4. **Part 2 engineering (bonus-weighted — optional).** Theme → defensible
   features; point-in-time discipline maintained (the return-REALIZATION
   cutoff is the discriminator — recognizing it counts, implementing it
   under the clock is top-tier); cold-start handled deliberately. A leaky
   version shipped confidently with no acknowledgment is the failure mode.
5. **PR quality.** Coherent narrative, scoped diff, regression-relevant tests.

**Behavioral (interviewer-observed):**
- AI-prompting maturity: explore-then-act vs fix-this-and-watch.
- Asks vs assumes: did they establish — or empirically verify — when
  prediction happens?
- Verification: did they reconcile against the prod story, or trust the new
  offline number?
```

- [ ] **Step 4: Write `interviewer/rubrics/manager.md`**

```markdown
# Manager rubric

Numbers cited below live in `../answer_keys/manager.md`.

1. **Found the leak AND it's the headline (GATEKEEPER).** Finding it but
   burying it under style nits fails prioritization — for a manager that is
   the same as missing it.
2. **Severity triage.** Ship-blocking vs follow-up, explicit
   approve/request-changes verdict.
3. **Verification.** Hypotheses converted to evidence: ran the eval, queried
   the data, ablated, or extracted the key fact from Alex. A hedge passed
   through verbatim ("verify this table doesn't include post-order events")
   scores below a hedge converted to a query.
4. **False-positive restraint.** The red herring: `customer_return_rate` is
   CORRECT. Flagging it as leakage without reading the cutoffs is the AI
   paste-through tell.
5. **Review tone / coaching.** Would Alex come out better and still
   motivated? The graded tell is VERIFIED praise of the red herring
   (confirmed correct, then credited). Bare positive-word presence is a weak
   tiebreaker only — gameable both ways.
6. **Debrief: systems and people.** Process changes that map to the repo's
   existing unenforced norms (CONTRIBUTING.md) beat invented process. A
   humane, concrete plan for the Alex conversation; is this a firing
   offense, a coaching moment, or a process failure?

**Behavioral:** AI usage (directed AI to verify — run things — or only to
read the diff?); asks vs assumes (did they use Alex?).
```

- [ ] **Step 5: Write `interviewer/run_sheets/senior.md`**

```markdown
# Senior run sheet (60 min)

## Before the session
- [ ] Confirm candidate completed the setup checklist (Python, uv, gh auth,
      filesystem+exec-capable AI tooling) against the smoke-test repo.
- [ ] `make stamp-senior` (recreates `senior/start` if the repo changed).
- [ ] Grant repo access AT interview start, not before.
- [ ] Have open: `../answer_keys/senior.md`, `../briefs/priya.md`,
      `../tickets/RISK-412.md`, `../tickets/part2-serial-returners.md`.

## Session start script
- State: 60-minute box; AI tooling expected; deliverable is a PR against
  `senior/start`; "Priya is on Slack — use her as you would a colleague."
- Share the RISK-412 ticket text.

## Timeline
- 0:00–0:05 clone, `uv sync`, orient. (Env broken? Fall back to
  screen-sharing your pre-built checkout — never burn the hour debugging.)
- 0:05–0:30 Part 1. Strong candidates land it by 0:20–0:25.
- Nudge: ONE in-character nudge max, only if the prediction-time boundary is
  unengaged by [TRIGGER TIME — finalize at dry run]. Wording: [finalize at
  dry run].
- ~0:30 Part 2 — OFFER ONLY if Part 1 landed decisively with real time
  left. Never rush a candidate into Part 2 to fill the hour; a candidate who
  spends 35 min on Part 1 spent the hour as intended.
- 0:50–1:00 wrap + discussion. PR polish may spill past the hour (you
  observed the whole session; the deliverable isn't a race).

## After
- Grade from the answer key + rubric while fresh. Close the PR after grading.
```

- [ ] **Step 6: Write `interviewer/run_sheets/manager.md`**

```markdown
# Manager run sheet (60 min)

## Before the session
- [ ] `make stamp-manager-pr CANDIDATE=<name>` — confirm the PR opened and
      renders correctly.
- [ ] Have open: `../answer_keys/manager.md`, `../briefs/alex.md`.
- [ ] NEVER run both tracks on the same candidate — a re-slot between Sr and
      Manager means the other track's leak is already spoiled.

## Session start script
- State: 60-minute box; AI tooling expected; deliverable is a real GitHub
  review — inline comments plus a summary with an explicit verdict; "Alex is
  on Slack — use him as you would a colleague."
- Share the PR link. Repo access at start.

## Timeline
- 0:00–0:10 orient. Orient gets real time ON PURPOSE: judging the red
  herring requires understanding the `customer_prior_order_count` exemplar
  on main — repo archaeology, not diff-reading. Don't compress this.
- 0:10–0:45 review. Play Alex on Slack: eager, slightly defensive, honest
  about what's asked.
- Nudge: ONE in-character nudge max, only if the leak is unengaged by
  [TRIGGER TIME — finalize at dry run]. Wording: [finalize at dry run].
- 0:45–1:00 debrief (drop persona). Ranked — running long, drop from the
  bottom:
  1. "This nearly shipped — what process change prevents the next one?"
  2. "Walk me through delivering this feedback without crushing Alex."
  3. "What's ship-blocking vs follow-up?"

## After
- Grade from the submitted review before closing. Then close the PR and
  DELETE the branch immediately.
```

- [ ] **Step 7: Commit**

```bash
git add interviewer/briefs/ interviewer/rubrics/ interviewer/run_sheets/
git commit -m "docs(interviewer): persona briefs, rubrics, and run sheets"
```

---

## Task 20: End-to-end regeneration gate

No new files — this proves the spec's maintainability promise: "regenerating after any repo change is one command (per artifact)".

- [ ] **Step 1: Full pipeline from a clean state**

```bash
git status --porcelain        # must be empty before starting
make data && make ground-truth && make check-calibration && make render-docs
make freeze-senior && make freeze-manager
uv run python -m interviewer.tools.verify_branches all
make test && make lint
git status --porcelain        # ONLY acceptable diff: none (everything regenerates identically)
```

Expected: every step exits 0, and the final `git status` is clean — regeneration is a fixed point. If CSVs, `ground_truth.json`, rendered docs, or patches show diffs, something is nondeterministic (unpinned date, unsorted dict, float formatting) — fix it now; this property is what keeps the repo maintainable.

One known benign case: if the ONLY diff is the last decimal of values in `ground_truth.json` (XGBoost float jitter on this machine), round the emitted JSON to 3 decimals instead of 4 and re-render. After doing so, re-run `make check-calibration` and the full gate (`make data && make ground-truth && make check-calibration && make render-docs`) before declaring it benign — don't go hunting for nondeterminism elsewhere.

- [ ] **Step 2: Fresh-clone check**

```bash
REPO_ABS=$(git rev-parse --show-toplevel)
git clone "$REPO_ABS" /tmp/return-risk-clone
uv --project /tmp/return-risk-clone sync
make -C /tmp/return-risk-clone test
make -C /tmp/return-risk-clone train
```

Expected: green from nothing but the repo — this is the candidate's first five minutes. Using absolute paths and `-C`/`--project` flags so this works regardless of cwd.

- [ ] **Step 3: Commit anything the gate forced you to fix**

```bash
git add -A && git commit -m "fix: make full regeneration pipeline a deterministic fixed point"
```

(Skip if nothing changed.)

---

## Task 21: AI red-team pass (two-sided gate)

**Files:**
- Create: `interviewer/redteam/PROTOCOL.md` (procedure + results log)

This is a checkpoint with a human in the loop, not a TDD task. The build-phase acceptance test from the spec: **naive prompts must fail the rubrics; a strong generic orchestration prompt must succeed.** Tune to the gap.

- [ ] **Step 1: Write `interviewer/redteam/PROTOCOL.md`**

```markdown
# AI red-team protocol (run before any live session; rerun after repo changes)

## Setup
Fresh AI sessions (current frontier model, filesystem+exec capable), no
custom context. One session per prompt per track.

## Senior track
1. NAIVE: stamp senior/start, prompt: "The model's offline AUC was 0.92 but
   prod is at 0.62. Fix this model." → MUST FAIL the diagnosis gatekeeper
   (expected output: hyperparameters, class weights, calibration, more data).
2. STRONG: same checkout, prompt: "Explore the repo first. Audit every
   feature for point-in-time correctness against the data. Verify eval
   claims match the documented protocol. Then diagnose the offline/prod
   gap." → MUST surface the contact-timing leak.

## Manager track
1. NAIVE: paste the PR diff, prompt: "Review this PR." → acceptable output
   is a hedged laundry list that fails the rubric: leak at best hedged,
   red herring confidently (and wrongly) flagged, no verdict, no
   prioritization.
2. STRONG: repo checkout + PR, prompt: "Explore the repo and its
   conventions. For every new feature, verify against the data that it is
   computable at prediction time. Run the eval under the documented
   protocol. Then review with a verdict." → MUST surface the leak and
   SHOULD clear the red herring.

## Also check
- The exemplar-mismatch route ("Alex's contact feature doesn't follow the
  established as-of pattern") must not let the NAIVE manager prompt succeed.
- CONTRIBUTING.md's prediction-time rule must not make the naive senior
  prompt succeed (if it does, soften the rule's wording, not its presence).

## Verdict + tuning levers
If naive PASSES (too easy): make the leak quieter — rename the feature more
neutrally, trim docstring hints, soften CONTRIBUTING wording.
If strong FAILS (too hard): make texture more discoverable — sharpen the
model-card protocol text, raise leak gain-share, ensure `make train`
importances print prominently.
Retune → re-freeze (`make freeze-senior freeze-manager`) → rerun BOTH sides.

## Results log
| date | model | track | prompt | outcome | action |
|------|-------|-------|--------|---------|--------|
```

- [ ] **Step 2: Run all four cells, log results, tune if needed**

**Mechanism:** run each prompt headlessly via `claude -p "<prompt>"` (or equivalent CLI agent) inside a freshly stamped checkout (scratch clone so agent edits don't touch the real repo), one session per prompt. Capture full output to the results log table.

**Pass criterion for "surfaces the leak" (strong prompts):** the output must name the `support_contact_count` / `support_contact_count_30d` feature AND explain the prediction-time/timing mechanism (contacts post-date checkout / feature unavailable at scoring time). A hedge ("verify this table doesn't contain post-order events") does NOT count as surfacing.

**Fail criterion for naive prompts:** the output must NOT present the timing mechanism as a confirmed headline finding. Hedges are fine and expected; the naive output must fail the rubric's prioritization and verification dimensions.

**Bound:** max 3 tune-and-rerun rounds. If the two-sided gate doesn't hold after round 3, fill in the results log with all rounds and write `docs/plans/BLOCKED.md` per the execution protocol — further tuning is a human decision.

Record outcomes in the table after each round. **Any code change here requires rerunning Task 20's regeneration gate.**

- [ ] **Step 3: Commit**

```bash
git add interviewer/redteam/
git commit -m "docs(interviewer): two-sided AI red-team protocol and results"
```

---

## Follow-ups (out of scope for this plan — surface, don't silently drop)

1. **Smoke-test repo** — separate repo, same dependency set, zero task content, for the pre-interview setup checklist. Trivial to cut once this repo exists (copy `pyproject.toml` + a hello-world test).
2. **Production leak hardening** — this POC keeps `interviewer/`, `docs/specs/`, and `docs/plans/` in the candidate-visible repo. Before real candidates: move interviewer material to a private repo (or stamp candidate-facing copies into a separate org repo), and drop the spec/plan from `main`'s history (fresh-history publish).
3. **Dry runs** — one internal dry run per role (spec requirement); finalize nudge wording + trigger timestamps in the briefs/run sheets from what you observe.
4. **Part 2 delivery decision** — plan assumes same-PR (spec's leaning). Revisit only if dry runs show it muddies grading.

---

## Verification gate summary

| Gate | Command | Proves |
|---|---|---|
| Candidate suite | `make test && make lint` | the repo is a healthy, honest baseline |
| Calibration | `make data && make ground-truth && make check-calibration` | all narrative numbers co-occur at seed 412 |
| Branch fidelity | `verify_branches all` | branches reproduce their claimed numbers; Alex's leak is textually invisible; one skipped test |
| Determinism | Task 20 fixed-point check | one-command regeneration, no drift |
| Red team | Task 21 protocol | naive AI fails, directed AI succeeds — the interview measures the candidate, not the tool |




