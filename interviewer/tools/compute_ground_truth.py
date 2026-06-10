"""Train every narrative variant and emit ground_truth.json.

Every number quoted anywhere (tickets, answer keys, model card, commit
messages, PR body) comes from this file's output. Never hand-type a metric.

Two modes:

- default (single seed): read the CSVs already in data/ (the SEED=412
  artifact), compute the consolidated metric set, write ground_truth.json.
  An existing `multi_seed` block is carried over unchanged.
- `--seeds 412,7,99,2024,31337`: regenerate the data per seed, compute the
  full metric set each time, and write per-seed values + medians for the
  noise-sensitive deltas (drift gap, rederived_lift, part2_lift) into a
  `multi_seed` block (calibration contract B2). The point values rendered
  into docs are always the seed-412 run; the tree is left regenerated at
  seed 412 regardless of seed order.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import median

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
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
LEAK = "support_contact_count_30d"
PART2_FEATURE = "customer_return_rate"
HOLDOUT_MONTHS = 3
DEFAULT_SEED = 412

# noise-sensitive deltas asserted as multi-seed medians (contract B2)
MULTI_SEED_METRICS = ("drift_gap", "rederived_lift", "part2_lift")


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
    """The contractor's weakened eval (B4a): a shuffled 80/20 split."""
    train, test = train_test_split(
        frame, test_size=0.2, random_state=42, stratify=frame["label"]
    )
    model = _model().fit(train[features], train["label"])
    return float(roc_auc_score(test["label"], model.predict_proba(test[features])[:, 1]))


def prod_sim_auc(frame: pd.DataFrame, features: list[str], dead_feature: str) -> float:
    """The literal deployed v1.3 model (B4b): trained on main+chaff+leak,
    scored with the leak forced to 0.

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


def compute_metrics() -> dict[str, float]:
    """Consolidated metric set over the CSVs currently in data/."""
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

    v13 = main_features + V13_CHAFF + [LEAK]

    gt: dict[str, float] = {}
    gt["baseline_auc"] = temporal_auc(frame, main_features)
    gt["v13_random_auc"] = random_auc(frame, v13)              # the claimed offline number
    gt["v13_temporal_auc"] = temporal_auc(frame, v13)          # layer-1 partial fix
    gt["prod_sim_auc"] = prod_sim_auc(frame, v13, LEAK)        # reversion to baseline
    gt["leak_gain_share"] = leak_gain_share(frame, v13, LEAK)  # importance ramp
    gt["senior_fixed_auc"] = temporal_auc(
        frame, main_features + V13_CHAFF
    )  # good-tier fix: remove leak, keep chaff, temporal split
    gt["rederived_auc"] = temporal_auc(
        frame, main_features + V13_CHAFF + ["support_contact_count_pre_checkout"]
    )  # best-tier fix
    gt["rederived_lift"] = gt["rederived_auc"] - gt["senior_fixed_auc"]
    # alex's PR headline: his own feature set (main + the leak only) on the
    # weakened random split. Distinct from v13_random_auc, which is the senior
    # v1.3 feature set (main + chaff + leak under a different key). customer_
    # return_rate was removed from alex's PR (spec A1): it lifted the leak-
    # removed baseline toward ~0.89 and destroyed the leak's load-bearing role,
    # so it is now the senior's Part-2-only exemplar and absent from alex's PR.
    gt["alex_random_auc"] = random_auc(
        frame, main_features + ["support_contact_count"]
    )
    # alex's prod simulation: his deployed model (main + the leak under his
    # join-count key) scored with the leak forced to 0 at serving time. On
    # alex's own feature set this reverts to ~baseline -- the manager-key
    # prod-sim row points here, NOT at the senior v1.3 prod_sim_auc.
    gt["alex_prod_sim_auc"] = prod_sim_auc(
        frame, main_features + ["support_contact_count"], "support_contact_count"
    )
    # part 2 narrative (reference implementation = customer_return_rate)
    gt["part2_auc"] = temporal_auc(frame, main_features + [PART2_FEATURE])
    gt["part2_lift"] = gt["part2_auc"] - gt["baseline_auc"]
    gt["drift_gap"] = gt["v13_random_auc"] - gt["v13_temporal_auc"]
    gt.update(texture_stats(tables))
    return gt


def regenerate(seed: int) -> None:
    subprocess.run(
        [sys.executable, str(REPO / "data" / "generate.py"), "--seed", str(seed)],
        check=True,
    )


def write_out(gt: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def _round(value: object) -> object:
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, dict):
            return {k: _round(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_round(v) for v in value]
        return value

    OUT.write_text(json.dumps({k: _round(v) for k, v in gt.items()}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        help="comma-separated seed list for the multi-seed sweep (B2), "
        "e.g. 412,7,99,2024,31337; regenerates data per seed and leaves "
        "the tree at seed 412",
    )
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
        # run 412 last so the tree (and the point values) end at the canonical seed
        ordered = [s for s in seeds if s != DEFAULT_SEED] + [DEFAULT_SEED]
        per_seed: dict[int, dict[str, float]] = {}
        for seed in ordered:
            regenerate(seed)
            per_seed[seed] = compute_metrics()
            print(f"seed {seed}: " + ", ".join(
                f"{m}={per_seed[seed][m]:.4f}" for m in MULTI_SEED_METRICS
            ))
        gt: dict = {"seed": DEFAULT_SEED, **per_seed[DEFAULT_SEED]}
        gt["multi_seed"] = {
            "seeds": seeds if DEFAULT_SEED in seeds else seeds + [DEFAULT_SEED],
        }
        for metric in MULTI_SEED_METRICS:
            values = [per_seed[s][metric] for s in gt["multi_seed"]["seeds"]]
            gt["multi_seed"][metric] = {"values": values, "median": median(values)}
        # full per-seed table on stdout for threshold pinning
        keys = list(next(iter(per_seed.values())))
        print(f"\n{'metric':<32s}" + "".join(f"{s:>10d}" for s in gt["multi_seed"]["seeds"]))
        for key in keys:
            row = "".join(f"{per_seed[s][key]:>10.4f}" for s in gt["multi_seed"]["seeds"])
            print(f"{key:<32s}{row}")
    else:
        gt = {"seed": DEFAULT_SEED, **compute_metrics()}
        if OUT.exists():  # keep the last sweep's medians for check_calibration
            previous = json.loads(OUT.read_text())
            if "multi_seed" in previous:
                gt["multi_seed"] = previous["multi_seed"]

    write_out(gt)
    for key, value in gt.items():
        if isinstance(value, float):
            print(f"{key:>32s}: {value:.4f}")
        elif key == "multi_seed":
            for metric in MULTI_SEED_METRICS:
                print(f"{key + '.' + metric + '.median':>32s}: {value[metric]['median']:.4f}")
        else:
            print(f"{key:>32s}: {value}")


if __name__ == "__main__":
    main()
