"""Assert every narrative number co-occurs at the pinned seed.

Reads interviewer/answer_keys/ground_truth.json (run `make ground-truth`
first). Exits nonzero with a per-check table if any band or ordering fails.

Two check modes (calibration contract B2):

- seed-412: level/band checks read the canonical single-seed point values.
- median[N]: noise-sensitive deltas (drift gap, rederived_lift, part2_lift)
  are smaller than single-seed variance, so they read the multi-seed MEDIAN
  from ground_truth.json's `multi_seed` block (written by
  `compute_ground_truth --seeds 412,7,99,2024,31337`).

Thresholds are pinned from the measured N=5 seed spread at the final config
(2026-06-09, PRE_CONTACT_LOGIT_LIFT=3.0; see the tuning log) -- every band
clears the observed min/max with margin, not aspiration. Observed spreads in
the inline comments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GT_PATH = (
    Path(__file__).resolve().parents[2] / "interviewer" / "answer_keys" / "ground_truth.json"
)

# (low, high) inclusive bands on seed-412 point values.
# Observed N=5 spread at the final config in comments.
BANDS: dict[str, tuple[float, float]] = {
    "baseline_auc": (0.79, 0.84),             # obs 0.8047-0.8106; "real perf ~0.80"
    "v13_random_auc": (0.90, 0.95),           # obs 0.9296-0.9326; "offline claims ~0.93"
    "v13_temporal_auc": (0.86, 0.92),         # obs 0.8976-0.9028; layer-1-only fix
    "prod_sim_auc": (0.79, 0.84),             # obs 0.8072-0.8122; reversion to baseline band
    "senior_fixed_auc": (0.79, 0.84),         # obs 0.8058-0.8129; correct fix = baseline band
    "leak_gain_share": (0.40, 0.75),          # obs 0.5593-0.6127; leak dominant but not total
    "post_checkout_contact_share": (0.85, 0.95),  # obs 0.9166-0.9198; "~90% post-date checkout"
    "leak_serving_time_mean": (0.0, 0.06),    # obs 0.0298-0.0306; leak is dead at checkout
}

# (low, high) inclusive bands on multi-seed MEDIANS (noise-sensitive levels).
MEDIAN_BANDS: dict[str, tuple[float, float]] = {
    # obs medians 0.0063 (per-seed 0.0052-0.0074): the re-derived pre-checkout
    # feature carries a modest GENUINE lift (assertion (a)) but stays scoreless
    # in interview terms (A3) -- clearly positive, nowhere near a score beat.
    "rederived_lift": (0.002, 0.02),
}

# (label, smaller_key, larger_key, min_gap) on seed-412 point values.
ORDERINGS: list[tuple[str, str, str, float]] = [
    # obs 0.1187-0.1252: the collapse is the offline lie (B1) -- ~0.93 claimed,
    # ~0.80 the moment the leak reads 0 at serving
    ("collapse: v13_random - prod_sim", "prod_sim_auc", "v13_random_auc", 0.09),
    # obs 0.345-0.358: return rates genuinely drift upward across the window
    ("drift exists: monthly return rate last - first",
     "monthly_return_rate_first", "monthly_return_rate_last", 0.10),
]

# (label, key_a, key_b, max_abs_diff) on seed-412 point values.
CLOSENESS: list[tuple[str, str, str, float]] = [
    # obs |diff| 0.0003-0.0014: reversion -- removing the leak lands exactly
    # where the deployed model already is in prod
    ("reversion: |prod_sim - senior_fixed|", "prod_sim_auc", "senior_fixed_auc", 0.01),
]

# (label, metric, min_median) on multi-seed MEDIANS.
MEDIAN_THRESHOLDS: list[tuple[str, str, float]] = [
    # obs median 0.0299 (per-seed 0.0293-0.0349): the swapped eval genuinely
    # flatters the number; pinned below the observed median with margin
    ("drift gap: median(v13_random - v13_temporal)", "drift_gap", 0.025),
    # obs median 0.0813 (per-seed 0.0772-0.0842): part 2 earns a real lift
    ("part 2: median(part2_auc - baseline_auc)", "part2_lift", 0.06),
]


def main() -> None:
    gt = json.loads(GT_PATH.read_text())
    multi = gt.get("multi_seed")
    failures: list[str] = []

    def report(label: str, mode: str, value: str, target: str, ok: bool, why: str) -> None:
        print(f"{label:<52s} {mode:<10s} {value:>8s}  {target:<18s} {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(why)

    print(f"{'check':<52s} {'mode':<10s} {'value':>8s}  {'target':<18s} result")

    for key, (low, high) in BANDS.items():
        if key not in gt:
            report(key, "seed-412", "", f"[{low}, {high}]", False,
                   f"{key}: missing from ground_truth.json")
            continue
        value = gt[key]
        report(key, "seed-412", f"{value:.4f}", f"[{low}, {high}]", low <= value <= high,
               f"{key}={value:.4f} outside [{low}, {high}]")

    n_seeds = len(multi["seeds"]) if multi else 0
    median_mode = f"median[{n_seeds}]"

    def get_median(metric: str) -> float | None:
        if multi is None or metric not in multi:
            return None
        return multi[metric]["median"]

    for key, (low, high) in MEDIAN_BANDS.items():
        value = get_median(key)
        if value is None:
            report(key, median_mode, "", f"[{low}, {high}]", False,
                   f"{key}: no multi_seed median; run compute_ground_truth --seeds ...")
            continue
        report(key, median_mode, f"{value:.4f}", f"[{low}, {high}]", low <= value <= high,
               f"median {key}={value:.4f} outside [{low}, {high}]")

    for label, small, large, gap in ORDERINGS:
        if small not in gt or large not in gt:
            report(label, "seed-412", "", f">= {gap}", False,
                   f"{label}: metric missing from ground_truth.json")
            continue
        diff = gt[large] - gt[small]
        report(label, "seed-412", f"{diff:.4f}", f">= {gap}", diff >= gap,
               f"{label}: {diff:.4f} < {gap}")

    for label, key_a, key_b, tol in CLOSENESS:
        if key_a not in gt or key_b not in gt:
            report(label, "seed-412", "", f"<= {tol}", False,
                   f"{label}: metric missing from ground_truth.json")
            continue
        diff = abs(gt[key_a] - gt[key_b])
        report(label, "seed-412", f"{diff:.4f}", f"<= {tol}", diff <= tol,
               f"{label}: {diff:.4f} > {tol}")

    for label, metric, threshold in MEDIAN_THRESHOLDS:
        value = get_median(metric)
        if value is None:
            report(label, median_mode, "", f">= {threshold}", False,
                   f"{label}: no multi_seed median; run compute_ground_truth --seeds ...")
            continue
        report(label, median_mode, f"{value:.4f}", f">= {threshold}", value >= threshold,
               f"{label}: median {value:.4f} < {threshold}")

    if failures:
        print(f"\n{len(failures)} calibration failure(s):")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    seeds = multi["seeds"] if multi else [gt["seed"]]
    print(f"\nall narrative numbers co-occur at seed {gt['seed']} "
          f"(medians over seeds {seeds})")


if __name__ == "__main__":
    main()
