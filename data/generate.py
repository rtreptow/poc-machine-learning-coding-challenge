"""Deterministic synthetic data for the return-risk repo.

`make data` runs this; the same SEED always produces identical CSVs.

The world it builds:
- ~120k orders across an 18-month window, ~6k customers (~20 orders/customer).
- A serial-returner segment that returns far more often but is otherwise
  decoupled (same order frequency, same bracket-buying as everyone) -- it leaves
  no observable footprint, recoverable only from return history (Part-2 lift).
- Observable order shape (value, item count, multi-size, prior_order_count) each
  carries genuine return signal on its own merits via direct logit weights.
- Base return rates rise and customer mix shifts across the window, and one
  observable weight (item_count) drifts over time (concept drift).
- Support contacts: a small share happen before checkout (pre-purchase
  friction, mildly predictive of a return); most happen after checkout,
  prompted by the return itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

SEED = 412

# ---- volume / window ----
N_CUSTOMERS = 6_000
TARGET_ORDERS = 120_000         # ~20 orders/customer -- per-customer density sharpens the
                                # return-history recovery (part2 ~0.89) without row bloat
WINDOW_START = pd.Timestamp("2024-11-01")
WINDOW_END = pd.Timestamp("2026-05-01")  # 18 months

# ---- calibration knobs ----
# Initial guesses. Tuned during the calibration loop against
# interviewer/tools/check_calibration.py; do not treat as final.
SERIAL_FRAC = 0.18              # share of customers in the serial-returner segment
SERIAL_ORDER_RATE_MULT = 1.0    # DECOUPLED: serial no longer orders more (no observable footprint)
SERIAL_LATE_SKEW = 1.6          # >1 skews serial order volume late (mix drift via timing)
SERIAL_LOGIT_BOOST = 4.5        # added to the return logit for serial customers
SERIAL_MULTI_SIZE_P = 0.10      # DECOUPLED: serial bracket-buys at the base rate
                                # (= BASE_MULTI_SIZE_P)
BASE_MULTI_SIZE_P = 0.10

BASE_RETURN_LOGIT_START = -1.9  # base return logit at window start ...
BASE_RETURN_LOGIT_END = -1.1    # ... and at window end (rates drift upward)

W_LOG_ORDER_VALUE = 1.75        # return-logit weights on observable order shape
                                # (cranked to rebuild baseline; D1 centered-baseline)
W_MULTI_SIZE = 1.4
W_PRIOR_ORDERS = 0.80           # NEW: independent prior_order_count term -- frequent shoppers
                                # return somewhat more, unrelated to `serial` (serving-safe)
# Concept drift: ONE dense ranking weight (item_count) is time-varying. The weight
# applied to z-scored item_count interpolates START->END across the window by t_frac,
# so a temporally-trained model applies stale weights to the late holdout (drift gap).
# item_count is the drift carrier (distinct from W_LOG_ORDER_VALUE, the primary baseline lever).
# Mean level kept meaningful so it carries baseline signal AND has leverage for the 2b drift gap.
W_ITEM_COUNT_START = 2.60
W_ITEM_COUNT_END = 0.20
LATENT_SD = 0.5                 # per-customer latent propensity: invisible in order shape,
                                # recoverable from return history (Part 2 lift);
                                # demoted to noise floor

PRE_CONTACT_P = 0.03            # P(pre-checkout support contact on an order)
PRE_CONTACT_LOGIT_LIFT = 3.0    # genuine signal: pre-purchase friction -> returns.
                                # Sized so the re-derived pre-checkout feature buys a
                                # small REAL lift (~0.005-0.01 AUC, multi-seed median)
                                # -- a method tell with a modest genuine payoff, never
                                # a score beat (capped by PRE_CONTACT_P=0.03 sparsity).
POST_CONTACT_P_RETURNED = 0.65  # returned orders usually generate a contact ...
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
    for oid, k, bracket in zip(orders["order_id"], item_counts, multi_size, strict=True):
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

    # prior_order_count: orders this customer placed strictly before this checkout.
    # orders is sorted by checkout_ts, so cumcount() over customer gives the as-of
    # count (matches feature_catalog.features.customers.customer_prior_order_count).
    prior_order_count = orders.groupby("customer_idx").cumcount().to_numpy()

    # Time-varying drift weight on z-scored item_count (the drift carrier).
    w_item_count_t = (
        W_ITEM_COUNT_START
        + (W_ITEM_COUNT_END - W_ITEM_COUNT_START) * orders["t_frac"].values
    )

    # ---- returns ----
    pre_contact = rng.random(n) < PRE_CONTACT_P
    logit = (
        BASE_RETURN_LOGIT_START
        + (BASE_RETURN_LOGIT_END - BASE_RETURN_LOGIT_START) * orders["t_frac"].values
        + np.where(orders["serial"], SERIAL_LOGIT_BOOST, 0.0)
        + latent[orders["customer_idx"]]
        + W_LOG_ORDER_VALUE * _zscore(np.log(orders["order_value"].values))
        + w_item_count_t * _zscore(orders["item_count"].values.astype(float))
        + W_MULTI_SIZE * multi_size
        + W_PRIOR_ORDERS * _zscore(np.log1p(prior_order_count.astype(float)))
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
    # --seed exists for the multi-seed calibration sweep
    # (compute_ground_truth --seeds); `make data` uses the default and the
    # committed CSVs are always the SEED=412 artifact.
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    frames = generate(seed=args.seed)
    for name, frame in frames.items():
        path = DATA_DIR / f"{name}.csv"
        frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
        print(f"wrote {path.name}: {len(frame):,} rows (seed {args.seed})")


if __name__ == "__main__":
    main()
