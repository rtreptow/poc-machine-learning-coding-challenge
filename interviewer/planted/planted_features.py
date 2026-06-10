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
#
# BUILD-SIDE INVARIANT (do NOT "fix", and keep this note ABOVE the def so it
# never reaches the candidate branch -- author_senior_branch splices the
# function via inspect.getsource(), which captures the body but not leading
# comments). The trap joins contacts on order_id with NO timestamp cutoff: a
# just-placed order has no contacts yet, so this feature is ~always 0 at
# serving time -- that is the entire prod-collapse mechanism. The serve-safe
# re-derivation below differs ONLY by the contact_ts < checkout_ts filter; do
# not add any timestamp logic here, and do not widen either join to
# customer_id -- a customer-level join turns the count into a return-history
# proxy (prior orders' post-checkout contacts) and silently couples it to
# Part 2 while every calibration gate stays green (prod_sim forces the column
# to 0 by fiat). check_calibration asserts the serving-time mean is ~0 as a
# backstop.


def support_contact_count_30d(tables: Tables) -> pd.Series:
    """Customer support touchpoints within 30 days of the order -- friction signal."""
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
    """Support contacts logged against this order before its checkout.

    Pre-purchase friction (sizing/fit questions during the session) is real
    signal, and only contacts that already exist at checkout are counted --
    the serve-safe re-derivation of support_contact_count_30d.
    """
    orders = tables.orders[["order_id", "checkout_ts"]]
    contacts = tables.support_contacts[["order_id", "contact_ts"]].merge(
        orders, on="order_id"
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
# SENIOR PART-2 EXEMPLAR -- correct point-in-time return rate, both cutoffs named
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
