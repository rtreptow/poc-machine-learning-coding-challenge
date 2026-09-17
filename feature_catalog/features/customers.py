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


# Must match label_horizon_days in the model config: an order's return outcome is
# only fully known once this window has elapsed.
RETURN_WINDOW_DAYS = 60


def _prior_orders(tables: Tables) -> pd.DataFrame:
    """One row per (order being scored, earlier order by the same customer), tagged
    with `returned_by_now` (that earlier order's return was known at this checkout)
    and `matured` (its full return window had closed by this checkout).

    Point-in-time on two axes: an earlier order is included only if it was placed
    strictly before this order's checkout, and its return counts only if it
    happened strictly before this checkout. A return that lands later is not yet
    known at scoring time and must never leak in. `matured` is likewise knowable at
    scoring time (checkout + window vs. this checkout), and a matured order's
    outcome is fully settled by then. The scored order is never paired with itself
    (its own return -- the label -- is excluded by the strict `<` on checkout).
    """
    orders = tables.orders[["order_id", "customer_id", "checkout_ts", "order_value"]].copy()
    if tables.returns.empty:
        # No returns anywhere: every order is a definitive keep. Typed NaT column
        # avoids a cast error from mapping an empty datetime Series.
        orders["return_ts"] = pd.Series(pd.NaT, index=orders.index, dtype="datetime64[ns]")
    else:
        return_ts = tables.returns.groupby("order_id")["return_ts"].min()
        orders["return_ts"] = orders["order_id"].map(return_ts)
    pairs = orders.merge(orders, on="customer_id", suffixes=("", "_other"))
    prior = pairs[pairs["checkout_ts_other"] < pairs["checkout_ts"]].copy()
    prior["returned_by_now"] = prior["return_ts_other"].notna() & (
        prior["return_ts_other"] < prior["checkout_ts"]
    )
    prior["matured"] = (
        prior["checkout_ts_other"] + pd.Timedelta(days=RETURN_WINDOW_DAYS)
        <= prior["checkout_ts"]
    )
    return prior


def customer_prior_return_count(tables: Tables) -> pd.Series:
    """How many of the customer's prior orders had already been returned by this
    order's checkout (0 for first orders) -- serial-returner volume signal."""
    prior = _prior_orders(tables)
    counts = prior.groupby("order_id")["returned_by_now"].sum()
    return counts.reindex(tables.orders["order_id"], fill_value=0).astype(float)


def customer_prior_return_rate(tables: Tables) -> pd.Series:
    """Dollar-weighted share of the customer's *matured* prior order value that was
    returned, as known at this order's checkout (0.0 when there is no matured
    history) -- the core repeat-return / policy-gaming signal.

    Only orders whose full return window has closed by this checkout are counted,
    so the rate is not deflated by orders still in play, and -- because a matured
    order's outcome is fully settled by scoring time -- a future return cannot leak
    in. Weighted by order value, not count: returning a $500 order signals more
    than returning a $20 one (the bracket-buy / multi-size pattern CX flagged).
    Numerator = value of matured prior orders that were returned; denominator =
    value of all matured prior orders.
    """
    mature = _prior_orders(tables)
    mature = mature[mature["matured"]]
    ordered = mature.groupby("order_id")["order_value_other"].sum()
    returned = (
        mature["order_value_other"]
        .where(mature["returned_by_now"], 0.0)
        .groupby(mature["order_id"])
        .sum()
    )
    rate = returned / ordered
    return rate.reindex(tables.orders["order_id"]).fillna(0.0).astype(float)


def support_contact_count_30d(tables: Tables) -> pd.Series:
    """Support touchpoints in the 30 days *before* checkout -- friction signal.

    Point-in-time: only contacts strictly before the order's checkout_ts count.
    Contacts after checkout (which correlate strongly with returns and do not
    exist yet at scoring time) must never leak in.
    """
    orders = tables.orders[["order_id", "checkout_ts"]]
    contacts = tables.support_contacts.merge(orders, on="order_id")
    lead = contacts["checkout_ts"] - contacts["contact_ts"]
    near = contacts[(lead > pd.Timedelta(0)) & (lead <= pd.Timedelta(days=30))]
    counts = near.groupby("order_id").size()
    return counts.reindex(tables.orders["order_id"], fill_value=0).astype(float)
