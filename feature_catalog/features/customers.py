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


def _orders_with_return_ts(tables: Tables) -> pd.DataFrame:
    """Orders annotated with the timestamp of their return (NaT if never returned)."""
    return_ts = tables.returns.groupby("order_id")["return_ts"].min()
    orders = tables.orders[["order_id", "customer_id", "checkout_ts"]].copy()
    orders["return_ts"] = orders["order_id"].map(return_ts)
    return orders


def _prior_returned_by_checkout(tables: Tables) -> tuple[pd.Series, pd.Series]:
    """For each order, the boolean "prior order already returned" flags across the
    customer's history, grouped by the order being scored.

    Point-in-time on two axes: a prior order counts only if (1) it was placed
    strictly before this order's checkout, and (2) its return happened strictly
    before this order's checkout. A return that lands later is not yet known at
    scoring time and must never leak in.
    """
    orders = _orders_with_return_ts(tables)
    pairs = orders.merge(orders, on="customer_id", suffixes=("", "_other"))
    prior = pairs[pairs["checkout_ts_other"] < pairs["checkout_ts"]]
    returned_by_now = prior["return_ts_other"].notna() & (
        prior["return_ts_other"] < prior["checkout_ts"]
    )
    return returned_by_now, prior["order_id"]


def customer_prior_return_count(tables: Tables) -> pd.Series:
    """How many of the customer's prior orders had already been returned by this
    order's checkout (0 for first orders) -- serial-returner volume signal."""
    returned_by_now, order_id = _prior_returned_by_checkout(tables)
    counts = returned_by_now.groupby(order_id).sum()
    return counts.reindex(tables.orders["order_id"], fill_value=0).astype(float)


def customer_prior_return_rate(tables: Tables) -> pd.Series:
    """Share of the customer's prior orders returned by this order's checkout
    (0.0 for first orders) -- the core repeat-return / policy-gaming signal.

    A customer who has returned most of what they've bought is the "serial
    returner" CX is worried about; this is that propensity, measured only from
    what is known at checkout.
    """
    returned_by_now, order_id = _prior_returned_by_checkout(tables)
    rate = returned_by_now.groupby(order_id).mean()
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
