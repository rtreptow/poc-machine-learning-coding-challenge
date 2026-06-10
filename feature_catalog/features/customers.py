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
