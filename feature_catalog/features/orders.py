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
