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
