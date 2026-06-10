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
