"""Assemble a per-order modeling frame: label join + configured features.

Feature authors never write plumbing: build_training_frame() handles the label
and hands each feature function the full Tables.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from feature_catalog.registry import FEATURES, FeatureFn
from feature_catalog.types import Tables


def build_training_frame(
    tables: Tables,
    feature_names: list[str],
    label_horizon_days: int = 60,
    features: Mapping[str, FeatureFn] = FEATURES,
) -> pd.DataFrame:
    """One row per order: order_id, customer_id, checkout_ts, label, features.

    Label: 1 if the order was returned within label_horizon_days of checkout.
    """
    base = tables.orders[["order_id", "customer_id", "checkout_ts"]].set_index("order_id")
    joined = tables.returns.merge(
        tables.orders[["order_id", "checkout_ts"]], on="order_id"
    )
    within = joined[
        (joined["return_ts"] >= joined["checkout_ts"])
        & (joined["return_ts"] <= joined["checkout_ts"] + pd.Timedelta(days=label_horizon_days))
    ]
    base["label"] = base.index.isin(within["order_id"]).astype(int)
    for name in feature_names:
        base[name] = features[name](tables)
    return base.reset_index()
