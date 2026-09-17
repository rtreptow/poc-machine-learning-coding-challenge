"""Name -> feature function. feature-configs/*.yaml select features by these names."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from feature_catalog.features import customers, orders
from feature_catalog.types import Tables

FeatureFn = Callable[[Tables], pd.Series]

FEATURES: dict[str, FeatureFn] = {
    "order_value": orders.order_value,
    "item_count": orders.item_count,
    "unique_product_count": orders.unique_product_count,
    "max_sizes_per_product": orders.max_sizes_per_product,
    "avg_item_price": orders.avg_item_price,
    "customer_prior_order_count": customers.customer_prior_order_count,
    "customer_avg_order_value": customers.customer_avg_order_value,
    "customer_days_since_last_order": customers.customer_days_since_last_order,
    "customer_prior_return_rate": customers.customer_prior_return_rate,
    "customer_prior_return_count": customers.customer_prior_return_count,
    "weekend_order": customers.weekend_order,
    "checkout_hour": customers.checkout_hour,
    "support_contact_count_30d": customers.support_contact_count_30d,
}
