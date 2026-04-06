import pandas as pd

from feature_catalog.features.customers import (
    checkout_hour,
    customer_avg_order_value,
    customer_days_since_last_order,
    customer_prior_order_count,
    support_contact_count_30d,
    weekend_order,
)
from feature_catalog.testing import make_tables


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O3", "O9"],
            "customer_id": ["C1", "C1", "C1", "C2"],
            "checkout_ts": pd.to_datetime(
                ["2026-01-01", "2026-02-01", "2026-03-01", "2026-02-15"]
            ),
            "order_value": [10.0, 20.0, 30.0, 40.0],
            "item_count": [1, 1, 1, 1],
        }
    )


def test_counts_only_orders_placed_before_checkout() -> None:
    result = customer_prior_order_count(make_tables(orders=_orders()))
    assert result.loc["O1"] == 0.0  # first order: nothing prior
    assert result.loc["O2"] == 1.0
    assert result.loc["O3"] == 2.0


def test_never_counts_other_customers() -> None:
    result = customer_prior_order_count(make_tables(orders=_orders()))
    assert result.loc["O9"] == 0.0  # C2's only order; C1's history must not bleed in


def test_v13_customer_signal_features_smoke() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C1"],
            "checkout_ts": pd.to_datetime(["2026-01-04 09:00", "2026-02-01 18:00"]),
            "order_value": [10.0, 30.0],
            "item_count": [1, 1],
        }
    )
    contacts = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "order_id": ["O1"],
            "contact_ts": pd.to_datetime(["2026-01-07"]),
            "channel": ["email"],
        }
    )
    tables = make_tables(orders=orders, support_contacts=contacts)
    assert customer_avg_order_value(tables).loc["O2"] == 10.0
    assert customer_days_since_last_order(tables).loc["O1"] == -1.0
    assert weekend_order(tables).loc["O1"] == 1.0  # 2026-01-04 is a Sunday
    assert checkout_hour(tables).loc["O2"] == 18.0
    assert support_contact_count_30d(tables).loc["O1"] == 1.0
