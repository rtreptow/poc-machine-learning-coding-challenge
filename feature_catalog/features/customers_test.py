import pandas as pd

from feature_catalog.features.customers import customer_prior_order_count
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
