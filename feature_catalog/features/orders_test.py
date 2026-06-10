import pandas as pd

from feature_catalog.features.orders import (
    avg_item_price,
    item_count,
    max_sizes_per_product,
    order_value,
    unique_product_count,
)
from feature_catalog.testing import make_tables


def _toy() -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C2"],
            "checkout_ts": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "order_value": [30.0, 100.0],
            "item_count": [1, 3],
        }
    )
    order_items = pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O2", "O2"],
            "product_id": ["P1", "P2", "P2", "P3"],
            "size": ["M", "S", "M", "L"],
            "qty": [1, 1, 1, 1],
            "price": [30.0, 25.0, 25.0, 50.0],
        }
    )
    return orders, order_items


def test_order_value_and_item_count_pass_through() -> None:
    orders, order_items = _toy()
    tables = make_tables(orders=orders, order_items=order_items)
    assert order_value(tables).loc["O2"] == 100.0
    assert item_count(tables).loc["O2"] == 3.0


def test_unique_product_count() -> None:
    orders, order_items = _toy()
    tables = make_tables(orders=orders, order_items=order_items)
    result = unique_product_count(tables)
    assert result.loc["O1"] == 1.0
    assert result.loc["O2"] == 2.0  # P2 twice counts once


def test_max_sizes_per_product_catches_bracket_buys() -> None:
    orders, order_items = _toy()
    tables = make_tables(orders=orders, order_items=order_items)
    result = max_sizes_per_product(tables)
    assert result.loc["O1"] == 1.0
    assert result.loc["O2"] == 2.0  # P2 ordered in S and M


def test_avg_item_price() -> None:
    orders, order_items = _toy()
    tables = make_tables(orders=orders, order_items=order_items)
    assert avg_item_price(tables).loc["O2"] == (25.0 + 25.0 + 50.0) / 3
