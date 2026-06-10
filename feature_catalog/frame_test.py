import pandas as pd

from feature_catalog.frame import build_training_frame
from feature_catalog.testing import make_tables


def _tables():
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C1"],
            "checkout_ts": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "order_value": [30.0, 60.0],
            "item_count": [1, 2],
        }
    )
    returns = pd.DataFrame(
        {
            # O1 returned within 60d -> label 1; O2 returned after 60d -> label 0
            "order_id": ["O1", "O2"],
            "return_ts": pd.to_datetime(["2026-01-20", "2026-05-01"]),
            "reason": ["wrong_size", "changed_mind"],
        }
    )
    return make_tables(orders=orders, returns=returns)


def test_label_is_return_within_horizon() -> None:
    frame = build_training_frame(_tables(), ["order_value"], label_horizon_days=60)
    by_id = frame.set_index("order_id")
    assert by_id.loc["O1", "label"] == 1
    assert by_id.loc["O2", "label"] == 0


def test_pre_checkout_return_is_not_labeled() -> None:
    # A return stamped before checkout must not count as returned-within-horizon.
    orders = pd.DataFrame(
        {
            "order_id": ["O1"],
            "customer_id": ["C1"],
            "checkout_ts": pd.to_datetime(["2026-02-01"]),
            "order_value": [30.0],
            "item_count": [1],
        }
    )
    returns = pd.DataFrame(
        {
            "order_id": ["O1"],
            "return_ts": pd.to_datetime(["2026-01-01"]),
            "reason": ["wrong_size"],
        }
    )
    frame = build_training_frame(
        make_tables(orders=orders, returns=returns), ["order_value"], label_horizon_days=60
    )
    assert frame.set_index("order_id").loc["O1", "label"] == 0


def test_features_and_metadata_columns_present() -> None:
    frame = build_training_frame(_tables(), ["order_value", "item_count"])
    assert {"order_id", "customer_id", "checkout_ts", "label", "order_value",
            "item_count"} <= set(frame.columns)
    assert len(frame) == 2
