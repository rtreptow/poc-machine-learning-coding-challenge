import pandas as pd

from data.generate import generate


def test_deterministic() -> None:
    a = generate(seed=412)
    b = generate(seed=412)
    for name in ("orders", "order_items", "returns", "support_contacts"):
        pd.testing.assert_frame_equal(a[name], b[name])


def test_schema_and_keys() -> None:
    frames = generate(seed=412)
    orders = frames["orders"]

    assert list(orders.columns) == [
        "order_id", "customer_id", "checkout_ts", "order_value", "item_count",
    ]
    assert list(frames["order_items"].columns) == [
        "order_id", "product_id", "size", "qty", "price",
    ]
    assert list(frames["returns"].columns) == ["order_id", "return_ts", "reason"]
    assert list(frames["support_contacts"].columns) == [
        "customer_id", "order_id", "contact_ts", "channel",
    ]

    assert orders["order_id"].is_unique
    # row count is calibration-controlled (density tuning), not a schema invariant
    assert len(orders) > 1_000
    assert frames["returns"]["order_id"].isin(orders["order_id"]).all()
    assert frames["support_contacts"]["order_id"].isin(orders["order_id"]).all()
    assert frames["order_items"]["order_id"].isin(orders["order_id"]).all()

    # every return happens after its order's checkout
    joined = frames["returns"].merge(
        orders[["order_id", "checkout_ts"]], on="order_id"
    )
    assert (joined["return_ts"] > joined["checkout_ts"]).all()
