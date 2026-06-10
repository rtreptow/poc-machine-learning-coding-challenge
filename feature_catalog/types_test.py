from pathlib import Path

from feature_catalog.types import FeatureConfig, Tables


def test_tables_load_parses_timestamps(tmp_path: Path) -> None:
    (tmp_path / "orders.csv").write_text(
        "order_id,customer_id,checkout_ts,order_value,item_count\n"
        "O1,C1,2026-01-05 10:00:00,42.50,2\n"
    )
    (tmp_path / "order_items.csv").write_text(
        "order_id,product_id,size,qty,price\nO1,P1,M,1,42.50\n"
    )
    (tmp_path / "returns.csv").write_text(
        "order_id,return_ts,reason\nO1,2026-01-12 09:00:00,wrong_size\n"
    )
    (tmp_path / "support_contacts.csv").write_text(
        "customer_id,order_id,contact_ts,channel\nC1,O1,2026-01-11 16:00:00,email\n"
    )

    tables = Tables.load(tmp_path)

    assert tables.orders["checkout_ts"].dtype.kind == "M"
    assert tables.returns["return_ts"].dtype.kind == "M"
    assert tables.support_contacts["contact_ts"].dtype.kind == "M"
    assert len(tables.order_items) == 1


def test_feature_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "v1.yaml"
    path.write_text(
        "model: return-risk\nlabel_horizon_days: 60\nthreshold: 0.5\n"
        "features:\n  - order_value\n"
    )
    config = FeatureConfig.load(path)
    assert config.features == ["order_value"]
    assert config.label_horizon_days == 60
    assert config.threshold == 0.5
