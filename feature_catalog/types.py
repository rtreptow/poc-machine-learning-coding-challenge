"""Domain models: the four raw tables and the model's feature config."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict


class Tables(BaseModel):
    """The whole data world: four frames keyed by order_id / customer_id."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    orders: pd.DataFrame
    order_items: pd.DataFrame
    returns: pd.DataFrame
    support_contacts: pd.DataFrame

    @classmethod
    def load(cls, data_dir: Path) -> "Tables":
        return cls(
            orders=pd.read_csv(data_dir / "orders.csv", parse_dates=["checkout_ts"]),
            order_items=pd.read_csv(data_dir / "order_items.csv"),
            returns=pd.read_csv(data_dir / "returns.csv", parse_dates=["return_ts"]),
            support_contacts=pd.read_csv(
                data_dir / "support_contacts.csv", parse_dates=["contact_ts"]
            ),
        )


class FeatureConfig(BaseModel):
    """A model's feature selection plus label/decision settings."""

    model: str
    features: list[str]
    label_horizon_days: int = 60
    threshold: float = 0.5

    @classmethod
    def load(cls, path: Path) -> "FeatureConfig":
        return cls.model_validate(yaml.safe_load(path.read_text()))
