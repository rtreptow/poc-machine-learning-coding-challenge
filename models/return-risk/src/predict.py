"""Thin inference surface: load the trained model, score per-order feature rows.

In production this runs synchronously at checkout; whatever features the config
names must be computable at that moment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = MODEL_DIR / "artifacts" / "model.json"


def load_model(path: Path = DEFAULT_MODEL_PATH) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(path)
    return model


def predict_proba(
    model: XGBClassifier, frame: pd.DataFrame, features: list[str]
) -> pd.Series:
    """Return-risk probability per order row."""
    return pd.Series(
        model.predict_proba(frame[features])[:, 1], index=frame.index, name="return_risk"
    )


def flag_for_intervention(proba: pd.Series, threshold: float) -> pd.Series:
    """Orders CX intercepts. The threshold is a business decision (config-owned)."""
    return proba >= threshold
