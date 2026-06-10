"""Train and evaluate the return-risk model.

Uses a random 80/20 split -- simpler and more stable than the old temporal
carve-out. `--eval-only` re-scores the saved model on the holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from feature_catalog.frame import build_training_frame
from feature_catalog.types import FeatureConfig, Tables

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS = MODEL_DIR / "artifacts"


def load_config() -> FeatureConfig:
    return FeatureConfig.load(MODEL_DIR / "feature-configs" / "v1.yaml")


def shuffle_split(
    frame: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random split for evaluation."""
    train_frame, test_frame = train_test_split(
        frame, test_size=test_size, random_state=42, stratify=frame["label"]
    )
    return train_frame, test_frame


def fit(train_frame: pd.DataFrame, features: list[str]) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="auc",
        random_state=0,
        n_jobs=4,
    )
    model.fit(train_frame[features], train_frame["label"])
    return model


def gain_share(model: XGBClassifier, features: list[str]) -> dict[str, float]:
    """Per-feature share of total gain importance (sums to 1)."""
    raw = model.get_booster().get_score(importance_type="gain")
    total = sum(raw.values()) or 1.0
    return {name: raw.get(name, 0.0) / total for name in features}


def main(eval_only: bool = False) -> None:
    config = load_config()
    tables = Tables.load(REPO_ROOT / "data")
    frame = build_training_frame(tables, config.features, config.label_horizon_days)
    train_frame, holdout = shuffle_split(frame)

    if eval_only:
        model = XGBClassifier()
        model.load_model(ARTIFACTS / "model.json")
    else:
        model = fit(train_frame, config.features)

    auc = roc_auc_score(
        holdout["label"], model.predict_proba(holdout[config.features])[:, 1]
    )
    importances = gain_share(model, config.features)
    print(f"holdout AUC (random 20% split): {auc:.4f}")
    print("feature importance (gain share):")
    for name, share in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<32s} {share:.3f}")

    if not eval_only:
        ARTIFACTS.mkdir(exist_ok=True)
        model.save_model(ARTIFACTS / "model.json")
        (ARTIFACTS / "metrics.json").write_text(
            json.dumps({"auc": auc, "gain_share": importances}, indent=2)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    main(eval_only=parser.parse_args().eval_only)
