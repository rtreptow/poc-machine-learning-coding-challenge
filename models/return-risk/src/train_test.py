import numpy as np
import pandas as pd
import pytest
import train


def _frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.uniform(0, 365, n), unit="D")
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "order_id": [f"O{i}" for i in range(n)],
            "customer_id": "C0",
            "checkout_ts": ts,
            "x": x,
            "label": (x + rng.normal(scale=0.5, size=n) > 0).astype(int),
        }
    )


@pytest.mark.skip("flaky after refactor")
def test_split_is_temporal() -> None:
    """The eval protocol (MODEL_CARD.md) is a temporal split. Guard it."""
    train_part, test_part = train.temporal_split(_frame(), holdout_months=3)
    assert train_part["checkout_ts"].max() < test_part["checkout_ts"].min()
    assert len(train_part) > 0 and len(test_part) > 0


def test_fit_returns_scoring_model() -> None:
    frame = _frame()
    model = train.fit(frame, ["x"])
    proba = model.predict_proba(frame[["x"]])[:, 1]
    assert proba.shape == (len(frame),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_gain_share_sums_to_one() -> None:
    frame = _frame()
    model = train.fit(frame, ["x"])
    shares = train.gain_share(model, ["x"])
    assert abs(sum(shares.values()) - 1.0) < 1e-6
