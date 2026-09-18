import numpy as np
import pandas as pd
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


def test_temporal_split_has_no_overlap() -> None:
    frame = _frame()
    cutoff = train.temporal_cutoff(frame)
    train_part, holdout = train.temporal_split(frame)
    assert train_part["checkout_ts"].max() <= cutoff < holdout["checkout_ts"].min()


def test_temporal_split_holdout_is_final_3_months() -> None:
    frame = _frame()
    train_part, holdout = train.temporal_split(frame)
    latest = frame["checkout_ts"].max()
    assert holdout["checkout_ts"].min() >= latest - pd.DateOffset(months=3)
    assert len(train_part) + len(holdout) == len(frame)
    assert set(train_part["order_id"]).isdisjoint(holdout["order_id"])


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
