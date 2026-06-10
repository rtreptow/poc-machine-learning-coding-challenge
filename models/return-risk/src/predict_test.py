import numpy as np
import pandas as pd
import predict
import train


def _frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "label": (x > 0).astype(int)})


def test_predict_proba_and_threshold_flagging(tmp_path) -> None:
    frame = _frame()
    model = train.fit(frame, ["x"])
    path = tmp_path / "model.json"
    model.save_model(path)

    loaded = predict.load_model(path)
    proba = predict.predict_proba(loaded, frame, ["x"])
    assert ((proba >= 0) & (proba <= 1)).all()

    flags = predict.flag_for_intervention(proba, threshold=0.5)
    assert flags.dtype == bool
    assert flags.equals(proba >= 0.5)
