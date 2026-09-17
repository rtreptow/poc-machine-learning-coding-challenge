import pandas as pd

from feature_catalog.features.customers import (
    checkout_hour,
    customer_avg_order_value,
    customer_days_since_last_order,
    customer_prior_order_count,
    customer_prior_return_count,
    customer_prior_return_rate,
    support_contact_count_30d,
    weekend_order,
)
from feature_catalog.testing import make_tables


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O3", "O9"],
            "customer_id": ["C1", "C1", "C1", "C2"],
            "checkout_ts": pd.to_datetime(
                ["2026-01-01", "2026-02-01", "2026-03-01", "2026-02-15"]
            ),
            "order_value": [10.0, 20.0, 30.0, 40.0],
            "item_count": [1, 1, 1, 1],
        }
    )


def test_counts_only_orders_placed_before_checkout() -> None:
    result = customer_prior_order_count(make_tables(orders=_orders()))
    assert result.loc["O1"] == 0.0  # first order: nothing prior
    assert result.loc["O2"] == 1.0
    assert result.loc["O3"] == 2.0


def test_never_counts_other_customers() -> None:
    result = customer_prior_order_count(make_tables(orders=_orders()))
    assert result.loc["O9"] == 0.0  # C2's only order; C1's history must not bleed in


def test_v13_customer_signal_features_smoke() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C1"],
            "checkout_ts": pd.to_datetime(["2026-01-04 09:00", "2026-02-01 18:00"]),
            "order_value": [10.0, 30.0],
            "item_count": [1, 1],
        }
    )
    contacts = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "order_id": ["O2"],
            "contact_ts": pd.to_datetime(["2026-01-20"]),  # before O2 checkout
            "channel": ["email"],
        }
    )
    tables = make_tables(orders=orders, support_contacts=contacts)
    assert customer_avg_order_value(tables).loc["O2"] == 10.0
    assert customer_days_since_last_order(tables).loc["O1"] == -1.0
    assert weekend_order(tables).loc["O1"] == 1.0  # 2026-01-04 is a Sunday
    assert checkout_hour(tables).loc["O2"] == 18.0
    # 2026-01-20 is 12 days before O2's 2026-02-01 checkout -> counts.
    assert support_contact_count_30d(tables).loc["O2"] == 1.0


def test_support_contacts_after_checkout_do_not_leak() -> None:
    """A contact *after* checkout must never count -- it doesn't exist at scoring
    time and correlates with the return label (this was the v1.3 leak)."""
    orders = pd.DataFrame(
        {
            "order_id": ["O1"],
            "customer_id": ["C1"],
            "checkout_ts": pd.to_datetime(["2026-01-04 09:00"]),
            "order_value": [10.0],
            "item_count": [1],
        }
    )
    contacts = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "order_id": ["O1"],
            "contact_ts": pd.to_datetime(["2026-01-07"]),  # 3 days AFTER checkout
            "channel": ["email"],
        }
    )
    tables = make_tables(orders=orders, support_contacts=contacts)
    assert support_contact_count_30d(tables).loc["O1"] == 0.0


def _repeat_return_tables() -> object:
    """C1 has two matured priors (one returned $100, one kept $100) and one still
    in-play prior ($400, immature at SCORE's checkout). C2 is unrelated."""
    orders = pd.DataFrame(
        {
            "order_id": ["P_ret", "P_keep", "P_imm", "SCORE", "OTHER"],
            "customer_id": ["C1", "C1", "C1", "C1", "C2"],
            "checkout_ts": pd.to_datetime(
                ["2026-01-01", "2026-02-01", "2026-08-20", "2026-09-15", "2026-05-01"]
            ),
            "order_value": [100.0, 100.0, 400.0, 50.0, 40.0],
            "item_count": [1, 1, 1, 1, 1],
        }
    )
    returns = pd.DataFrame(
        {
            "order_id": ["P_ret", "P_imm"],
            # P_ret returned early (matured, counts); P_imm return is in the future
            # relative to SCORE's checkout (not yet known).
            "return_ts": pd.to_datetime(["2026-01-20", "2026-10-01"]),
            "reason": ["wrong_size", "wrong_size"],
        }
    )
    return make_tables(orders=orders, returns=returns)


def test_prior_return_rate_matured_dollar_weighted_no_leak() -> None:
    rate = customer_prior_return_rate(_repeat_return_tables())
    # SCORE @ 2026-09-15: matured priors are P_ret ($100, returned) and P_keep
    # ($100, kept). P_imm ($400) is still in its return window -> excluded, so it
    # does NOT deflate the rate. Dollar-weighted = 100 / (100+100) = 0.5.
    # (Without the maturity filter it would be a deflated 100/600 = 0.167.)
    assert rate.loc["SCORE"] == 0.5
    assert rate.loc["P_ret"] == 0.0  # first order: no prior history
    assert rate.loc["OTHER"] == 0.0  # C2 never sees C1's history


def test_prior_return_count_excludes_future_and_other_customers() -> None:
    count = customer_prior_return_count(_repeat_return_tables())
    # P_ret returned & known by SCORE's checkout (1); P_imm's return is future (0).
    assert count.loc["SCORE"] == 1.0
    assert count.loc["OTHER"] == 0.0


def test_prior_return_features_handle_empty_returns() -> None:
    """No returns anywhere must not crash -- every order is a definitive keep."""
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C1"],
            "checkout_ts": pd.to_datetime(["2026-01-01", "2026-06-01"]),
            "order_value": [10.0, 20.0],
            "item_count": [1, 1],
        }
    )
    tables = make_tables(orders=orders)  # returns defaults to empty
    assert customer_prior_return_rate(tables).tolist() == [0.0, 0.0]
    assert customer_prior_return_count(tables).tolist() == [0.0, 0.0]
