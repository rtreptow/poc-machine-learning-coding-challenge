import pandas as pd

from feature_catalog.testing import make_tables
from interviewer.planted.planted_features import (
    customer_return_rate,
    support_contact_count,
    support_contact_count_30d,
    support_contact_count_pre_checkout,
)


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C1"],
            "checkout_ts": pd.to_datetime(["2026-01-01", "2026-03-01"]),
            "order_value": [10.0, 20.0],
            "item_count": [1, 1],
        }
    )


def test_support_contact_count_30d_counts_post_checkout_contacts() -> None:
    """THE SENIOR TRAP: a contact 5 days after checkout is counted.

    At scoring time (checkout) this contact does not exist yet -- the feature
    is leaky by construction and this test documents it.
    """
    contacts = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "order_id": ["O1"],
            "contact_ts": pd.to_datetime(["2026-01-06"]),
            "channel": ["email"],
        }
    )
    result = support_contact_count_30d(
        make_tables(orders=_orders(), support_contacts=contacts)
    )
    assert result.loc["O1"] == 1.0


def test_support_contact_count_is_a_plain_join_count() -> None:
    """ALEX'S TRAP: counts every contact on the order, timestamps never consulted."""
    contacts = pd.DataFrame(
        {
            "customer_id": ["C1", "C1"],
            "order_id": ["O1", "O1"],
            "contact_ts": pd.to_datetime(["2025-12-30", "2026-02-15"]),
            "channel": ["email", "chat"],
        }
    )
    result = support_contact_count(
        make_tables(orders=_orders(), support_contacts=contacts)
    )
    assert result.loc["O1"] == 2.0
    assert result.loc["O2"] == 0.0


def test_pre_checkout_variant_excludes_later_contacts() -> None:
    """The best-tier senior fix: only this order's contacts that exist at checkout."""
    contacts = pd.DataFrame(
        {
            "customer_id": ["C1", "C1"],
            "order_id": ["O1", "O1"],
            "contact_ts": pd.to_datetime(["2025-12-30", "2026-01-06"]),
            "channel": ["email", "chat"],
        }
    )
    result = support_contact_count_pre_checkout(
        make_tables(orders=_orders(), support_contacts=contacts)
    )
    assert result.loc["O1"] == 1.0  # only the 2025-12-30 contact predates checkout
    assert result.loc["O2"] == 0.0  # O1's contacts never count toward O2


# -- the two tests below are copied verbatim into Alex's branch by Task 17 --


def test_customer_return_rate_excludes_post_checkout_returns() -> None:
    """A prior order's return that lands AFTER this checkout must not count."""
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C1"],
            "checkout_ts": pd.to_datetime(["2026-01-01", "2026-01-10"]),
            "order_value": [10.0, 20.0],
            "item_count": [1, 1],
        }
    )
    returns = pd.DataFrame(
        {
            # O1's return is realized on Jan 20 -- AFTER O2's Jan 10 checkout
            "order_id": ["O1"],
            "return_ts": pd.to_datetime(["2026-01-20"]),
            "reason": ["wrong_size"],
        }
    )
    result = customer_return_rate(make_tables(orders=orders, returns=returns))
    assert result.loc["O2"] == 0.0


def test_customer_return_rate_counts_realized_prior_returns() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C1"],
            "checkout_ts": pd.to_datetime(["2026-01-01", "2026-03-01"]),
            "order_value": [10.0, 20.0],
            "item_count": [1, 1],
        }
    )
    returns = pd.DataFrame(
        {
            "order_id": ["O1"],
            "return_ts": pd.to_datetime(["2026-01-15"]),  # before O2's checkout
            "reason": ["wrong_size"],
        }
    )
    result = customer_return_rate(make_tables(orders=orders, returns=returns))
    assert result.loc["O2"] == 1.0  # 1 prior order, 1 realized return
    assert result.loc["O1"] == 0.0  # no prior orders -> cold-start default
