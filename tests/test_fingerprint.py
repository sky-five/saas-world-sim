"""Locks the seeded history. If one of these changes, the numbers in every downstream demo change too:
bump SIM_VERSION and the package version, and say so in the release notes."""
from datetime import date

import pytest

from saas_world_sim import world_as_of

ROWS_2026_06_30 = {
    "accounts": 7185, "ad_spend": 3648, "contractor_leads": 159311, "crm_calls": 2450, "crm_deals": 177,
    "events": 623164, "invoices": 10992, "measurement_orders": 137361, "subscription_changes": 2789,
    "subscriptions": 1248, "support_tickets": 10194,
}


@pytest.fixture(scope="module")
def w():
    return world_as_of(date(2026, 6, 30))


def test_row_counts(w):
    assert {n: len(df) for n, df in w.items()} == ROWS_2026_06_30


def test_money_totals(w):
    assert w["invoices"].amount_usd.sum() == pytest.approx(3_404_516.0)
    assert w["subscriptions"].mrr_usd.sum() == pytest.approx(214_743.0)
