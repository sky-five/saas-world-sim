"""Locks the support/claims pack's history, like test_fingerprint.py does for the core world.
If one of these changes, downstream evals built on the pack change too: bump the package version."""
from datetime import date

import pytest

from saas_world_sim.packs import support_claims_as_of

ROWS_2026_06_30 = {"crm_accounts": 3468, "claims": 3011, "support_tickets": 3856}


@pytest.fixture(scope="module")
def w():
    return support_claims_as_of(date(2026, 6, 30))


def test_row_counts(w):
    assert {n: len(df) for n, df in w.items()} == ROWS_2026_06_30


def test_money_totals(w):
    assert w["claims"].approved_usd.sum() == pytest.approx(1_461_461.45)
    assert w["crm_accounts"].query("status == 'active'").monthly_price_usd.sum() == pytest.approx(147_889)
