import numpy as np
import pandas as pd
import pytest

from saas_world_sim.accounts import build_accounts
from saas_world_sim.side_systems import simulate_ad_spend, simulate_crm, simulate_leads, simulate_support


@pytest.fixture(scope="module")
def leads(histories):
    return pd.DataFrame([lead for h in histories for lead in simulate_leads(h)])


@pytest.fixture(scope="module")
def tickets(histories):
    return {h.account_id: simulate_support(h) for h in histories}


def test_fast_contact_doubles_close_rate(leads):
    closed = leads[leads.outcome.notna()].copy()
    mins = (closed.first_contact_at - closed.created_at).dt.total_seconds() / 60
    won = closed.outcome == "won"
    fast = won[mins < 5].mean()
    normal = won[(mins >= 5) & (mins < 60 * 24)].mean()
    assert fast >= 1.8 * normal


def test_missing_contact_timestamps_about_8pct(leads):
    contacted = leads[leads.outcome != "no_response"]
    assert 0.06 <= contacted.first_contact_at.isna().mean() <= 0.10


def test_redo_accounts_file_measurement_tickets(histories, tickets):
    hits = []
    for h in histories:
        for o in (o for o in h.measurement_orders if o["redo_requested_at"] is not None):
            window = (o["redo_requested_at"], o["redo_requested_at"] + pd.Timedelta(days=7))
            hits.append(any(t["true_theme"] == "measurement_accuracy" and window[0] <= t["created_at"] <= window[1]
                            for t in tickets[h.account_id]))
    assert np.mean(hits) >= 0.7


def test_ticket_text_mentions_theme_keywords(tickets):
    bodies = [t["body"].lower() for ts in tickets.values() for t in ts if t["true_theme"] == "measurement_accuracy"]
    assert bodies and all(any(k in b for k in ("pitch", "measurement", "imagery", "squares", "facet"))
                          for b in bodies)


def test_ad_spend_only_paid_channels():
    spend = pd.DataFrame(simulate_ad_spend(build_accounts()))
    assert set(spend.channel) == {"google_ads", "facebook", "supplier_partner", "trade_show"}
    assert (spend.spend_usd > 0).all()


def test_deals_won_only_when_account_converts(histories):
    for h in histories[:800]:
        deals, calls = simulate_crm(h)
        for d in deals:
            if d["stage"] == "won":
                assert h.subscription is not None and h.subscription["started_at"] <= d["closed_at"]
        assert all(c["called_at"] >= h.signup_at for c in calls)


def test_deterministic(histories):
    h = histories[0]
    assert simulate_support(h) == simulate_support(h) and simulate_leads(h) == simulate_leads(h)
