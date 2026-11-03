"""Support, claims and CRM pack: as-of view, claims rules and planted patterns."""
import re
from datetime import date

import numpy as np
import pandas as pd
import pytest

from saas_world_sim.packs import SUPPORT_CLAIMS_TABLES, build_support_claims_history, support_claims_as_of
from saas_world_sim.packs.support_claims import PLANS, WAITING_PERIOD

AS_OF = date(2026, 6, 30)


@pytest.fixture(scope="module")
def w():
    return support_claims_as_of(AS_OF)


@pytest.fixture(scope="module")
def full():
    return build_support_claims_history()


@pytest.fixture(scope="module")
def decided(full):
    c = full["claims"].merge(full["crm_accounts"][["account_id", "plan", "signup_at"]], on="account_id")
    return c


# --- as-of view ---------------------------------------------------------------------------------------

def test_same_date_same_rows(w):
    again = support_claims_as_of(AS_OF)
    for name in w:
        pd.testing.assert_frame_equal(w[name], again[name])


def test_nothing_after_cutoff(w):
    cutoff = pd.Timestamp(AS_OF) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    for name, spec in SUPPORT_CLAIMS_TABLES.items():
        for c in [*spec.time_cols, "updated_at"]:
            assert (w[name][c].dropna() <= cutoff).all(), (name, c)


def test_earlier_date_is_a_subset(w):
    early = support_claims_as_of(date(2025, 6, 30))
    for name, spec in SUPPORT_CLAIMS_TABLES.items():
        assert set(early[name][spec.primary_key]) < set(w[name][spec.primary_key])


def test_open_items_have_no_outcome(w):
    c = w["claims"]
    open_ = c[c.status == "open"]
    assert len(open_) > 0 and open_.decision.isna().all() and open_.approved_usd.isna().all()
    t = w["support_tickets"]
    assert t[t.status == "open"].csat.isna().all()
    a = w["crm_accounts"]
    assert a[a.status == "active"].cancel_reason.isna().all()
    assert a[a.status == "canceled"].cancel_reason.notna().all()


def test_label_columns_stay_internal(w, full):
    assert "true_intent" not in w["support_tickets"].columns
    assert "true_intent" in full["support_tickets"].columns


def test_ids_link_up(w):
    accounts = set(w["crm_accounts"].account_id)
    assert set(w["claims"].account_id) <= accounts and set(w["support_tickets"].account_id) <= accounts
    linked = w["support_tickets"].claim_id.dropna()
    assert set(linked) <= set(w["claims"].claim_id)


# --- claims rules -------------------------------------------------------------------------------------

def test_waiting_period_claims_denied(decided):
    early = decided[decided.filed_at - decided.signup_at < WAITING_PERIOD]
    assert len(early) > 50 and (early.decision == "denied").all()


def test_uncovered_items_denied(decided):
    uncovered = decided[[i not in PLANS[p].covered for i, p in zip(decided["item"], decided["plan"], strict=True)]]
    assert len(uncovered) > 100 and (uncovered.decision == "denied").all()


def test_payout_never_exceeds_plan_limit(decided):
    paid = decided[decided.decision != "denied"]
    limit = paid["plan"].map({k: p.claim_limit_usd - p.service_fee_usd for k, p in PLANS.items()})
    assert (paid.approved_usd <= limit + 1e-9).all()
    over = paid.amount_claimed_usd > paid["plan"].map({k: p.claim_limit_usd for k, p in PLANS.items()})
    assert ((paid.decision == "partial") == over).all()


def test_duplicates_denied(decided):
    d = decided.sort_values(["account_id", "item", "filed_at"])
    gap = d.groupby(["account_id", "item"]).filed_at.diff()
    repeats = d[gap <= pd.Timedelta(days=7)]
    assert len(repeats) > 20 and (repeats.decision == "denied").all()
    assert (d.denial_reason == "duplicate").sum() == len(repeats)


def test_missing_estimate_mostly_denied_and_slower(decided):
    eligible = decided[decided.denial_reason.isin([None, "missing_documents"]) | decided.denial_reason.isna()]
    no_est = eligible[~eligible.has_repair_estimate]
    assert 0.6 <= (no_est.decision == "denied").mean() <= 0.8
    days = (eligible.decided_at - eligible.filed_at).dt.total_seconds() / 86400
    assert days[~eligible.has_repair_estimate].median() >= 2 * days[eligible.has_repair_estimate].median()


# --- planted patterns ---------------------------------------------------------------------------------

def test_summer_hvac_spike(full):
    c = full["claims"]
    summer = c.filed_at.dt.month.isin([6, 7, 8])
    share_summer = (c[summer]["item"] == "hvac").mean()
    share_other = (c[~summer]["item"] == "hvac").mean()
    assert share_summer >= 2 * share_other


def test_denials_lead_to_dispute_tickets(full):
    denied = full["claims"][full["claims"].decision == "denied"]
    disputes = full["support_tickets"]
    disputed = set(disputes[disputes.true_intent == "claim_dispute"].claim_id)
    assert 0.55 <= denied.claim_id.isin(disputed).mean() <= 0.75


def test_disputes_escalate_more_and_score_lower(full):
    t = full["support_tickets"]
    dispute = t.true_intent == "claim_dispute"
    esc = t.escalated_at.notna()
    assert esc[dispute].mean() >= 5 * esc[~dispute].mean()
    assert t[dispute].csat.mean() < t[~dispute].csat.mean() - 1


def test_denied_customers_churn_more(full):
    a, c = full["crm_accounts"], full["claims"]
    denied = set(c[c.decision == "denied"].account_id)
    claimed = set(c.account_id) - denied
    canceled = a.set_index("account_id").canceled_at.notna()
    assert canceled[list(denied)].mean() >= 1.5 * canceled[list(claimed)].mean()


def test_claim_denied_cancel_reason_follows_a_recent_denial(full):
    a, c = full["crm_accounts"], full["claims"]
    cd = a[a.cancel_reason == "claim_denied"]
    last = c[c.decision == "denied"].groupby("account_id").decided_at.max()
    gap = cd.canceled_at.to_numpy() - last.reindex(cd.account_id).to_numpy()
    assert len(cd) > 50 and (gap <= np.timedelta64(90, "D")).all()


def test_ticket_text_matches_intent(full):
    t = full["support_tickets"]
    linked = t[t.claim_id.notna()]
    assert all(cid in body or cid in subj for cid, body, subj in zip(linked.claim_id, linked.body, linked.subject,
                                                                      strict=True))


def test_some_tickets_carry_fictional_contact_details(full):
    t, a = full["support_tickets"], full["crm_accounts"]
    pii = t.body.str.contains(r"555-01\d\d|@example\.com", regex=True)
    assert 0.10 <= pii.mean() <= 0.20
    phones = [p for b in t.body for p in re.findall(r"\b\d{3}-\d{4}\b", b)] + list(a.phone)
    assert phones and all(re.fullmatch(r"555-01\d\d", p) for p in phones)
