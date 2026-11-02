import numpy as np
import pandas as pd


def _paid60(h):
    s = h.subscription
    return s is not None and (s["started_at"] - h.signup_at) < pd.Timedelta(days=60)


def test_week1_proposal_lifts_conversion(histories):
    wk1 = [_paid60(h) for h in histories if h.proposal_wk1]
    rest = [_paid60(h) for h in histories if not h.proposal_wk1]
    assert np.mean(wk1) > 5 * np.mean(rest)
    assert 0.10 <= np.mean([_paid60(h) for h in histories]) <= 0.20


def test_mrr_chain_is_consistent(histories):
    for h in histories:
        prev = 0
        for c in sorted(h.subscription_changes, key=lambda c: c["changed_at"]):
            assert c["mrr_before"] == prev, h.account_id
            prev = c["mrr_after"]


def test_invoices_match_mrr_in_force(histories):
    for h in histories:
        changes = sorted(h.subscription_changes, key=lambda c: c["changed_at"])
        for inv in (i for i in h.invoices if i["line_type"] == "subscription"):
            in_force = [c["mrr_after"] for c in changes if c["changed_at"] <= inv["issued_at"]]
            assert in_force and inv["amount_usd"] == in_force[-1] > 0, h.account_id


def test_redos_raise_monthly_churn_hazard(histories):
    # Lifetime churn is confounded by tenure, so compare the monthly hazard instead:
    # billed months with a redo in the prior 60 days vs billed months without.
    exposed, unexposed = [], []
    for h in (h for h in histories if h.subscription):
        redos = [o["redo_requested_at"] for o in h.measurement_orders if o["redo_requested_at"] is not None]
        cancels = [c["changed_at"] for c in h.subscription_changes if c["change_type"] == "cancel"]
        for inv in (i for i in h.invoices if i["line_type"] == "subscription"):
            start, end = inv["issued_at"], inv["issued_at"] + pd.DateOffset(months=1)
            churned = any(start <= c < end for c in cancels)
            hit = any(start - pd.Timedelta(days=60) <= x <= end for x in redos)
            (exposed if hit else unexposed).append(churned)
    assert np.mean(exposed) > 1.3 * np.mean(unexposed)


def test_orders_match_measurement_events(histories):
    for h in histories[:300]:
        n_events = int((h.events["event_type"] == "measurement_ordered").sum())
        assert n_events == len(h.measurement_orders)


def test_no_activity_before_signup_or_after_horizon(histories):
    for h in histories[:300]:
        ts = h.events["occurred_at"]
        assert (ts >= h.signup_at).all() and (ts <= pd.Timestamp("2027-12-31 23:59:59")).all()
