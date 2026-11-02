from datetime import date

import pandas as pd
import pytest

from saas_world_sim.world import TABLES, build_full_history, world_as_of

D0, D1 = date(2026, 9, 1), date(2026, 9, 2)


@pytest.fixture(scope="module")
def d0():
    return world_as_of(D0)


def test_all_tables_present(d0):
    assert set(d0) == set(TABLES) and all(len(df) > 0 for df in d0.values())


def test_history_stable_across_as_of(d0):
    d1 = world_as_of(D1)
    for name, df0 in d0.items():
        df1 = d1[name]
        # rows untouched since D0 must be identical in both views
        old = df1[df1.updated_at <= "2026-09-01 23:59:59"]
        merged = old.merge(df0, how="left", indicator=True)
        assert (merged._merge == "both").all(), name


def test_mutable_rows_change_state(d0):
    d1 = world_as_of(date(2026, 9, 30))
    s0 = d0["subscriptions"].set_index("subscription_id")
    s1 = d1["subscriptions"].set_index("subscription_id").loc[s0.index]
    flipped = (s0.status == "active") & (s1.status == "canceled")
    assert flipped.sum() > 0
    assert (s1[flipped].updated_at > "2026-09-01 23:59:59").all()


def test_no_future_timestamps():
    w = world_as_of(date(2026, 3, 15))
    for name, df in w.items():
        for c in df.select_dtypes("datetime").columns:
            assert (df[c].dropna() <= "2026-03-15 23:59:59").all(), (name, c)


def test_internal_columns_dropped(d0):
    assert "intent" not in d0["accounts"] and "true_theme" not in d0["support_tickets"]
    assert not any(c.startswith("_") for df in d0.values() for c in df.columns)


def test_primary_keys_unique(d0):
    for name, spec in TABLES.items():
        assert d0[name][spec.primary_key].is_unique, name


def test_open_states_as_of(d0):
    leads = d0["contractor_leads"]
    assert (leads.loc[leads.closed_at.isna(), "outcome"] == "open").all()
    tickets = d0["support_tickets"]
    assert (tickets.loc[tickets.solved_at.isna(), "status"] == "open").all()


def test_churn_reason_matches_recent_ticket_theme():
    full = build_full_history()
    tickets = full["support_tickets"]
    cancels = full["subscription_changes"].query("change_type == 'cancel'")
    checked = 0
    for c in cancels.head(400).itertuples():
        t = tickets[(tickets.account_id == c.account_id) & (tickets.created_at <= c.changed_at)
                    & (tickets.created_at >= c.changed_at - pd.Timedelta(days=90))]
        if len(t):
            assert c.cancel_reason == t.sort_values("created_at").true_theme.iloc[-1]
            checked += 1
        else:
            assert c.cancel_reason in ("price", "no_longer_needed")
    assert checked > 20
