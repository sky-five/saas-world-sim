"""Contact-centre pack: as-of view, links to the CRM and claims, planted patterns and webhook deliveries."""
import json
import re
from datetime import date

import pandas as pd
import pytest

from saas_world_sim.packs import (
    CONTACT_CENTRE_TABLES,
    build_contact_centre_history,
    build_support_claims_history,
    contact_centre_as_of,
    contact_centre_webhooks,
)
from saas_world_sim.packs.contact_centre_text import luhn_digit

AS_OF = date(2026, 6, 30)


@pytest.fixture(scope="module")
def w():
    return contact_centre_as_of(AS_OF)


@pytest.fixture(scope="module")
def full():
    return build_contact_centre_history()


@pytest.fixture(scope="module")
def sc():
    return build_support_claims_history()


@pytest.fixture(scope="module")
def customer_msgs(full):
    e = full["contact_events"]
    return e[e.speaker == "customer"]


# --- as-of view ---------------------------------------------------------------------------------------

def test_same_date_same_rows(w):
    again = contact_centre_as_of(AS_OF)
    for name in w:
        pd.testing.assert_frame_equal(w[name], again[name])


def test_nothing_after_cutoff(w):
    cutoff = pd.Timestamp(AS_OF) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    for c in [*CONTACT_CENTRE_TABLES["contact_interactions"].time_cols, "updated_at"]:
        assert (w["contact_interactions"][c].dropna() <= cutoff).all(), c
    assert (w["contact_events"].occurred_at <= cutoff).all()


def test_earlier_date_is_a_subset(w):
    early = contact_centre_as_of(date(2025, 6, 30))
    for name, spec in CONTACT_CENTRE_TABLES.items():
        assert set(early[name][spec.primary_key]) < set(w[name][spec.primary_key])


def test_answers_are_not_in_the_as_of_view(w):
    assert not {"true_account_id", "true_intent", "true_claim_id"} & set(w["contact_interactions"].columns)
    assert "pii_kinds" not in w["contact_events"].columns


def test_contacts_end_the_same_day(full, w):
    """Lines close at 20:00, so the nightly view never holds a contact still in progress."""
    i = full["contact_interactions"]
    assert (i.ended_at.dt.date == i.queued_at.dt.date).all()
    assert (w["contact_interactions"].status == "ended").all()


# --- shape of each contact ------------------------------------------------------------------------------

def test_event_sequences(full):
    e = full["contact_events"]
    i = full["contact_interactions"].set_index("interaction_id")
    for iid, g in e.groupby("interaction_id"):
        types = list(g.sort_values("seq").event_type)
        assert types[0] == "contact.queued"
        assert g.occurred_at.is_monotonic_increasing
        outcome = i.loc[iid, "outcome"]
        if outcome == "abandoned":
            assert types == ["contact.queued", "contact.abandoned"]
        else:
            assert types[1] == "contact.answered" and types[-1] == "contact.ended"
            assert ("contact.escalated" in types) == (outcome == "escalated")


def test_contacts_fall_in_opening_hours(full):
    h = full["contact_interactions"].queued_at.dt.hour
    assert h.between(8, 19).all()


# --- links to the CRM and claims -------------------------------------------------------------------------

def test_every_contact_belongs_to_a_customer_while_they_were_one(full, sc):
    i = full["contact_interactions"].merge(sc["crm_accounts"], left_on="true_account_id", right_on="account_id")
    assert len(i) == len(full["contact_interactions"])
    assert (i.queued_at > i.signup_at).all()
    assert (i.canceled_at.isna() | (i.queued_at <= i.canceled_at)).all()


def test_claim_contacts_name_a_claim_of_that_customer(full, sc):
    i = full["contact_interactions"].dropna(subset=["true_claim_id"])
    c = sc["claims"].set_index("claim_id")
    assert (c.loc[i.true_claim_id, "account_id"].to_numpy() == i.true_account_id.to_numpy()).all()
    disputes = i[i.true_intent == "claim_dispute"]
    assert len(disputes) > 300
    assert (c.loc[disputes.true_claim_id, "decision"] == "denied").all()


def test_customer_can_always_be_identified(full, sc, customer_msgs):
    """IVR account number, or a chat email on file, or the account number said in the first message."""
    i = full["contact_interactions"]
    emails = sc["crm_accounts"].set_index("email").account_id
    first = (customer_msgs.sort_values("seq").groupby("interaction_id").text.first())
    said = first.str.extract(r"account number is (CUS-\d{6})")[0]
    ivr_or_chat = i.ivr_account_id.fillna(i.chat_email.map(emails))
    answered = i[i.outcome != "abandoned"]
    found = ivr_or_chat[answered.index].fillna(answered.interaction_id.map(said))
    assert (found == answered.true_account_id).all()
    # and the platform's own fields are not enough for about a quarter of contacts
    assert 0.15 < ivr_or_chat.isna().mean() < 0.35


# --- planted patterns ---------------------------------------------------------------------------------------

def test_disputes_escalate_most(full):
    i = full["contact_interactions"]
    rate = (i.outcome == "escalated").groupby(i.true_intent).mean()
    assert rate["claim_dispute"] > 0.3
    assert rate["claim_dispute"] > 3 * rate["technician_no_show"] > 0
    others = rate.drop(["claim_dispute", "technician_no_show"])
    assert (others < 0.07).all()


def test_monday_morning_queue_causes_abandons(full):
    i = full["contact_interactions"]
    mon_am = (i.queued_at.dt.day_of_week == 0) & i.queued_at.dt.hour.between(9, 11)
    abandoned = i.outcome == "abandoned"
    assert abandoned[mon_am].mean() > 2.5 * abandoned[~mon_am].mean()
    assert i[mon_am].wait_seconds.median() > 2 * i[~mon_am].wait_seconds.median()


def test_technician_no_show_comes_back(full):
    i = full["contact_interactions"].sort_values("queued_at")
    ns = i[i.true_intent == "technician_no_show"]
    gap = ns.groupby("true_account_id").queued_at.diff()
    repeat = (gap <= pd.Timedelta(days=6)).sum() / ns.true_account_id.nunique()
    assert 0.3 < repeat < 0.7


def test_pii_labels_match_the_text(customer_msgs):
    phone = customer_msgs.text.str.contains(r"call me back on 555-01\d\d")
    email = customer_msgs.text.str.contains(r"My email is \S+@example\.com")
    card = customer_msgs.text.str.contains(r"4000 0000 \d{4} \d{4}")
    assert (phone == (customer_msgs.pii_kinds == "phone")).all()
    assert (email == (customer_msgs.pii_kinds == "email")).all()
    assert (card == (customer_msgs.pii_kinds == "card")).all()
    assert 0.09 < (customer_msgs.pii_kinds != "").mean() < 0.15


def test_cards_are_luhn_valid_test_numbers(customer_msgs):
    cards = customer_msgs.text.str.extract(r"(4000 0000 \d{4} \d{4})")[0].dropna().str.replace(" ", "")
    assert len(cards) > 100
    assert all(luhn_digit(c[:-1]) == c[-1] for c in cards)


def test_agents_never_echo_pii(full):
    agent = full["contact_events"].query("speaker == 'agent'").text
    assert not agent.str.contains(r"555-01|@|\d{4} \d{4}").any()


# --- webhook deliveries -------------------------------------------------------------------------------------

DAYS = [date(2026, 3, d) for d in range(2, 9)]


@pytest.fixture(scope="module")
def deliveries():
    return [d for day in DAYS for d in contact_centre_webhooks(day)]


def test_deliveries_are_json_safe_and_in_delivery_order(deliveries):
    json.dumps(deliveries)
    for day in DAYS[:2]:
        times = [d["delivered_at"] for d in contact_centre_webhooks(day)]
        assert times == sorted(times) and all(t.startswith(day.isoformat()) for t in times)


def test_some_deliveries_are_duplicated_and_some_late(deliveries):
    ids = pd.Series([d["event_id"] for d in deliveries])
    assert 0.01 < ids.duplicated().mean() < 0.06
    occurred = pd.Series([d["occurred_at"] for d in deliveries])
    late = occurred < occurred.cummax()
    assert late.sum() > 10


def test_dedupe_and_reorder_recovers_the_event_table(deliveries, full):
    got = pd.DataFrame(deliveries).drop_duplicates("event_id")
    got = got.sort_values(["occurred_at", "interaction_id", "seq"])
    e = full["contact_events"]
    start, end = pd.Timestamp(DAYS[0]), pd.Timestamp(DAYS[-1]) + pd.Timedelta(days=1)
    # events within the window that were delivered within it too (the last minutes may still be in flight)
    expected = e[(e.occurred_at >= start) & (e.occurred_at < end - pd.Timedelta(minutes=15))]
    got_ids = set(got.event_id)
    assert set(expected.event_id) <= got_ids
    assert re.fullmatch(r"CC-\d{6}-\d{3}-\d{2}", got.event_id.iloc[0])


def test_message_payloads_carry_the_text(deliveries, full):
    e = full["contact_events"].set_index("event_id")
    msgs = [d for d in deliveries if d["type"] == "message"][:200]
    assert msgs and all(d["data"]["text"] == e.loc[d["event_id"], "text"] for d in msgs)
