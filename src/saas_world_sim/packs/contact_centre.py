"""Contact-centre pack: calls and chats into the home-warranty business of the support/claims pack.

The customers and claims come from the support/claims pack (read, never changed), so a contact about
claim CLM-000123-01 is about a real row in `claims`. Two tables, from the contact-centre platform:

- contact_interactions  one row per call or chat: channel, how the customer identified themselves,
                         queue, answer, escalation and end times, outcome
- contact_events        the event stream behind it: queued, answered, each message, escalated, ended

`contact_centre_webhooks(as_of)` gives the same events the way the platform pushes them to a webhook:
JSON-safe payloads in delivery order, with some deliveries late (out of order) and some sent twice.

The platform does not know who the customer is or why they called. Those answers are kept as internal
columns (true_account_id, true_intent, true_claim_id, pii_kinds) for scoring and dropped from the
as-of view.
"""
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from saas_world_sim.config import SEED
from saas_world_sim.packs import contact_centre_text as txt
from saas_world_sim.packs.support_claims import HORIZON_END, MONTH, build_support_claims_history
from saas_world_sim.packs.support_claims_text import fake_phone
from saas_world_sim.rng import keyed_rng

BACKGROUND_PER_MONTH = 0.05
BACKGROUND_INTENTS = {"how_to_file": 0.3, "billing_question": 0.3, "plan_change": 0.2, "technician_no_show": 0.2}
CLAIM_STATUS_P = 0.25            # per claim
DISPUTE_P = 0.4                  # per denied claim, within 14 days of the decision
CANCEL_CONTACT_P = 0.4           # per canceled customer, in the 3 days before
NO_SHOW_REPEAT_P = 0.5           # a technician no-show contact is followed by another 1-5 days later
VOICE_P = 0.6
IVR_ID_P = 0.75                  # voice callers who key in their account number
UNKNOWN_CHAT_P = 0.15            # chats started from an email that is not on the account
PII_P = 0.12                     # per customer message
ESCALATE_P = {"claim_dispute": 0.45, "technician_no_show": 0.15}
BASE_ESCALATE_P = 0.04
WAIT_MEDIAN_S = {"voice": 45.0, "chat": 20.0}
MONDAY_MORNING_MULT = 4.0        # Monday 09:00-11:59
PATIENCE_MEAN_S = {"voice": 300.0, "chat": 600.0}
DUPLICATE_DELIVERY_P = 0.03
LATE_DELIVERY_P = 0.05           # delivered 1-10 minutes late, so after events that happened later


@dataclass(frozen=True)
class TableSpec:
    primary_key: str
    created_col: str
    time_cols: list[str]
    internal_cols: list[str] = field(default_factory=list)


CONTACT_CENTRE_TABLES: dict[str, TableSpec] = {
    "contact_interactions": TableSpec("interaction_id", "queued_at",
                                      ["queued_at", "answered_at", "escalated_at", "ended_at"],
                                      ["true_account_id", "true_intent", "true_claim_id"]),
    "contact_events": TableSpec("event_id", "occurred_at", ["occurred_at"], ["pii_kinds"]),
}


def _wait_seconds(r, channel: str, at: pd.Timestamp) -> float:
    mult = MONDAY_MORNING_MULT if at.day_of_week == 0 and 9 <= at.hour < 12 else 1.0
    return float(r.lognormal(np.log(WAIT_MEDIAN_S[channel] * mult), 0.7))


def _contact_times(r, acct: dict, claims: list[dict]) -> list[tuple[pd.Timestamp, str, dict | None]]:
    """When this customer gets in touch, and why."""
    end = acct["canceled_at"] if pd.notna(acct["canceled_at"]) else HORIZON_END
    out: list[tuple[pd.Timestamp, str, dict | None]] = []
    months = max((end - acct["signup_at"]) / MONTH, 0.0)
    for _ in range(r.poisson(BACKGROUND_PER_MONTH * months)):
        at = acct["signup_at"] + (end - acct["signup_at"]) * float(r.random())
        out.append((at, str(r.choice(list(BACKGROUND_INTENTS), p=list(BACKGROUND_INTENTS.values()))), None))
    for c in claims:
        if r.random() < CLAIM_STATUS_P:
            out.append((c["filed_at"] + pd.Timedelta(days=float(r.uniform(1, 6))), "claim_status", c))
        if c["decision"] == "denied" and r.random() < DISPUTE_P:
            out.append((c["decided_at"] + pd.Timedelta(days=float(r.uniform(0, 14))), "claim_dispute", c))
    if pd.notna(acct["canceled_at"]) and r.random() < CANCEL_CONTACT_P:
        out.append((acct["canceled_at"] - pd.Timedelta(days=float(r.uniform(0.1, 3))), "cancel_request", None))
    for at, intent, c in list(out):
        if intent == "technician_no_show" and r.random() < NO_SHOW_REPEAT_P:
            out.append((at + pd.Timedelta(days=float(r.uniform(1, 5))), intent, c))
    # contact centre hours are 08:00-20:00; move each contact into them, same day
    timed = []
    for at, intent, c in out:
        day = at.normalize()
        timed.append((day + pd.Timedelta(seconds=int(r.integers(8 * 3600, 20 * 3600))), intent, c))
    return sorted((t for t in timed if acct["signup_at"] < t[0] <= end), key=lambda t: t[0])


def _pii(r, acct: dict) -> tuple[str, str]:
    kind = str(r.choice(["phone", "email", "card"], p=[0.45, 0.35, 0.2]))
    if kind == "phone":
        return kind, f" You can call me back on {fake_phone(r)}."
    if kind == "email":
        return kind, f" My email is {acct['email']}."
    return kind, f" Can you just charge the fee to {txt.fake_card(r)}?"


def _contact(r, k: int, acct: dict, at: pd.Timestamp, intent: str, claim: dict | None):
    iid = f"CC-{acct['account_id'][4:]}-{k:03d}"
    channel = "voice" if r.random() < VOICE_P else "chat"
    ivr_id, phone, chat_email, needs_id = None, None, None, False
    if channel == "voice":
        phone = acct["phone"]
        if r.random() < IVR_ID_P:
            ivr_id = acct["account_id"]
        else:
            needs_id = True
    else:
        if r.random() < UNKNOWN_CHAT_P:
            chat_email, needs_id = f"guest{int(r.integers(10_000, 99_999))}@example.net", True
        else:
            chat_email = acct["email"]

    events: list[dict] = []

    def ev(t: pd.Timestamp, etype: str, speaker: str | None = None, text: str | None = None, pii: str = ""):
        events.append({"event_id": f"{iid}-{len(events) + 1:02d}", "interaction_id": iid, "seq": len(events) + 1,
                       "event_type": etype, "occurred_at": t, "speaker": speaker, "text": text, "pii_kinds": pii})

    ev(at, "contact.queued")
    wait = _wait_seconds(r, channel, at)
    patience = float(r.exponential(PATIENCE_MEAN_S[channel]))
    row = {"interaction_id": iid, "channel": channel, "caller_phone": phone, "chat_email": chat_email,
           "ivr_account_id": ivr_id, "queued_at": at, "answered_at": None, "escalated_at": None,
           "true_account_id": acct["account_id"], "true_intent": intent,
           "true_claim_id": claim["claim_id"] if claim else None}
    if wait > patience:
        end = at + pd.Timedelta(seconds=patience)
        ev(end, "contact.abandoned")
        row.update(ended_at=end, outcome="abandoned", wait_seconds=round(patience, 1))
        return row, events

    t = at + pd.Timedelta(seconds=wait)
    row.update(answered_at=t, wait_seconds=round(wait, 1))
    ev(t, "contact.answered")
    step = (15, 60) if channel == "voice" else (20, 120)

    def later(t):
        return t + pd.Timedelta(seconds=float(r.uniform(*step)))

    t = later(t)
    ev(t, "message", "agent", str(r.choice(txt.AGENT_LINES)))
    first = txt.opener(intent, r, claim["claim_id"] if claim else None, claim["item"] if claim else None)
    if needs_id:
        first += f" My account number is {acct['account_id']}."
    for i in range(1 + int(r.integers(1, 4))):
        text = first if i == 0 else txt.follow_up(intent, r)
        pii = ""
        if r.random() < PII_P:
            pii, extra = _pii(r, acct)
            text += extra
        t = later(t)
        ev(t, "message", "customer", text, pii)
        t = later(t)
        ev(t, "message", "agent", str(r.choice(txt.AGENT_LINES)))
    escalated = r.random() < ESCALATE_P.get(intent, BASE_ESCALATE_P)
    if escalated:
        t = later(t)
        ev(t, "message", "agent", str(r.choice(txt.ESCALATION_LINES)))
        t = later(t)
        ev(t, "contact.escalated")
        row["escalated_at"] = t
        t = t + pd.Timedelta(minutes=float(r.uniform(3, 15)))
    else:
        t = later(t)
        ev(t, "message", "agent", str(r.choice(txt.CLOSING_LINES)))
    t = later(t)
    ev(t, "contact.ended")
    row.update(ended_at=t, outcome="escalated" if escalated else "handled")
    return row, events


@lru_cache(maxsize=1)
def _full_history_cached(seed: int) -> dict[str, pd.DataFrame]:
    sc = build_support_claims_history(seed)
    claims_by_acct: dict[str, list[dict]] = {}
    for c in sc["claims"].to_dict("records"):
        claims_by_acct.setdefault(c["account_id"], []).append(c)
    rows, events = [], []
    for acct in sc["crm_accounts"].to_dict("records"):
        r = keyed_rng("cc_customer", acct["account_id"], seed=seed)
        for k, (at, intent, c) in enumerate(_contact_times(r, acct, claims_by_acct.get(acct["account_id"], [])),
                                            start=1):
            row, evs = _contact(r, k, acct, at, intent, c)
            rows.append(row)
            events += evs
    cols = ["interaction_id", "channel", "caller_phone", "chat_email", "ivr_account_id", "queued_at",
            "wait_seconds", "answered_at", "escalated_at", "ended_at", "outcome",
            "true_account_id", "true_intent", "true_claim_id"]
    out = {"contact_interactions": pd.DataFrame(rows)[cols], "contact_events": pd.DataFrame(events)}
    for name, df in out.items():
        for c in CONTACT_CENTRE_TABLES[name].time_cols:
            df[c] = pd.to_datetime(df[c]).dt.floor("s").astype("datetime64[us]")
        out[name] = df.sort_values(CONTACT_CENTRE_TABLES[name].primary_key).reset_index(drop=True)
    return out


def build_contact_centre_history(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """The full history up to HORIZON_DATE, including the internal answer columns."""
    return {n: df.copy() for n, df in _full_history_cached(seed).items()}


def _cutoff(as_of: date) -> pd.Timestamp:
    return pd.Timestamp(as_of) + pd.Timedelta(hours=23, minutes=59, seconds=59)


def contact_centre_as_of(as_of: date, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """The pack as the contact-centre platform would show it at the end of `as_of`."""
    cutoff = _cutoff(as_of)
    full = _full_history_cached(seed)
    i = full["contact_interactions"]
    i = i[i.queued_at <= cutoff].copy()
    for c in CONTACT_CENTRE_TABLES["contact_interactions"].time_cols:
        i.loc[i[c] > cutoff, c] = pd.NaT
    live = i.ended_at.isna()
    i.loc[live, "outcome"] = None
    i.loc[i.answered_at.isna() & live, "wait_seconds"] = np.nan
    i.insert(i.columns.get_loc("outcome"), "status", np.where(live, "in_progress", "ended"))
    i["updated_at"] = i[CONTACT_CENTRE_TABLES["contact_interactions"].time_cols].max(axis=1)
    e = full["contact_events"]
    e = e[e.occurred_at <= cutoff].copy()
    return {"contact_interactions": i.drop(columns=CONTACT_CENTRE_TABLES["contact_interactions"].internal_cols)
            .reset_index(drop=True),
            "contact_events": e.drop(columns=CONTACT_CENTRE_TABLES["contact_events"].internal_cols)
            .reset_index(drop=True)}


def _iso(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def contact_centre_webhooks(as_of: date, seed: int = SEED) -> list[dict]:
    """The webhook deliveries made during the day `as_of`, in delivery order, as JSON-safe dicts.

    Each delivery has its own delivery_id; the event_id is stable, so a receiver can drop the
    duplicates and restore event order with (occurred_at, interaction_id, seq).
    """
    cutoff = _cutoff(as_of)
    full = _full_history_cached(seed)
    first_window = cutoff - pd.Timedelta(days=1)
    e = full["contact_events"]
    e = e[(e.occurred_at > first_window - pd.Timedelta(minutes=15)) & (e.occurred_at <= cutoff)]
    inter = full["contact_interactions"].set_index("interaction_id")
    out = []
    for ev in e.to_dict("records"):
        r = keyed_rng("cc_webhook", ev["event_id"], seed=seed)
        lag = float(r.lognormal(np.log(2), 0.5))
        if r.random() < LATE_DELIVERY_P:
            lag += float(r.uniform(60, 600))
        copies = [lag]
        if r.random() < DUPLICATE_DELIVERY_P:
            copies.append(lag + float(r.uniform(30, 300)))
        data: dict = {}
        if ev["event_type"] == "contact.queued":
            src = inter.loc[ev["interaction_id"]]
            data = {k: (None if pd.isna(src[k]) else str(src[k]))
                    for k in ("channel", "caller_phone", "chat_email", "ivr_account_id")}
        elif ev["event_type"] == "message":
            data = {"speaker": ev["speaker"], "text": ev["text"]}
        for n, lag_s in enumerate(copies, start=1):
            delivered = ev["occurred_at"] + pd.Timedelta(seconds=lag_s)
            if first_window < delivered <= cutoff:
                out.append({"delivery_id": f"{ev['event_id']}-d{n}", "event_id": ev["event_id"],
                            "interaction_id": ev["interaction_id"], "seq": int(ev["seq"]),
                            "type": ev["event_type"], "occurred_at": _iso(ev["occurred_at"]),
                            "delivered_at": delivered, "data": data})
    out.sort(key=lambda d: (d["delivered_at"], d["delivery_id"]))
    for d in out:
        d["delivered_at"] = _iso(d["delivered_at"])
    return out
