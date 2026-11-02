"""Systems around the core app: support desk, CRM, ad spend, and contractors' own homeowner leads."""
import numpy as np
import pandas as pd

from saas_world_sim.config import CHANNELS, HORIZON_DATE, SEED, START_DATE
from saas_world_sim.lifecycle import HORIZON_END, AccountHistory, sigmoid
from saas_world_sim.rng import keyed_rng
from saas_world_sim.text import render_ticket

REPS = ["Avery", "Jordan", "Riley", "Casey", "Morgan", "Quinn"]
THEME_MIX = {"how_to": 0.35, "billing_price": 0.2, "missing_feature": 0.2, "bug_outage": 0.15,
             "integration": 0.1}
LEAD_SOURCES = {  # median minutes to first contact, close-odds shift
    "instant_estimator": (6, 0.0),
    "google_lsa": (25, -0.2),
    "referral": (60, 0.9),
    "door_knocking": (1, 0.3),
    "facebook": (90, -0.8),
    "home_show": (600, -0.3),
}
LEAD_SOURCE_P = [0.28, 0.22, 0.15, 0.15, 0.12, 0.08]
LEADS_PER_MONTH = 8
CREW_LEAD_MULT = {"1": 0.5, "2-5": 1.0, "6-15": 2.0, "16+": 4.0}
CLICKS_PER_SIGNUP = {"google_ads": 25, "facebook": 60}
MONTH = pd.Timedelta(days=30.44)


def active_intervals(h: AccountHistory) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Paid periods from the subscription change log."""
    out, start = [], None
    for c in sorted(h.subscription_changes, key=lambda c: c["changed_at"]):
        if c["mrr_after"] > 0 and start is None:
            start = c["changed_at"]
        elif c["mrr_after"] == 0 and start is not None:
            out.append((start, c["changed_at"]))
            start = None
    if start is not None:
        out.append((start, HORIZON_END))
    return out


def _poisson_times(r, rate_per_month: float, start, end) -> list[pd.Timestamp]:
    if end <= start:
        return []
    n = r.poisson(rate_per_month * ((end - start) / MONTH))
    return list(pd.DatetimeIndex(start + (end - start) * r.random(n)).sort_values())


def simulate_support(h: AccountHistory) -> list[dict]:
    r = keyed_rng("support", h.account_id)
    short = h.account_id[4:]
    tickets: list[tuple[pd.Timestamp, str]] = []
    for start, end in active_intervals(h):
        for t in _poisson_times(r, 0.3, start, end):
            tickets.append((t, str(r.choice(list(THEME_MIX), p=list(THEME_MIX.values())))))
    if not h.subscription:  # free accounts rarely write in
        free_end = min(h.signup_at + pd.Timedelta(days=60), HORIZON_END)
        tickets += [(t, "how_to") for t in _poisson_times(r, 0.1, h.signup_at, free_end)]
    for o in h.measurement_orders:
        if o["redo_requested_at"] is not None and r.random() < 0.8:
            tickets.append((o["redo_requested_at"] + pd.Timedelta(days=float(r.uniform(0, 7))),
                            "measurement_accuracy"))
    out = []
    for i, (t, theme) in enumerate(sorted(tickets), start=1):
        if t > HORIZON_END:
            continue
        subject, body = render_ticket(theme, r)
        first_reply = t + pd.Timedelta(minutes=float(r.lognormal(np.log(45), 0.9)))
        out.append({
            "ticket_id": f"TCK-{short}-{i:04d}",
            "account_id": h.account_id,
            "created_at": t,
            "first_reply_at": first_reply,
            "solved_at": first_reply + pd.Timedelta(hours=float(r.lognormal(np.log(10), 1.0))),
            "priority": str(r.choice(["low", "normal", "high", "urgent"], p=[0.2, 0.55, 0.2, 0.05])),
            "channel": str(r.choice(["email", "chat", "phone"], p=[0.5, 0.35, 0.15])),
            "subject": subject,
            "body": body,
            "true_theme": theme,
        })
    return out


def simulate_crm(h: AccountHistory) -> tuple[list[dict], list[dict]]:
    r = keyed_rng("crm", h.account_id)
    short = h.account_id[4:]
    calls, deals = [], []
    if r.random() >= 0.35:
        return deals, calls
    rep = str(r.choice(REPS))
    called_at = h.signup_at + pd.Timedelta(days=float(r.uniform(2, 30)))
    if called_at > HORIZON_END:
        return deals, calls
    early = h.events[h.events.occurred_at <= h.signup_at + pd.Timedelta(days=14)]
    busy = len(early) > 8
    outcome = str(r.choice(["connected", "voicemail", "no_answer", "demo_booked"],
                           p=[0.3, 0.3, 0.25, 0.15] if busy else [0.25, 0.35, 0.35, 0.05]))
    calls.append({
        "call_id": f"CAL-{short}-01",
        "account_id": h.account_id,
        "rep": rep,
        "called_at": called_at,
        "outcome": outcome,
        "duration_s": int(r.integers(20, 60)) if outcome in ("voicemail", "no_answer")
        else int(r.integers(120, 1500)),
    })
    if outcome == "demo_booked":
        paid_at = h.subscription["started_at"] if h.subscription else None
        won = paid_at is not None and paid_at <= called_at + pd.Timedelta(days=45)
        closed_at = max(paid_at, called_at) if won else called_at + pd.Timedelta(days=45)
        deals.append({
            "deal_id": f"DEA-{short}-01",
            "account_id": h.account_id,
            "owner_rep": rep,
            "created_at": called_at,
            "stage": "won" if won else "lost",
            "closed_at": closed_at if closed_at <= HORIZON_END else None,
            "amount_usd": 349 * 12 if h.crew_size in ("6-15", "16+") else 249 * 12,
        })
    return deals, calls


def simulate_leads(h: AccountHistory) -> list[dict]:
    """The contractor's own homeowner leads while they're a paying customer."""
    r = keyed_rng("leads", h.account_id)
    short = h.account_id[4:]
    sources = list(LEAD_SOURCES)
    rate = LEADS_PER_MONTH * CREW_LEAD_MULT[h.crew_size]
    out = []
    for start, end in active_intervals(h):
        for created in _poisson_times(r, rate, start, end):
            if 4 <= created.month <= 8 and r.random() < 0.8:  # storm season ~1.8x: add a twin lead
                out.append(_lead(r, h, short, len(out) + 1, created + pd.Timedelta(hours=float(r.uniform(1, 72))),
                                 sources))
            out.append(_lead(r, h, short, len(out) + 1, created, sources))
    out = [lead for lead in out if lead["created_at"] <= HORIZON_END]
    return sorted(out, key=lambda x: x["created_at"])


def _lead(r, h, short: str, n: int, created: pd.Timestamp, sources: list[str]) -> dict:
    src = str(r.choice(sources, p=LEAD_SOURCE_P))
    median, shift = LEAD_SOURCES[src]
    mins = float(np.exp(r.normal(np.log(median), 1.1)))
    fast = mins < 5
    contacted = r.random() < 0.9
    first_contact = created + pd.Timedelta(minutes=mins) if contacted else None
    inspection = proposal = None
    if contacted and r.random() < sigmoid(0.6 + 0.8 * fast + 0.5 * shift):
        inspection = first_contact + pd.Timedelta(days=float(r.gamma(2, 1.5)))
        if r.random() < 0.82:
            proposal = inspection + pd.Timedelta(days=float(r.gamma(2, 1)))
    if not contacted:
        outcome, closed = "no_response", created + pd.Timedelta(days=float(r.uniform(3, 10)))
    elif proposal is not None:
        if r.random() < sigmoid(-0.5 + 0.7 * fast + shift):
            outcome, closed = "won", proposal + pd.Timedelta(days=float(r.gamma(2, 3)))
        else:
            outcome = str(r.choice(["lost_price", "lost_competitor", "lost_no_decision"], p=[0.4, 0.35, 0.25]))
            closed = proposal + pd.Timedelta(days=float(r.gamma(2, 6)))
    else:
        outcome = "unqualified"
        closed = (inspection or first_contact) + pd.Timedelta(days=float(r.uniform(0.5, 5)))
    value = round(float(np.exp(r.normal(np.log(14500), 0.45))), -1) if proposal is not None else None
    # planted data gap: stage times sometimes not logged
    if first_contact is not None and r.random() < 0.08:
        first_contact = None
    if inspection is not None and r.random() < 0.08:
        inspection = None
    return {
        "lead_id": f"LEA-{short}-{n:05d}",
        "account_id": h.account_id,
        "source": src,
        "created_at": created,
        "first_contact_at": first_contact,
        "inspection_at": inspection,
        "proposal_sent_at": proposal,
        "closed_at": closed,
        "outcome": outcome,
        "proposal_value_usd": value,
    }


def simulate_ad_spend(accounts: pd.DataFrame, seed: int = SEED) -> list[dict]:
    """Daily spend per paid channel. Search/social spend tracks the signups it bought."""
    r = keyed_rng("ads", seed=seed)
    days = pd.date_range(START_DATE, HORIZON_DATE, freq="D")
    signups = accounts.assign(d=accounts.signup_at.dt.normalize()).groupby(["d", "channel"]).size()
    out = []
    for day in days:
        for ch, spec in CHANNELS.items():
            if spec.cpc_usd is not None:
                n = int(signups.get((day, ch), 0))
                clicks = int(max(n, 0.3) * CLICKS_PER_SIGNUP[ch] * r.uniform(0.85, 1.15))
                spend = round(clicks * spec.cpc_usd, 2)
            elif spec.monthly_cost_usd is not None:
                clicks = 0
                spend = round(spec.monthly_cost_usd / day.days_in_month, 2)
            else:
                continue
            out.append({
                "spend_date": day.date(),
                "channel": ch,
                "campaign_id": f"{ch}-{day:%Y%m}",
                "spend_usd": spend,
                "impressions": int(clicks * r.uniform(30, 60)) if clicks else int(spend * 8),
                "clicks": clicks,
            })
    return out
