"""Support, claims and CRM pack: a simulated home-warranty business.

Customers buy a monthly plan that covers home appliances and systems, file claims when something
breaks, and write to support. Three tables, from three source systems:

- crm_accounts    (CRM)          one row per customer: plan, contact details, cancel date and reason
- claims          (claims desk)  claims intake with a free-text description, decision and payout
- support_tickets (help desk)    tickets with free-text bodies, some linked to a claim

It is a different business from the core roofing-software world and shares only the engine: keyed
random streams, one fixed history, and an as-of view where the same date always gives the same rows.
"""
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from saas_world_sim.config import HORIZON_DATE, SEED, START_DATE
from saas_world_sim.packs.support_claims_text import FIRST_NAMES, LAST_NAMES, fake_phone, render_claim, render_ticket
from saas_world_sim.rng import keyed_rng

HORIZON_END = pd.Timestamp(HORIZON_DATE) + pd.Timedelta(hours=23, minutes=59, seconds=59)
DAY = pd.Timedelta(days=1)
MONTH = pd.Timedelta(days=30.44)


@dataclass(frozen=True)
class Plan:
    share: float
    monthly_usd: int
    covered: frozenset[str]
    claim_limit_usd: int
    service_fee_usd: int


APPLIANCES = frozenset({"refrigerator", "washer", "dryer", "dishwasher", "oven_range"})
SYSTEMS = frozenset({"hvac", "water_heater", "plumbing", "electrical"})
EXTRAS = frozenset({"roof_leak", "pool_pump"})
PLANS: dict[str, Plan] = {
    "basic": Plan(0.45, 39, APPLIANCES, 1500, 75),
    "plus": Plan(0.35, 59, APPLIANCES | SYSTEMS, 3000, 100),
    "premier": Plan(0.20, 79, APPLIANCES | SYSTEMS | EXTRAS, 5000, 125),
}
# share of claims by item, and median repair cost
ITEMS: dict[str, tuple[float, float]] = {
    "refrigerator": (0.13, 650), "washer": (0.10, 450), "dryer": (0.08, 400), "dishwasher": (0.08, 380),
    "oven_range": (0.07, 520), "hvac": (0.18, 1900), "water_heater": (0.10, 1300), "plumbing": (0.10, 600),
    "electrical": (0.07, 500), "roof_leak": (0.05, 1400), "pool_pump": (0.04, 900),
}
STATES = {"TX": 0.2, "FL": 0.18, "AZ": 0.12, "GA": 0.1, "NC": 0.1, "OH": 0.1, "CA": 0.1, "IL": 0.1}
SIGNUP_CHANNELS = {"web": 0.5, "real_estate_closing": 0.3, "phone": 0.2}
OTHER_CANCEL_REASONS = ["price", "sold_home", "no_longer_needed"]
BACKGROUND_INTENTS = {"how_to_file": 0.3, "billing_question": 0.3, "plan_change": 0.2, "technician_no_show": 0.2}

BASE_DAILY_SIGNUPS = 3.0
CLAIMS_PER_MONTH = 0.07
SUMMER_HVAC_MULT = 3.0           # Jun-Aug
WAITING_PERIOD = pd.Timedelta(days=30)
DUPLICATE_P = 0.03               # chance a claim is filed again for the same item a few days later
NO_ESTIMATE_P = 0.15
NO_ESTIMATE_DENY_P = 0.7
BASE_MONTHLY_CHURN = 0.015
DENIAL_CHURN_MULT = 4.0          # for 90 days after a denied claim
DISPUTE_TICKET_P = 0.65
PII_P = 0.15


@dataclass(frozen=True)
class TableSpec:
    primary_key: str
    created_col: str
    time_cols: list[str]
    internal_cols: list[str] = field(default_factory=list)


SUPPORT_CLAIMS_TABLES: dict[str, TableSpec] = {
    "crm_accounts": TableSpec("account_id", "signup_at", ["signup_at", "canceled_at"]),
    "claims": TableSpec("claim_id", "filed_at", ["filed_at", "decided_at", "paid_at"]),
    "support_tickets": TableSpec("ticket_id", "created_at",
                                 ["created_at", "first_reply_at", "escalated_at", "solved_at"], ["true_intent"]),
}


def _signup_times(seed: int) -> list[pd.Timestamp]:
    days = pd.date_range(START_DATE, HORIZON_DATE, freq="D")
    counts = keyed_rng("sc_arrivals", seed=seed).poisson(BASE_DAILY_SIGNUPS * np.linspace(1.0, 1.8, len(days)))
    r = keyed_rng("sc_arrival_times", seed=seed)
    return [d + pd.Timedelta(seconds=int(r.integers(8 * 3600, 21 * 3600))) for d, k in zip(days, counts, strict=True)
            for _ in range(k)]


def _decide(claim: dict, plan: Plan, signup_at: pd.Timestamp, earlier: list[dict], r) -> None:
    """The claims rules, applied in order. Fills decision, denial_reason, approved_usd, decided_at."""
    dup = any(c["item"] == claim["item"] and claim["filed_at"] - c["filed_at"] <= 7 * DAY for c in earlier)
    days_to_decide = float(r.lognormal(np.log(2), 0.6))
    reason = None
    if dup:
        reason = "duplicate"
    elif claim["filed_at"] - signup_at < WAITING_PERIOD:
        reason = "waiting_period"
    elif claim["item"] not in plan.covered:
        reason = "not_covered"
    elif not claim["has_repair_estimate"]:
        days_to_decide = float(r.lognormal(np.log(6), 0.5))
        if r.random() < NO_ESTIMATE_DENY_P:
            reason = "missing_documents"
    decided_at = claim["filed_at"] + pd.Timedelta(days=days_to_decide)
    if reason:
        decision, approved, paid_at = "denied", 0.0, None
    else:
        capped = min(claim["amount_claimed_usd"], plan.claim_limit_usd)
        decision = "partial" if claim["amount_claimed_usd"] > plan.claim_limit_usd else "approved"
        approved = float(max(capped - plan.service_fee_usd, 0))
        paid_at = decided_at + pd.Timedelta(days=float(r.uniform(1, 5)))
    claim.update(decided_at=decided_at, decision=decision, denial_reason=reason, approved_usd=approved,
                 paid_at=paid_at)


def _ticket(r, n: int, acct: dict, at: pd.Timestamp, intent: str, claim: dict | None) -> dict:
    escalate_p = 0.5 if intent == "claim_dispute" else 0.05
    first_reply = at + pd.Timedelta(minutes=float(r.lognormal(np.log(60), 0.9)))
    solved = first_reply + pd.Timedelta(hours=float(r.lognormal(np.log(12), 1.0)))
    escalated = r.random() < escalate_p
    csat_p = [0.25, 0.25, 0.25, 0.15, 0.1] if intent == "claim_dispute" else [0.05, 0.05, 0.15, 0.35, 0.4]
    subject, body = render_ticket(intent, r, claim["claim_id"] if claim else None, claim["item"] if claim else None,
                                  acct["email"], add_pii=r.random() < PII_P)
    return {
        "ticket_id": f"TKT-{acct['account_id'][4:]}-{n:03d}",
        "account_id": acct["account_id"],
        "claim_id": claim["claim_id"] if claim else None,
        "created_at": at,
        "channel": str(r.choice(["email", "chat", "phone"], p=[0.45, 0.35, 0.2])),
        "priority": "high" if intent in ("claim_dispute", "technician_no_show") else "normal",
        "subject": subject,
        "body": body,
        "first_reply_at": first_reply,
        "escalated_at": first_reply + pd.Timedelta(hours=float(r.uniform(1, 24))) if escalated else None,
        "solved_at": solved + (pd.Timedelta(days=2) if escalated else pd.Timedelta(0)),
        "csat": int(r.choice([1, 2, 3, 4, 5], p=csat_p)) if r.random() < 0.4 else None,
        "true_intent": intent,
    }


def _simulate_customer(n: int, signup_at: pd.Timestamp, seed: int) -> tuple[dict, list[dict], list[dict]]:
    r = keyed_rng("sc_customer", n, seed=seed)
    first, last = str(r.choice(FIRST_NAMES)), str(r.choice(LAST_NAMES))
    plan_name = str(r.choice(list(PLANS), p=[p.share for p in PLANS.values()]))
    plan = PLANS[plan_name]
    acct = {
        "account_id": f"CUS-{n:06d}",
        "customer_name": f"{first} {last}",
        "email": f"{first}.{last}{n}@example.com".lower(),
        "phone": fake_phone(r),
        "state": str(r.choice(list(STATES), p=list(STATES.values()))),
        "signup_channel": str(r.choice(list(SIGNUP_CHANNELS), p=list(SIGNUP_CHANNELS.values()))),
        "plan": plan_name,
        "monthly_price_usd": plan.monthly_usd,
        "signup_at": signup_at,
        "canceled_at": None,
        "cancel_reason": None,
    }
    item_names, item_w = list(ITEMS), np.array([w for w, _ in ITEMS.values()])
    claims: list[dict] = []
    tickets: list[tuple[pd.Timestamp, str, dict | None]] = []
    last_denial: pd.Timestamp | None = None
    month_start = signup_at
    while month_start <= HORIZON_END:
        month_end = month_start + MONTH
        hvac_mult = SUMMER_HVAC_MULT if month_start.month in (6, 7, 8) else 1.0
        w = item_w * np.where(np.array(item_names) == "hvac", hvac_mult, 1.0)
        filed = sorted(month_start + (month_end - month_start) * r.random(r.poisson(CLAIMS_PER_MONTH)))
        for at in filed:
            item = str(r.choice(item_names, p=w / w.sum()))
            batch = [(at, item)]
            if r.random() < DUPLICATE_P:
                batch.append((at + pd.Timedelta(days=float(r.uniform(1, 6))), item))
            for t, it in batch:
                c = {
                    "claim_id": f"CLM-{n:06d}-{len(claims) + 1:02d}",
                    "account_id": acct["account_id"],
                    "filed_at": t,
                    "item": it,
                    "description": render_claim(it, r),
                    "amount_claimed_usd": float(round(r.lognormal(np.log(ITEMS[it][1]), 0.6), 2)),
                    "has_photos": bool(r.random() < 0.8),
                    "has_repair_estimate": bool(r.random() >= NO_ESTIMATE_P),
                }
                _decide(c, plan, signup_at, claims, r)
                claims.append(c)
                if r.random() < 0.2:
                    tickets.append((t + pd.Timedelta(days=float(r.uniform(1, 5))), "claim_status", c))
                if c["decision"] == "denied":
                    last_denial = c["decided_at"]
                    if r.random() < DISPUTE_TICKET_P:
                        tickets.append((c["decided_at"] + pd.Timedelta(days=float(r.uniform(0, 14))),
                                        "claim_dispute", c))
        for at in month_start + (month_end - month_start) * r.random(r.poisson(0.05)):
            tickets.append((at, str(r.choice(list(BACKGROUND_INTENTS), p=list(BACKGROUND_INTENTS.values()))),
                            None))
        recently_denied = last_denial is not None and month_end - last_denial <= 90 * DAY
        if r.random() < BASE_MONTHLY_CHURN * (DENIAL_CHURN_MULT if recently_denied else 1.0):
            acct["canceled_at"] = month_end
            acct["cancel_reason"] = "claim_denied" if recently_denied else str(r.choice(OTHER_CANCEL_REASONS))
            if r.random() < 0.5:
                tickets.append((month_end - pd.Timedelta(days=float(r.uniform(0.1, 3))), "cancel_request", None))
            break
        month_start = month_end

    end = acct["canceled_at"] or HORIZON_END
    claims = [c for c in claims if c["filed_at"] <= end]
    out_tickets = [_ticket(r, i, acct, at, intent, c)
                   for i, (at, intent, c) in enumerate(sorted((t for t in tickets if t[0] <= end),
                                                              key=lambda t: t[0]), start=1)]
    return acct, claims, out_tickets


@lru_cache(maxsize=1)
def _full_history_cached(seed: int) -> dict[str, pd.DataFrame]:
    accounts, claims, tickets = [], [], []
    for n, signup_at in enumerate(_signup_times(seed), start=1):
        a, c, t = _simulate_customer(n, signup_at, seed)
        accounts.append(a)
        claims += c
        tickets += t
    out = {"crm_accounts": pd.DataFrame(accounts), "claims": pd.DataFrame(claims),
           "support_tickets": pd.DataFrame(tickets)}
    for name, df in out.items():
        for c in SUPPORT_CLAIMS_TABLES[name].time_cols:
            df[c] = pd.to_datetime(df[c]).dt.floor("s").astype("datetime64[us]")
        out[name] = df.sort_values(SUPPORT_CLAIMS_TABLES[name].primary_key).reset_index(drop=True)
    out["support_tickets"]["csat"] = out["support_tickets"].csat.astype("Int64")
    return out


def build_support_claims_history(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """The full history up to HORIZON_DATE, including internal label columns (true_intent)."""
    return {n: df.copy() for n, df in _full_history_cached(seed).items()}


def support_claims_as_of(as_of: date, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """The pack as its source systems would show it at the end of `as_of`."""
    cutoff = pd.Timestamp(as_of) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    out: dict[str, pd.DataFrame] = {}
    for name, spec in SUPPORT_CLAIMS_TABLES.items():
        df = _full_history_cached(seed)[name]
        df = df[df[spec.created_col] <= cutoff].copy()
        for c in spec.time_cols:
            df.loc[df[c] > cutoff, c] = pd.NaT
        df["updated_at"] = df[spec.time_cols].max(axis=1)
        out[name] = df

    a = out["crm_accounts"]
    a.loc[a.canceled_at.isna(), "cancel_reason"] = None
    a.insert(a.columns.get_loc("plan"), "status", np.where(a.canceled_at.isna(), "active", "canceled"))
    c = out["claims"]
    undecided = c.decided_at.isna()
    c.loc[undecided, ["decision", "denial_reason"]] = None
    c.loc[undecided, "approved_usd"] = np.nan
    c.insert(c.columns.get_loc("decision"), "status", np.where(undecided, "open", c.decision))
    t = out["support_tickets"]
    t.loc[t.solved_at.isna(), "csat"] = pd.NA
    t.insert(t.columns.get_loc("subject"), "status", np.where(t.solved_at.isna(), "open", "solved"))

    for name, spec in SUPPORT_CLAIMS_TABLES.items():
        out[name] = out[name].drop(columns=spec.internal_cols).reset_index(drop=True)
    return out
