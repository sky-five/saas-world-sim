"""One account's life: early usage, conversion, plan changes, billing, measurement orders."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from saas_world_sim.config import ADDON_PRICES, CHANNELS, CREW_SIZES, HORIZON_DATE, PLAN_PRICES, REPORT_PRICE
from saas_world_sim.rng import keyed_rng

HORIZON_END = pd.Timestamp(HORIZON_DATE) + pd.Timedelta(hours=23, minutes=59, seconds=59)
DAY = pd.Timedelta(days=1)

# weekly event rates after the first 14 days
PAID_WEEKLY = {"login": 5, "measurement_ordered": 2.5, "proposal_sent": 2, "job_created": 1.2,
               "invoice_paid": 0.8}
FREE_WEEKLY = {"login": 1, "measurement_ordered": 0.4, "proposal_sent": 0.15}
ROOF_TYPES = (["gable", "hip", "complex", "flat"], [0.4, 0.3, 0.2, 0.1])
REDO_REASONS = ["pitch_wrong", "tree_cover", "missing_facet", "imagery_outdated"]
BASE_MONTHLY_CHURN = 0.045


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


@dataclass
class AccountHistory:
    account_id: str
    signup_at: pd.Timestamp
    channel: str
    region: str
    crew_size: str
    intent: float
    proposal_wk1: bool = False
    # columns: account_id, event_type, occurred_at, value_usd
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    subscription: dict | None = None
    subscription_changes: list[dict] = field(default_factory=list)
    invoices: list[dict] = field(default_factory=list)
    measurement_orders: list[dict] = field(default_factory=list)


class _Builder:
    """Accumulates one account's records while the simulation walks forward in time."""

    def __init__(self, acct, rng: np.random.Generator):
        self.a = acct
        self.r = rng
        self.short = acct.account_id[4:]
        self.ev_type: list[str] = []
        self.ev_time: list[pd.Timestamp] = []
        self.orders: list[dict] = []
        self.paid = False

    def add_events(self, etype: str, times) -> None:
        times = [t for t in times if self.a.signup_at <= t <= HORIZON_END]
        self.ev_type += [etype] * len(times)
        self.ev_time += times
        if etype == "measurement_ordered":
            for t in times:
                self._order(t)

    def burst(self, rates: dict[str, float], start: pd.Timestamp, end: pd.Timestamp) -> None:
        """Poisson activity at weekly `rates` between start and end."""
        weeks = (end - start) / pd.Timedelta(days=7)
        if weeks <= 0:
            return
        for etype, lam in rates.items():
            n = self.r.poisson(lam * weeks)
            self.add_events(etype, list(pd.DatetimeIndex(start + (end - start) * self.r.random(n)).sort_values()))

    def _order(self, t: pd.Timestamp) -> None:
        r = self.r
        tier = str(r.choice(["2h", "6h", "24h"], p=[0.3, 0.4, 0.3])) if self.paid else "24h"
        roof = str(r.choice(ROOF_TYPES[0], p=ROOF_TYPES[1]))
        delivered = t + pd.Timedelta(hours=int(tier[:-1]) * float(r.uniform(0.6, 1.1)))
        p_redo = 0.04 * (2.5 if roof == "complex" else 1) * (1.8 if self.a.region == "Canada" else 1)
        redo_at = delivered + pd.Timedelta(days=float(r.uniform(0.5, 3))) if r.random() < p_redo else None
        self.orders.append({
            "order_id": f"ORD-{self.short}-{len(self.orders) + 1:05d}",
            "account_id": self.a.account_id,
            "ordered_at": t,
            "delivered_at": delivered,
            "tier": tier,
            "price_usd": REPORT_PRICE["paid" if self.paid else "free"],
            "roof_type": roof,
            "redo_requested_at": redo_at,
            "redo_reason": str(r.choice(REDO_REASONS)) if redo_at is not None else None,
        })

    def redo_since(self, t: pd.Timestamp) -> bool:
        return any(o["redo_requested_at"] is not None and t <= o["redo_requested_at"] for o in self.orders)


def _mrr(plan: str | None, addons: list[str]) -> int:
    if plan is None:
        return 0
    return PLAN_PRICES[plan] + sum(ADDON_PRICES[a] for a in addons)


def simulate_account(acct) -> AccountHistory:
    r = keyed_rng("life", acct.account_id)
    b = _Builder(acct, r)
    ch, crew = CHANNELS[acct.channel], CREW_SIZES[acct.crew_size]
    s0 = acct.signup_at
    h = AccountHistory(acct.account_id, s0, acct.channel, acct.region, acct.crew_size, acct.intent)

    # --- first 14 days
    engaged = r.random() < sigmoid(-0.3 + acct.intent + 0.5 * ch.conv_logit)
    n_meas = r.poisson(2.2 if engaged else 0.4)
    wk1 = bool(engaged and r.random() < sigmoid(-0.4 + 0.8 * acct.intent + 0.3 * crew.conv_logit))
    n_prop = r.poisson(1.5) + 1 if wk1 else (r.poisson(0.5) if engaged else 0)
    n_signed = r.binomial(n_prop, 0.25)
    n_login = r.poisson(6 if engaged else 1.5)
    h.proposal_wk1 = wk1

    def within(days: float, n: int) -> list[pd.Timestamp]:
        return sorted(s0 + pd.Timedelta(days=float(d)) for d in r.uniform(0, days, n))

    b.add_events("login", within(14, n_login))
    b.add_events("measurement_ordered", within(14, n_meas))
    props = within(7 if wk1 else 14, n_prop)
    b.add_events("proposal_sent", props)
    signed = [p + pd.Timedelta(days=float(r.uniform(0.5, 5))) for p in props[:n_signed]]
    b.add_events("proposal_signed", signed)

    # --- conversion
    logit = (-3.1 + 0.9 * acct.intent + ch.conv_logit + crew.conv_logit + 1.6 * wk1
             + 0.25 * min(n_meas, 6) + 0.5 * (n_signed > 0))
    converts = r.random() < sigmoid(logit)
    paid_at = s0 + pd.Timedelta(days=float(r.gamma(2.0, 12 if wk1 else 25)) + 3) if converts else None
    if paid_at is not None and paid_at > HORIZON_END:
        paid_at = None

    day14 = s0 + pd.Timedelta(days=14)
    if paid_at is None:
        b.burst(FREE_WEEKLY, day14, min(day14 + pd.Timedelta(days=float(r.exponential(40))), HORIZON_END))
    else:
        b.burst(FREE_WEEKLY, day14, max(day14, paid_at))
        _paid_life(h, b, r, crew, paid_at)

    events = pd.DataFrame({"event_type": b.ev_type, "occurred_at": pd.to_datetime(b.ev_time)})
    events = events.sort_values("occurred_at", kind="stable").reset_index(drop=True)
    is_prop = events.event_type.isin(["proposal_sent", "proposal_signed"])
    events["value_usd"] = np.where(is_prop, np.round(np.exp(r.normal(np.log(14500), 0.45, len(events))), -1),
                                   np.nan)
    events.insert(0, "account_id", acct.account_id)
    h.events = events
    h.measurement_orders = b.orders
    return h


def _paid_life(h: AccountHistory, b: _Builder, r: np.random.Generator, crew, paid_at: pd.Timestamp) -> None:
    sub_id = f"SUB-{b.short}"
    plan = "scale" if r.random() < crew.scale_plan_p else "essentials"
    addons: list[str] = []
    mrr = 0
    changes, invoices = h.subscription_changes, h.invoices

    def change(t: pd.Timestamp, ctype: str, new_plan: str | None, new_addons: list[str]) -> None:
        nonlocal mrr
        after = _mrr(new_plan, new_addons)
        changes.append({
            "change_id": f"CHG-{b.short}-{len(changes) + 1:04d}",
            "subscription_id": sub_id,
            "account_id": h.account_id,
            "changed_at": t,
            "change_type": ctype,
            "plan_after": new_plan,
            "addons_after": ",".join(sorted(new_addons)),
            "mrr_before": mrr,
            "mrr_after": after,
        })
        mrr = after

    def invoice(t: pd.Timestamp) -> None:
        unpaid = r.random() < 0.03
        invoices.append({
            "invoice_id": f"INV-{b.short}-{len(invoices) + 1:05d}",
            "account_id": h.account_id,
            "subscription_id": sub_id,
            "period_start_date": t.normalize(),
            "issued_at": t,
            "amount_usd": mrr,
            "paid_at": t + pd.Timedelta(days=float(r.uniform(30, 40) if unpaid else r.uniform(0, 5))),
            "line_type": "subscription",
        })

    b.paid = True
    change(paid_at, "new", plan, addons)
    active, anchor, k = True, paid_at, 0  # k = months since anchor
    canceled_at = None
    months_canceled = 0
    while True:
        period_start = anchor + pd.DateOffset(months=k)
        if period_start > HORIZON_END:
            break
        period_end = anchor + pd.DateOffset(months=k + 1)
        if active:
            invoice(period_start)
            b.burst(PAID_WEEKLY, period_start, min(period_end, HORIZON_END))
            t = period_start + (period_end - period_start) * float(r.uniform(0.05, 0.95))
            if t > HORIZON_END:
                break
            p_churn = BASE_MONTHLY_CHURN * crew.churn_mult * (1.8 if b.redo_since(t - 60 * DAY) else 1)
            u = r.random()
            if u < p_churn:
                change(t, "cancel", None, [])
                active, canceled_at, months_canceled = False, t, 0
            elif u < p_churn + 0.015 and plan == "essentials":
                plan = "scale"
                change(t, "upgrade", plan, addons)
            elif u < p_churn + 0.023 and plan == "scale":
                plan = "essentials"
                change(t, "downgrade", plan, addons)
            elif u < p_churn + 0.053 and len(addons) < 2:
                addons = sorted(addons + [str(r.choice([a for a in ADDON_PRICES if a not in addons]))])
                change(t, "addon_added", plan, addons)
            elif u < p_churn + 0.063 and addons:
                addons = addons[1:]
                change(t, "addon_removed", plan, addons)
        else:
            months_canceled += 1
            if months_canceled > 6:
                break
            if r.random() < 0.06:
                t = period_start + (period_end - period_start) * float(r.uniform(0.05, 0.95))
                if t > HORIZON_END:
                    break
                addons = []
                change(t, "reactivate", plan, addons)
                active, canceled_at, anchor, k = True, None, t, 0
                continue
        k += 1

    h.subscription = {
        "subscription_id": sub_id,
        "account_id": h.account_id,
        "plan": plan,
        "started_at": paid_at,
        "canceled_at": canceled_at,
        "cancel_reason": None,  # set by world.py from the latest ticket theme
    }
