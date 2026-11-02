"""The whole simulated history, and the view of it as of any date.

`build_full_history` simulates every account up to HORIZON_DATE once and caches it as parquet.
`world_as_of` cuts that history at a date: rows created later are dropped, later timestamps are
nulled, statuses are recomputed, and `updated_at` is set. The same date always gives the same rows.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from saas_world_sim.accounts import build_accounts
from saas_world_sim.config import HORIZON_DATE, SEED, SIM_VERSION
from saas_world_sim.lifecycle import simulate_account
from saas_world_sim.rng import keyed_rng
from saas_world_sim.side_systems import simulate_ad_spend, simulate_crm, simulate_leads, simulate_support

# parquet cache of the full history; set SAAS_WORLD_SIM_CACHE to move it
CACHE_DIR = Path(os.environ.get("SAAS_WORLD_SIM_CACHE", ".cache"))


@dataclass(frozen=True)
class TableSpec:
    system: str
    primary_key: str
    mutable: bool
    created_col: str
    time_cols: list[str] = field(default_factory=list)
    internal_cols: list[str] = field(default_factory=list)


TABLES: dict[str, TableSpec] = {
    "accounts": TableSpec("app_db", "account_id", False, "signup_at", ["signup_at"], ["intent"]),
    "subscriptions": TableSpec("app_db", "subscription_id", True, "started_at"),  # state built from changes
    "subscription_changes": TableSpec("app_db", "change_id", False, "changed_at", ["changed_at"]),
    "invoices": TableSpec("app_db", "invoice_id", True, "issued_at", ["issued_at", "paid_at"]),
    "measurement_orders": TableSpec("app_db", "order_id", True, "ordered_at",
                                    ["ordered_at", "delivered_at", "redo_requested_at"]),
    "events": TableSpec("product_events", "event_id", False, "occurred_at", ["occurred_at"]),
    "crm_deals": TableSpec("crm", "deal_id", True, "created_at", ["created_at", "closed_at"]),
    "crm_calls": TableSpec("crm", "call_id", False, "called_at", ["called_at"]),
    "support_tickets": TableSpec("support", "ticket_id", True, "created_at",
                                 ["created_at", "first_reply_at", "solved_at"], ["true_theme"]),
    "ad_spend": TableSpec("ads", "spend_id", False, "_day_end", ["_day_end"]),
    "contractor_leads": TableSpec("contractor_ops", "lead_id", True, "created_at",
                                  ["created_at", "first_contact_at", "inspection_at", "proposal_sent_at",
                                   "closed_at"]),
}


def _simulate_chunk(chunk: pd.DataFrame) -> dict[str, list]:
    out: dict[str, list] = {k: [] for k in
                            ["events", "subscriptions", "subscription_changes", "invoices", "measurement_orders",
                             "support_tickets", "crm_deals", "crm_calls", "contractor_leads"]}
    for acct in chunk.itertuples(index=False):
        h = simulate_account(acct)
        out["events"].append(h.events)
        if h.subscription:
            out["subscriptions"].append(h.subscription)
        out["subscription_changes"] += h.subscription_changes
        out["invoices"] += h.invoices
        out["measurement_orders"] += h.measurement_orders
        out["support_tickets"] += simulate_support(h)
        deals, calls = simulate_crm(h)
        out["crm_deals"] += deals
        out["crm_calls"] += calls
        out["contractor_leads"] += simulate_leads(h)
    return out


def _cancel_reasons(changes: pd.DataFrame, tickets: pd.DataFrame) -> pd.Series:
    """Theme of the latest ticket in the 90 days before each cancel, else price/no_longer_needed."""
    cancels = changes[changes.change_type == "cancel"][["change_id", "account_id", "changed_at"]]
    t = tickets[["account_id", "created_at", "true_theme"]].sort_values("created_at")
    m = pd.merge_asof(cancels.sort_values("changed_at"), t, left_on="changed_at", right_on="created_at",
                      by="account_id", direction="backward", tolerance=pd.Timedelta(days=90))
    fallback = [str(keyed_rng("cancel", cid).choice(["price", "no_longer_needed"])) for cid in m.change_id]
    m["reason"] = m.true_theme.where(m.true_theme.notna(), pd.Series(fallback, index=m.index))
    return changes.change_id.map(m.set_index("change_id").reason)


def _simulate(seed: int) -> dict[str, pd.DataFrame]:
    accounts = build_accounts(seed)
    chunks = [accounts.iloc[i::8] for i in range(8)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        parts = list(ex.map(_simulate_chunk, chunks))

    t: dict[str, pd.DataFrame] = {"accounts": accounts}
    events = pd.concat([e for p in parts for e in p["events"]], ignore_index=True)
    events = events.sort_values(["account_id", "occurred_at", "event_type"], kind="stable")
    seq = events.groupby("account_id").cumcount() + 1
    events.insert(0, "event_id", "EVT-" + events.account_id.str[4:] + "-" + seq.astype(str).str.zfill(6))
    t["events"] = events.reset_index(drop=True)
    for name in ["subscriptions", "subscription_changes", "invoices", "measurement_orders", "support_tickets",
                 "crm_deals", "crm_calls", "contractor_leads"]:
        df = pd.DataFrame([r for p in parts for r in p[name]])
        t[name] = df.sort_values(TABLES[name].primary_key).reset_index(drop=True)

    t["subscription_changes"]["cancel_reason"] = _cancel_reasons(t["subscription_changes"], t["support_tickets"])
    ads = pd.DataFrame(simulate_ad_spend(accounts, seed))
    ads["spend_date"] = pd.to_datetime(ads.spend_date)
    ads.insert(0, "spend_id", ads.spend_date.dt.strftime("%Y%m%d") + "-" + ads.channel)
    ads["_day_end"] = ads.spend_date + pd.Timedelta(hours=23, minutes=59, seconds=59)
    t["ad_spend"] = ads

    for df in t.values():  # uniform datetime dtype, None -> NaT
        for c in df.columns:
            if c.endswith("_at") or c in ("period_start_date", "_day_end", "spend_date"):
                df[c] = pd.to_datetime(df[c]).dt.floor("s").astype("datetime64[us]")
    return t


@lru_cache(maxsize=1)
def _full_history_cached(seed: int) -> dict[str, pd.DataFrame]:
    folder = CACHE_DIR / f"full-{seed}-{HORIZON_DATE}-v{SIM_VERSION}"
    if folder.exists() and all((folder / f"{n}.parquet").exists() for n in TABLES):
        return {n: pd.read_parquet(folder / f"{n}.parquet") for n in TABLES}
    t = _simulate(seed)
    folder.mkdir(parents=True, exist_ok=True)
    for n, df in t.items():
        df.to_parquet(folder / f"{n}.parquet", index=False)
    return t


def build_full_history(seed: int = SEED) -> dict[str, pd.DataFrame]:
    return {n: df.copy() for n, df in _full_history_cached(seed).items()}


def _subscriptions_as_of(full: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> pd.DataFrame:
    ch = full["subscription_changes"]
    ch = ch[ch.changed_at <= cutoff].sort_values(["subscription_id", "changed_at"])
    last = ch.groupby("subscription_id").tail(1).set_index("subscription_id")
    subs = full["subscriptions"].set_index("subscription_id").loc[last.index].reset_index()
    subs = subs.drop(columns=["plan", "canceled_at", "cancel_reason"])
    last = last.reindex(subs.subscription_id)
    canceled = (last.mrr_after == 0).to_numpy()
    subs["status"] = np.where(canceled, "canceled", "active")
    # canceled subs keep the last plan they were on
    prior_plan = ch.dropna(subset=["plan_after"]).groupby("subscription_id").plan_after.last()
    subs["plan"] = subs.subscription_id.map(prior_plan).to_numpy()
    subs["addons"] = last.addons_after.to_numpy()
    subs["mrr_usd"] = last.mrr_after.to_numpy()
    subs["canceled_at"] = last.changed_at.where(canceled).to_numpy()
    subs["cancel_reason"] = np.where(canceled, last.cancel_reason, None)
    subs["updated_at"] = last.changed_at.to_numpy()
    return subs


def world_as_of(as_of: date, seed: int = SEED) -> dict[str, pd.DataFrame]:
    cutoff = pd.Timestamp(as_of) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    full = _full_history_cached(seed)
    out: dict[str, pd.DataFrame] = {}
    for name, spec in TABLES.items():
        if name == "subscriptions":
            out[name] = _subscriptions_as_of(full, cutoff)
            continue
        df = full[name]
        df = df[df[spec.created_col] <= cutoff].copy()
        for c in spec.time_cols:
            df.loc[df[c] > cutoff, c] = pd.NaT
        df["updated_at"] = df[spec.time_cols].max(axis=1)
        out[name] = df

    inv = out["invoices"]
    inv["status"] = np.where(inv.paid_at.isna(), "open", "paid")
    mo = out["measurement_orders"]
    mo.loc[mo.redo_requested_at.isna(), "redo_reason"] = None
    st = out["support_tickets"]
    st["status"] = np.where(st.solved_at.isna(), "open", "solved")
    deals = out["crm_deals"]
    deals.loc[deals.closed_at.isna(), "stage"] = "open"
    leads = out["contractor_leads"]
    leads.loc[leads.closed_at.isna(), "outcome"] = "open"
    leads.loc[leads.proposal_sent_at.isna(), "proposal_value_usd"] = np.nan
    ch = out["subscription_changes"]
    ch.loc[ch.change_type != "cancel", "cancel_reason"] = None

    for name, spec in TABLES.items():
        drop = [c for c in out[name].columns if c.startswith("_") or c in spec.internal_cols]
        out[name] = out[name].drop(columns=drop).reset_index(drop=True)
    return out
