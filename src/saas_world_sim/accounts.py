"""Contractor accounts signing up for Ridgeline, from START_DATE to HORIZON_DATE."""
from functools import lru_cache

import numpy as np
import pandas as pd

from saas_world_sim.config import CHANNELS, CREW_SIZES, HORIZON_DATE, REGIONS, SEED, START_DATE
from saas_world_sim.rng import keyed_rng

BASE_DAILY_SIGNUPS = 4.5
PAID_CAMPAIGN_CHANNELS = {"google_ads", "facebook"}

_ADJECTIVES = [
    "Summit", "Cedar", "Iron", "Blue", "Granite", "Eagle", "Northern", "Prairie", "Coastal", "Valley",
    "Liberty", "Pioneer", "Red", "Silver", "Oak", "Lone", "Golden", "Rocky", "Harbor", "Maple",
]
_NOUNS = [
    "Peak", "Ridge", "Shield", "Crown", "Crest", "Gable", "Shingle", "Top", "Line", "Pitch",
    "Rafter", "Truss", "Eave", "Valley", "Canyon", "Point", "Forge", "Stone", "Field", "Bay",
]


def _daily_rate(days: pd.DatetimeIndex) -> np.ndarray:
    growth = np.linspace(1.0, 2.2, len(days))
    season = np.where((days.month >= 4) & (days.month <= 8), 1.6, 1.0)
    return BASE_DAILY_SIGNUPS * growth * season


def build_accounts(seed: int = SEED) -> pd.DataFrame:
    return _build_accounts(seed).copy()


@lru_cache(maxsize=2)
def _build_accounts(seed: int) -> pd.DataFrame:
    days = pd.date_range(START_DATE, HORIZON_DATE, freq="D")
    counts = keyed_rng("arrivals", seed=seed).poisson(_daily_rate(days))

    ch_names, ch_p = list(CHANNELS), [c.share for c in CHANNELS.values()]
    crew_names, crew_p = list(CREW_SIZES), [c.share for c in CREW_SIZES.values()]
    reg_names, reg_p = list(REGIONS), list(REGIONS.values())

    rows = []
    n = 0
    for day, k in zip(days, counts, strict=True):
        for _ in range(k):
            n += 1
            r = keyed_rng("account", n, seed=seed)
            channel = str(r.choice(ch_names, p=ch_p))
            signup_at = day + pd.Timedelta(seconds=int(r.integers(7 * 3600, 20 * 3600)))
            rows.append({
                "account_id": f"ACC-{n:06d}",
                "company_name": f"{r.choice(_ADJECTIVES)} {r.choice(_NOUNS)} Roofing",
                "signup_at": signup_at,
                "channel": channel,
                "campaign_id": f"{channel}-{day:%Y%m}" if channel in PAID_CAMPAIGN_CHANNELS else None,
                "region": str(r.choice(reg_names, p=reg_p)),
                "crew_size": str(r.choice(crew_names, p=crew_p)),
                "intent": float(r.normal()),
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["signup_at", "account_id"], kind="stable").reset_index(drop=True)
