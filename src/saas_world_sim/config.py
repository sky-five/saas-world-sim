"""Fixed settings for the simulated roofing-software business. All values are made up."""
from dataclasses import dataclass
from datetime import date

SEED = 20260101
SIM_VERSION = 2  # bump when simulator logic changes (invalidates the history cache)
START_DATE = date(2024, 1, 1)
HORIZON_DATE = date(2027, 12, 31)


@dataclass(frozen=True)
class ChannelSpec:
    share: float
    conv_logit: float
    monthly_cost_usd: float | None
    cpc_usd: float | None


@dataclass(frozen=True)
class CrewSpec:
    share: float
    conv_logit: float
    churn_mult: float
    scale_plan_p: float


CHANNELS: dict[str, ChannelSpec] = {
    "google_ads": ChannelSpec(0.30, 0.0, None, 4.2),
    "organic": ChannelSpec(0.20, 0.2, None, None),
    "facebook": ChannelSpec(0.18, -0.5, None, 1.6),
    "referral": ChannelSpec(0.12, 1.0, None, None),
    "supplier_partner": ChannelSpec(0.12, 0.6, 9000, None),
    "trade_show": ChannelSpec(0.08, 0.3, 14000, None),
}

CREW_SIZES: dict[str, CrewSpec] = {
    "1": CrewSpec(0.35, -0.4, 1.3, 0.05),
    "2-5": CrewSpec(0.40, 0.0, 1.0, 0.20),
    "6-15": CrewSpec(0.18, 0.4, 0.8, 0.70),
    "16+": CrewSpec(0.07, 0.6, 0.6, 0.90),
}

REGIONS: dict[str, float] = {
    "Southeast": 0.28,
    "Southwest": 0.22,
    "Midwest": 0.18,
    "Northeast": 0.14,
    "West": 0.10,
    "Canada": 0.08,
}

PLAN_PRICES: dict[str, int] = {"essentials": 249, "scale": 349}
ADDON_PRICES: dict[str, int] = {"instant_estimator": 149, "ai_receptionist": 99, "sms": 49}
REPORT_PRICE: dict[str, int] = {"free": 19, "paid": 13}
