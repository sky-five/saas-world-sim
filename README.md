# saas-world-sim

A seeded, deterministic simulated SaaS business that you can view as of any date, plus a mock paginated API over it. It is the shared synthetic-data engine behind the analytics demos in [ridgeline-analytics](https://github.com/sky-five/ridgeline-analytics).

> **All data is synthetic.** Every account, invoice, ticket and event is generated from a fixed seed. No real company's data is used.

## What it simulates

A subscription software business sold to roofing contractors, from 2024-01-01 to 2027-12-31, across six source systems:

| System | Tables |
|---|---|
| App database | `accounts`, `subscriptions`, `subscription_changes`, `invoices`, `measurement_orders` |
| Product events | `events` (login, measurement ordered, proposal sent, ...) |
| CRM | `crm_deals`, `crm_calls` |
| Support desk | `support_tickets` (with free-text bodies) |
| Ad platforms | `ad_spend` |
| Customers' own sales ops | `contractor_leads` |

About 7,200 accounts and 960,000 rows as of 2026-06-30.

## How the data is made

- **One fixed history.** The whole period is simulated once from a single seed and cached as parquet. Random draws use keyed streams (`blake2b` of the keys), so results do not depend on process, order or machine.
- **Revealed one day at a time.** `world_as_of(date)` cuts that history at the end of a date: rows created later are dropped, later timestamps are nulled, statuses (open/paid/solved/canceled) are recomputed and `updated_at` is set. A nightly job can move the date forward by one day and see a business that grew by one day.
- **Same date, same rows.** Calling it twice with the same date gives identical tables; `tests/test_fingerprint.py` locks row counts and money totals for 2026-06-30.

## Planted patterns

Built in on purpose so analysis has something real to find (most are checked by a test):

- Channel quality differs: referral and supplier-partner signups convert best, Facebook worst.
- Sending a proposal in week 1 lifts conversion more than 5x.
- Bigger crews convert more, choose the higher plan and churn less.
- Storm season (Apr-Aug) brings over 30% more signups per month.
- Leads answered in under 5 minutes close about 2x as often.
- Measurement-report redos lead to support tickets and raise the monthly churn hazard.
- About 8% of lead stage timestamps are never logged.
- Cancel reasons match the theme of the account's most recent support ticket.

## Install and use

```bash
pip install "saas-world-sim @ git+https://github.com/sky-five/saas-world-sim@v0.1.0"
```

```python
from datetime import date
from saas_world_sim import MockApi, world_as_of

tables = world_as_of(date(2026, 6, 30))          # dict of pandas DataFrames
tables["subscriptions"].mrr_usd.sum()            # 214743.0

api = MockApi(as_of=date(2026, 6, 30))
for page in api.fetch("invoices", updated_since=None, page_size=5000):
    ...                                          # list[dict], JSON-safe
for batch in api.fetch_arrow("events"):
    ...                                          # pyarrow.Table pages for bulk loads
```

`fetch` returns pages sorted by `updated_at` then primary key; pass the last `updated_at` you saw as `updated_since` to pull only changed rows, the way an incremental loader would.

The first call simulates the full history (about a minute on 8 cores) and caches it in `./.cache`; set `SAAS_WORLD_SIM_CACHE` to put it elsewhere.

## Tests

```bash
pip install -e ".[dev]"
pytest -q        # 38 tests: determinism, as-of view, API paging, planted patterns, fingerprint
```

CI runs lint and the full suite on every push.

## Licence

MIT
