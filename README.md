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

## Domain pack: support, claims and CRM

A second, smaller business on the same engine: a home-warranty provider whose customers pay monthly for a plan, file claims when an appliance or home system breaks, and write to support. It is separate from the roofing world above, so adding it changed none of the core numbers. It is built for testing support and claims agents against known answers.

| System | Table | What is in it |
|---|---|---|
| CRM | `crm_accounts` | customer, plan (basic / plus / premier), contact details, status, cancel date and reason |
| Claims desk | `claims` | claims intake: item, free-text description, amount, photos / repair estimate, decision, payout |
| Help desk | `support_tickets` | free-text subject and body, optional linked claim, escalation, CSAT |

3,468 customers, 3,011 claims and 3,856 tickets as of 2026-06-30. The full history to 2027-12-31 builds in about 10 seconds and is not cached on disk.

```python
from datetime import date
from saas_world_sim.packs import support_claims_as_of

tables = support_claims_as_of(date(2026, 6, 30))  # crm_accounts, claims, support_tickets
tables["claims"].approved_usd.sum()                # 1461461.45
```

The as-of view works like the core one: later rows dropped, later timestamps nulled, statuses recomputed (open claims have no decision yet, open tickets no CSAT), `updated_at` set.

**Claims rules**, applied in order, so an agent's answer can be checked against them:

1. Same item claimed again within 7 days: denied, `duplicate`.
2. Filed within 30 days of signup: denied, `waiting_period`.
3. Item not covered by the plan: denied, `not_covered` (basic covers appliances; plus adds HVAC, water heater, plumbing, electrical; premier adds roof leaks and pool pumps).
4. No repair estimate: decided about 3x slower and 70% denied, `missing_documents`.
5. Otherwise paid: the claim capped at the plan limit ($1,500 / $3,000 / $5,000) minus the service fee ($75 / $100 / $125); `partial` when the claim was over the limit.

**Planted patterns** (each checked by a test in `tests/test_support_claims.py`):

- HVAC is 42% of claims in June-August against 20% the rest of the year.
- 64% of denied claims get a dispute ticket within 14 days; dispute tickets escalate 10x as often (50% vs 5%) and score lower CSAT (2.5 vs 4.0).
- Customers with a denied claim cancel more (37% vs 23% of other claimants); `claim_denied` as a cancel reason always follows a denial in the last 90 days.
- About 15% of ticket bodies contain contact details, for redaction tests. Phone numbers are all in the 555-0100 to 555-0199 range set aside for fiction; emails use example.com.
- About 3% of claims are duplicates.

Ticket intent labels (`true_intent`) are kept in `build_support_claims_history()` for scoring and dropped from the as-of view. `tests/test_support_claims_fingerprint.py` locks the pack's row counts and totals.

## Domain pack: contact centre

Calls and chats into the same home-warranty business, from its contact-centre platform. The customers and claims are the support/claims pack's (read, never changed), so a call about `CLM-000123-01` is about a real claim, and adding the pack changed none of the other numbers. It is built for testing webhook intake, customer lookup, intent, escalation and PII redaction against known answers.

| Table | What is in it |
|---|---|
| `contact_interactions` | one row per call or chat: channel, caller phone or chat email, account number keyed into the phone menu (if any), wait, answer, escalation and end times, outcome (handled / escalated / abandoned) |
| `contact_events` | the event stream behind each contact: `contact.queued`, `contact.answered`, each `message` (speaker and text), `contact.escalated`, `contact.ended` or `contact.abandoned` |

3,689 contacts and 37,020 events as of 2026-06-30 (2,866 handled, 370 escalated, 453 abandoned).

```python
from datetime import date
from saas_world_sim.packs import contact_centre_as_of, contact_centre_webhooks

tables = contact_centre_as_of(date(2026, 6, 30))   # contact_interactions, contact_events
deliveries = contact_centre_webhooks(date(2026, 6, 30))
deliveries[0]["type"], deliveries[0]["data"]["ivr_account_id"]   # ('contact.queued', 'CUS-002417')
```

`contact_centre_webhooks(day)` returns the webhook deliveries made during that day, in delivery order, as JSON-safe dicts. About 3% of events are delivered twice (same `event_id`, new `delivery_id`) and about 5% arrive 1 to 10 minutes late, after events that happened later. Dropping repeated `event_id`s and sorting by `(occurred_at, interaction_id, seq)` gives back the event table; a test checks that.

The platform does not know who called or why. Those answers are kept in `build_contact_centre_history()` as `true_account_id`, `true_intent`, `true_claim_id` and, per message, `pii_kinds`, and dropped from the as-of view.

**Planted patterns** (each checked by a test in `tests/test_contact_centre.py`):

- The customer can always be identified, but not always from the platform's fields: about 21% of contacts have neither a keyed-in account number nor a chat email on file, and the customer says the account number in their first message instead.
- Claim disputes follow denied claims and escalate 40% of the time, technician no-shows 12%, everything else 3-4%.
- Monday 09:00-12:00 the queue is longer (median wait 90 s vs 30 s) and 33% of contacts are abandoned, against 11% at other times.
- About half of technician no-show contacts come back within a few days.
- 12% of customer messages contain a phone number (555-0100 to 555-0199, set aside for fiction), an email (example.com) or a card number (Luhn-valid, in the 4000 0000 test range). The labels match the text exactly, and agents never repeat them.

`tests/test_contact_centre_fingerprint.py` locks the pack's row counts, outcomes and one day's deliveries.

## Install and use

```bash
pip install "saas-world-sim @ git+https://github.com/sky-five/saas-world-sim@v0.3.0"
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
pytest -q        # 81 tests: determinism, as-of view, API paging, planted patterns, webhook dedupe, fingerprints
```

CI runs lint and the full suite on every push.

## Licence

MIT
