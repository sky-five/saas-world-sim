"""Locks the contact-centre pack's history, like test_fingerprint.py does for the core world.
If one of these changes, downstream evals built on the pack change too: bump the package version."""
from datetime import date

from saas_world_sim.packs import contact_centre_as_of, contact_centre_webhooks


def test_row_counts():
    w = contact_centre_as_of(date(2026, 6, 30))
    assert {n: len(df) for n, df in w.items()} == {"contact_interactions": 3689, "contact_events": 37020}


def test_one_days_deliveries():
    d = contact_centre_webhooks(date(2026, 6, 30))
    assert (len(d), d[0]["delivery_id"], d[-1]["delivery_id"]) == (73, "CC-002417-001-01-d1", "CC-000047-003-06-d1")


def test_outcomes():
    i = contact_centre_as_of(date(2026, 6, 30))["contact_interactions"]
    assert i.outcome.value_counts().to_dict() == {"handled": 2866, "abandoned": 453, "escalated": 370}
