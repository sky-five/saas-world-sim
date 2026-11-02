import json
from datetime import date, datetime

import pytest

from saas_world_sim.api import MockApi
from saas_world_sim.world import world_as_of

D = date(2024, 6, 30)


@pytest.fixture(scope="module")
def api():
    return MockApi(D)


def test_pages_cover_all_rows_once(api):
    rows = [r for page in api.fetch("support_tickets", page_size=500) for r in page]
    world = world_as_of(D)["support_tickets"]
    assert len(rows) == len(world)
    assert len({r["ticket_id"] for r in rows}) == len(rows)


def test_page_size_respected(api):
    pages = list(api.fetch("events", page_size=1000))
    assert len(pages) > 1 and all(len(p) <= 1000 for p in pages)


def test_updated_since_filters_strictly(api):
    cursor = datetime(2024, 6, 1)
    rows = [r for page in api.fetch("events", updated_since=cursor) for r in page]
    assert rows and all(r["updated_at"] > "2024-06-01T00:00:00" for r in rows)


def test_empty_page_stream_when_nothing_new(api):
    latest = max(r["updated_at"] for p in api.fetch("crm_calls") for r in p)
    assert list(api.fetch("crm_calls", updated_since=datetime.fromisoformat(latest))) == []


def test_json_safe(api):
    for table in ["contractor_leads", "subscriptions", "events"]:
        page = next(api.fetch(table, page_size=200))
        text = json.dumps(page)
        assert "NaN" not in text


def test_events_have_properties_dict(api):
    page = next(api.fetch("events", page_size=5000))
    props = [r["properties"] for r in page if r["event_type"] == "proposal_sent"]
    assert props and all(isinstance(p, dict) and p["value_usd"] > 0 for p in props)


def test_unknown_table(api):
    with pytest.raises(KeyError):
        next(api.fetch("nope"))


def test_arrow_pages_match_json_pages(api):
    arrow_rows = sum(t.num_rows for t in api.fetch_arrow("contractor_leads", page_size=10_000))
    json_rows = sum(len(p) for p in api.fetch("contractor_leads"))
    assert arrow_rows == json_rows > 0
