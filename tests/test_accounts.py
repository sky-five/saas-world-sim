from saas_world_sim.accounts import build_accounts


def test_accounts_volume_and_season():
    a = build_accounts()
    to_date = a[a.signup_at < "2026-10-08"]
    assert 7000 <= len(to_date) <= 9000
    m = to_date.signup_at.dt.month
    per_month_storm = m.between(4, 8).sum() / 5
    per_month_other = (~m.between(4, 8)).sum() / 7
    assert per_month_storm > 1.3 * per_month_other


def test_ids_unique_and_sorted():
    a = build_accounts()
    assert a.account_id.is_unique and a.signup_at.is_monotonic_increasing


def test_channel_shares_close_to_config():
    a = build_accounts()
    assert abs((a.channel == "google_ads").mean() - 0.30) < 0.02


def test_deterministic():
    assert build_accounts().equals(build_accounts())
