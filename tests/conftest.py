import pytest

from saas_world_sim.accounts import build_accounts
from saas_world_sim.lifecycle import simulate_account


def _histories(n, seed=1):
    a = build_accounts()
    a = a[a.signup_at < "2026-06-01"].sample(n, random_state=seed)
    return [simulate_account(r) for r in a.itertuples(index=False)]


@pytest.fixture(scope="session")
def histories():
    return _histories(2500)
