import subprocess
import sys

from saas_world_sim.rng import keyed_rng


def test_same_keys_same_stream():
    assert keyed_rng("acct", 7).random() == keyed_rng("acct", 7).random()


def test_different_keys_differ():
    assert keyed_rng("acct", 7).random() != keyed_rng("acct", 8).random()


def test_stable_across_processes():
    # fails if someone swaps the stable hash for Python's salted hash()
    code = "from saas_world_sim.rng import keyed_rng; print(repr(keyed_rng('acct', 1).random()))"
    out = subprocess.check_output([sys.executable, "-c", code])
    assert float(out) == keyed_rng("acct", 1).random()
