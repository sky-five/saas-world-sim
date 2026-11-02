"""Keyed random streams: the same keys always give the same numbers, in any process."""
import hashlib

import numpy as np

from saas_world_sim.config import SEED


def _stable_hash(key: str | int) -> int:
    return int.from_bytes(hashlib.blake2b(str(key).encode(), digest_size=8).digest(), "little")


def keyed_rng(*keys: str | int, seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *(_stable_hash(k) for k in keys)]))
