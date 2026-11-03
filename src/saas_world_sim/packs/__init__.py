"""Domain packs: smaller simulated businesses built on the same engine (keyed RNG, as-of view).

Each pack is separate from the core roofing-software world, so adding one never changes the core numbers.
"""
from saas_world_sim.packs.support_claims import (
    SUPPORT_CLAIMS_TABLES,
    build_support_claims_history,
    support_claims_as_of,
)

__all__ = ["SUPPORT_CLAIMS_TABLES", "build_support_claims_history", "support_claims_as_of"]
