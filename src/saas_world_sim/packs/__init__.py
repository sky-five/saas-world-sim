"""Domain packs: smaller simulated businesses built on the same engine (keyed RNG, as-of view).

Each pack is separate from the core roofing-software world, so adding one never changes the core numbers.
"""
from saas_world_sim.packs.contact_centre import (
    CONTACT_CENTRE_TABLES,
    build_contact_centre_history,
    contact_centre_as_of,
    contact_centre_webhooks,
)
from saas_world_sim.packs.support_claims import (
    SUPPORT_CLAIMS_TABLES,
    build_support_claims_history,
    support_claims_as_of,
)

__all__ = [
    "CONTACT_CENTRE_TABLES",
    "build_contact_centre_history",
    "contact_centre_as_of",
    "contact_centre_webhooks",
    "SUPPORT_CLAIMS_TABLES", "build_support_claims_history", "support_claims_as_of"]
