"""Conversation lines for the contact-centre pack. Templates are invented; slots are filled per contact."""
import numpy as np

from saas_world_sim.packs.support_claims_text import ITEM_NAMES

OPENERS = {
    "claim_status": ["Hi, I'm calling about {claim}. Has anything happened with it?",
                     "I filed a claim for my {item_name} and want to know where it stands. It's {claim}.",
                     "Checking on {claim}, nobody has called me back."],
    "claim_dispute": ["My claim {claim} got denied and I want to know why.",
                      "You turned down {claim} for my {item_name}. I want to appeal.",
                      "I've paid for this plan for months and {claim} was still denied. Get me a supervisor."],
    "how_to_file": ["My {item_name} stopped working. How do I start a claim?",
                    "What do I need to send in to file a claim?",
                    "Can I file a claim over the phone?"],
    "billing_question": ["I was charged twice this month.",
                         "Why did my monthly price go up?",
                         "There's a charge on my card I don't recognise."],
    "plan_change": ["I want to upgrade so my AC is covered.",
                    "Can I switch to a cheaper plan?",
                    "What does the premier plan add?"],
    "technician_no_show": ["The technician was supposed to come this morning and never showed.",
                           "Nobody came for my {item_name} repair today.",
                           "This is the second missed appointment."],
    "cancel_request": ["I'd like to cancel my plan.",
                       "We sold the house, please cancel the coverage.",
                       "Cancel my plan, it's not worth it for me."],
}

FOLLOW_UPS = {
    "claim_dispute": ["I sent the photos already.", "That's not what I was told when I signed up.",
                      "I want this escalated."],
    "technician_no_show": ["I took the day off for this.", "Can someone come tomorrow?"],
}
GENERIC_FOLLOW_UPS = ["Okay.", "How long will that take?", "Can you send that to me in writing?", "Thanks."]

AGENT_LINES = ["Thanks for calling, let me pull up your account.", "I can help with that.",
               "Let me check that for you.", "One moment while I look at the notes.",
               "I've added a note to your account."]
ESCALATION_LINES = ["I'm going to bring in a supervisor for this.", "Let me transfer you to our claims review team."]
CLOSING_LINES = ["Is there anything else I can help with?", "Thanks for your patience, have a good day."]


def luhn_digit(digits: str) -> str:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def fake_card(rng: np.random.Generator) -> str:
    """A Luhn-valid 16-digit number in the 4000 0000 test range; never a real card."""
    body = "40000000" + "".join(str(int(d)) for d in rng.integers(0, 10, 7))
    full = body + luhn_digit(body)
    return " ".join(full[i:i + 4] for i in range(0, 16, 4))


def opener(intent: str, rng: np.random.Generator, claim_id: str | None, item: str | None) -> str:
    slots = {"claim": claim_id or "my claim", "item_name": ITEM_NAMES.get(item or "", "appliance")}
    return str(rng.choice(OPENERS[intent])).format(**slots)


def follow_up(intent: str, rng: np.random.Generator) -> str:
    return str(rng.choice(FOLLOW_UPS.get(intent, GENERIC_FOLLOW_UPS)))
