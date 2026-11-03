"""Free text for the support and claims pack. Templates are invented; slots are filled per row."""
import numpy as np

FIRST_NAMES = ["Alex", "Sam", "Taylor", "Jamie", "Pat", "Robin", "Drew", "Chris", "Dana", "Lee",
               "Kim", "Terry", "Jesse", "Shawn", "Kelly", "Reese", "Blake", "Cameron", "Hayden", "Emerson"]
LAST_NAMES = ["Walker", "Hughes", "Price", "Bennett", "Foster", "Graham", "Reed", "Ellis", "Shaw", "Porter",
              "Hayes", "Fisher", "Barnes", "Cole", "Wells", "Lane", "Hart", "Fox", "Stone", "Webb"]

CLAIM_TEXT = {
    "refrigerator": ["Fridge stopped cooling overnight, lost a week of groceries.",
                     "Refrigerator is making a loud clicking noise and the freezer is warm.",
                     "Ice maker leaking water under the fridge, floor is getting soaked."],
    "washer": ["Washer won't drain, water sitting in the drum.",
               "Washing machine stops mid cycle with an error code.",
               "Washer is leaking from the bottom during the spin cycle."],
    "dryer": ["Dryer runs but no heat, clothes still wet after two cycles.",
              "Dryer drum won't turn, motor hums.",
              "Dryer shuts off after five minutes."],
    "dishwasher": ["Dishwasher not draining, standing water after every load.",
                   "Dishwasher door latch broke and it won't start.",
                   "Dishwasher leaking onto the kitchen floor."],
    "oven_range": ["Oven won't heat past 200 degrees.",
                   "Two burners on the range stopped working.",
                   "Oven igniter clicks but never lights."],
    "hvac": ["AC is blowing warm air, house is at 85 degrees.",
             "Furnace won't turn on, thermostat shows no response.",
             "Outdoor AC unit making a grinding noise and then shuts off."],
    "water_heater": ["No hot water since yesterday, water heater pilot won't stay lit.",
                     "Water heater tank is leaking from the bottom.",
                     "Hot water runs out after two minutes."],
    "plumbing": ["Main drain backed up into the basement.",
                 "Pipe under the kitchen sink burst.",
                 "Toilet keeps running and the shutoff valve is broken."],
    "electrical": ["Half the outlets in the living room went dead.",
                   "Breaker keeps tripping when the kitchen lights are on.",
                   "Light switch sparks when I flip it."],
    "roof_leak": ["Water coming through the ceiling in the upstairs bedroom after rain.",
                  "Roof leak over the garage, drywall is stained.",
                  "Shingles lifted in the wind and now there is a drip in the attic."],
    "pool_pump": ["Pool pump won't prime, water is turning green.",
                  "Pool pump motor is humming but not running.",
                  "Pool pump leaking at the seal."],
}

_SUBJECTS = {
    "claim_status": ["Status of claim {claim}", "Any update on {claim}?", "When will someone come out?"],
    "claim_dispute": ["Why was {claim} denied?", "Appeal for claim {claim}", "Denial makes no sense"],
    "how_to_file": ["How do I file a claim?", "What do I need to submit a claim?", "Filing a claim"],
    "billing_question": ["Charged twice this month", "Question about my bill", "Price went up?"],
    "plan_change": ["Upgrade my plan", "Does the higher plan cover my AC?", "Switching plans"],
    "technician_no_show": ["Technician never showed up", "Missed appointment", "Nobody came today"],
    "cancel_request": ["Cancel my plan", "Please cancel", "I want to cancel my coverage"],
}

_BODIES = {
    "claim_status": ["I filed {claim} a few days ago and have not heard anything. Can you check?",
                     "Is {claim} approved yet? My {item_name} is still broken.",
                     "Still waiting on {claim}. How long does this usually take?"],
    "claim_dispute": ["{claim} was denied and I don't understand why. I have paid every month.",
                      "You denied {claim} for my {item_name}. I want to appeal this decision.",
                      "This is the second time I've been told no. {claim} should be covered. I want a supervisor."],
    "how_to_file": ["My {item_name} broke. What do I need to send to file a claim?",
                    "Do I need a repair estimate before I file?",
                    "Can I file a claim by phone or only online?"],
    "billing_question": ["I see two charges on my card this month. Please fix.",
                         "My monthly price went up and I didn't change anything.",
                         "Can you explain the service fee on my last claim?"],
    "plan_change": ["I'd like to move to a plan that covers heating and cooling.",
                    "What's the difference between the plans? I have a pool now.",
                    "Can I downgrade to the basic plan to save money?"],
    "technician_no_show": ["The technician was booked for this morning and never came.",
                           "I took the day off work and nobody showed up for my {item_name}.",
                           "Second missed appointment this week."],
    "cancel_request": ["Please cancel my plan effective today.",
                       "I'm selling the house, please cancel my coverage.",
                       "Not worth the money for me anymore, cancel please."],
}

ITEM_NAMES = {
    "refrigerator": "fridge", "washer": "washer", "dryer": "dryer", "dishwasher": "dishwasher",
    "oven_range": "oven", "hvac": "AC", "water_heater": "water heater", "plumbing": "plumbing",
    "electrical": "wiring", "roof_leak": "roof", "pool_pump": "pool pump",
}


def fake_phone(rng: np.random.Generator) -> str:
    """555-0100 to 555-0199 are reserved for fiction, so these never reach a real line."""
    return f"555-01{int(rng.integers(0, 100)):02d}"


def render_claim(item: str, rng: np.random.Generator) -> str:
    return str(rng.choice(CLAIM_TEXT[item]))


def render_ticket(intent: str, rng: np.random.Generator, claim_id: str | None, item: str | None,
                  email: str, add_pii: bool) -> tuple[str, str]:
    slots = {"claim": claim_id or "my claim", "item_name": ITEM_NAMES.get(item or "", "appliance")}
    subject = str(rng.choice(_SUBJECTS[intent])).format(**slots)
    body = str(rng.choice(_BODIES[intent])).format(**slots)
    if add_pii:  # contact details customers paste into tickets; the redaction tests need them
        body += str(rng.choice([f" Call me at {fake_phone(rng)}.", f" You can reach me at {email}.",
                                f" My number is {fake_phone(rng)}, email {email}."]))
    return subject, body
