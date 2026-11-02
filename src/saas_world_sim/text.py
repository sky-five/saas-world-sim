"""Support ticket text by theme. Templates are invented; slots are filled per ticket."""
import numpy as np

THEMES = ["measurement_accuracy", "billing_price", "missing_feature", "how_to", "bug_outage", "integration"]

_SUBJECTS = {
    "measurement_accuracy": [
        "Pitch on report is wrong", "Measurement off on {street}", "Report missing a facet",
        "Squares don't match my tape",
    ],
    "billing_price": [
        "Charged twice this month", "Question about invoice {inv}", "Price went up?", "Need a refund",
    ],
    "missing_feature": [
        "Can you add metal roofing?", "Need multi-crew scheduling", "Feature request: supplier pricing",
        "Any way to track gutters?",
    ],
    "how_to": [
        "How do I send a proposal?", "Adding a teammate", "Where is the job board?", "Changing my logo",
    ],
    "bug_outage": [
        "App won't load", "Proposal PDF blank", "Photos not uploading", "Error when signing",
    ],
    "integration": [
        "QuickBooks sync stopped", "Connect to my CRM?", "Zapier not triggering", "Calendar not syncing",
    ],
}

_BODIES = {
    "measurement_accuracy": [
        "The report for {street} shows a {p1}/12 pitch but it's {p2}/12. That changes my bid.",
        "Trees cover half the roof on {street} and the measurement looks short by about {sq} squares.",
        "Report is missing the back facet on {street}. Customer caught it before I did.",
        "Imagery for {street} looks a few years old, there's an addition now. Can you redo it?",
        "My tape says {sq} squares more than the report on {street}. Second time this month.",
        "Pitch came back {p1}/12 on {street}. Steep roof, it's at least {p2}/12.",
    ],
    "billing_price": [
        "I see two charges on invoice {inv}. Please fix.",
        "My bill went up and I didn't change anything. What happened?",
        "Honestly the add-ons are getting expensive for a small crew like mine.",
        "Can you explain invoice {inv}? I thought I was on the cheaper plan.",
        "Thinking about cancelling, it costs more than it saves me right now.",
        "Need a refund for the report on {street}, it was wrong.",
    ],
    "missing_feature": [
        "We do a lot of metal roofs and there's no way to price them.",
        "I have 3 crews and can't schedule them separately.",
        "Would be great to pull supplier prices into the proposal.",
        "Customers keep asking for gutters on the quote, I can't add them.",
        "Need better reports for my sales guys. Who closed what?",
        "If you added financing options I'd use this for every job.",
    ],
    "how_to": [
        "How do I send a proposal from my phone?",
        "Where do I add a new teammate?",
        "How do I change the logo on proposals?",
        "Can't find where to set my default margin.",
        "How do I order a faster report?",
        "Where do jobs go after the customer signs?",
    ],
    "bug_outage": [
        "App has been spinning for 10 minutes, can't load jobs.",
        "Proposal PDF comes out blank for {street}.",
        "Photos fail to upload from the roof, tried wifi and data.",
        "Customer got an error when signing the proposal.",
        "Lost my notes on the {street} job after the update.",
        "Getting logged out every few minutes today.",
    ],
    "integration": [
        "QuickBooks sync stopped on invoice {inv}.",
        "Is there a way to connect this to my CRM?",
        "Zapier trigger for new leads isn't firing.",
        "Jobs aren't showing on my Google Calendar anymore.",
        "Supplier order didn't go through from the app.",
        "Need my leads to sync both ways with my other tools.",
    ],
}

_STREETS = ["Maple Dr", "Oak St", "Ridge Rd", "Cedar Ln", "Pine Ave", "Hillcrest Way", "Lakeview Ct", "Elm St"]


def render_ticket(theme: str, rng: np.random.Generator) -> tuple[str, str]:
    slots = {
        "street": f"{rng.integers(10, 9999)} {rng.choice(_STREETS)}",
        "inv": f"INV-{rng.integers(10000, 99999)}",
        "p1": int(rng.integers(4, 8)),
        "p2": int(rng.integers(8, 12)),
        "sq": int(rng.integers(2, 9)),
    }
    subject = str(rng.choice(_SUBJECTS[theme])).format(**slots)
    body = str(rng.choice(_BODIES[theme])).format(**slots)
    return subject, body
