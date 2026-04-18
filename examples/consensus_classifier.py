"""Stochastic consensus: 5 agents, different framings, majority vote.

Useful when a single classification call might be wrong. Spawn N agents
with different framings, take the majority answer.
"""

import re

from pipeline import StochasticConsensus

FRAMINGS = [
    "Approach this like a conservative lawyer who errs on safety.",
    "Approach this like a skeptical journalist looking for red flags.",
    "Approach this like a product manager focused on user intent.",
    "Approach this like a security engineer thinking about abuse.",
    "Approach this like an experienced sales rep reading buyer intent.",
]


def extract_label(text: str) -> str:
    """Extract the label from 'ANSWER: <label>' format."""
    match = re.search(r"ANSWER:\s*(\w+)", text)
    return match.group(1).lower() if match else "unknown"


if __name__ == "__main__":
    consensus = StochasticConsensus(
        framings=FRAMINGS,
        base_system=(
            "Classify the email as one of: SALES, SUPPORT, SPAM, PERSONAL.\n"
            "End your response with 'ANSWER: <label>' on its own line."
        ),
    )

    email = (
        "Hi, we're a 50-person SaaS and we're evaluating vendors for our "
        "customer support tooling. Can we jump on a 20-min call next week?"
    )

    result = consensus.run(email, extract=extract_label)

    print(f"Consensus: {result['consensus']}")
    print(f"Agreement: {result['agreement'] * 100:.0f}%")
    print(f"Distribution: {result['distribution']}")
