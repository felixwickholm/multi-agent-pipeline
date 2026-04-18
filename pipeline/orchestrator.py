"""Multi-agent orchestration patterns: parallel, sequential, and consensus.

Three patterns that cover 90% of multi-agent use cases:
- Pipeline: sequential agents, each sees the previous output
- Parallel (via Pipeline.parallel): fan out, independent agents, merge results
- StochasticConsensus: spawn N agents with different framings, aggregate

All built on the Anthropic messages API directly — no LangChain.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from anthropic import Anthropic


@dataclass
class Agent:
    name: str
    system_prompt: str
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        self.client = Anthropic()

    def run(self, user_message: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return "\n".join(b.text for b in response.content if b.type == "text")


@dataclass
class Pipeline:
    """Chain agents so each one processes the previous output."""

    agents: list[Agent]
    verbose: bool = False

    def run(self, initial_input: str) -> dict[str, str]:
        results: dict[str, str] = {}
        current = initial_input
        for agent in self.agents:
            if self.verbose:
                print(f"▸ {agent.name} running...")
            output = agent.run(current)
            results[agent.name] = output
            current = output
        return results

    @staticmethod
    def parallel(
        agents: list[Agent],
        shared_input: str,
        max_workers: int = 5,
    ) -> dict[str, str]:
        """Fan out: all agents see the same input, run in parallel."""
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(a.run, shared_input): a for a in agents}
            return {futures[f].name: f.result() for f in futures}


@dataclass
class StochasticConsensus:
    """Spawn N agents with varied framings, aggregate their answers.

    Majority vote for categorical outputs. Useful for classification,
    routing decisions, and any task where one call could be wrong.
    """

    framings: list[str]
    base_system: str = "You are a careful analyst."
    model: str = "claude-sonnet-4-6"

    def run(
        self,
        question: str,
        extract: Callable[[str], str] | None = None,
    ) -> dict[str, Any]:
        agents = [
            Agent(
                name=f"agent_{i}",
                system_prompt=f"{self.base_system}\n\n{framing}",
                model=self.model,
            )
            for i, framing in enumerate(self.framings)
        ]

        with ThreadPoolExecutor(max_workers=len(agents)) as pool:
            raw_answers = list(pool.map(lambda a: a.run(question), agents))

        if extract:
            extracted = [extract(a) for a in raw_answers]
        else:
            extracted = raw_answers

        counts = Counter(extracted)
        winner, winner_count = counts.most_common(1)[0]

        return {
            "consensus": winner,
            "agreement": winner_count / len(agents),
            "all_answers": raw_answers,
            "extracted": extracted,
            "distribution": dict(counts),
        }
