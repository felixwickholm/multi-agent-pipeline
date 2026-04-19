"""Tests for Pipeline, parallel fan-out, and StochasticConsensus.

Uses a fake Agent subclass — no network calls.
"""

from __future__ import annotations

from unittest.mock import patch

from pipeline import Agent, Pipeline, StochasticConsensus


class FakeAgent(Agent):
    """Agent whose .run() returns a scripted value (or raises)."""

    def __init__(self, name: str, output):
        super().__init__(name=name, system_prompt="ignored")
        self._output = output

    def run(self, _user_message: str) -> str:
        if isinstance(self._output, Exception):
            raise self._output
        return self._output


def test_pipeline_passes_output_forward():
    a = FakeAgent("a", "A said: foo")
    b = FakeAgent("b", "B got: whatever")
    results = Pipeline([a, b]).run("start")
    assert results["a"] == "A said: foo"
    assert results["b"] == "B got: whatever"


def test_pipeline_stops_on_error():
    a = FakeAgent("a", "ok")
    b = FakeAgent("b", RuntimeError("boom"))
    c = FakeAgent("c", "never runs")
    results = Pipeline([a, b, c]).run("start")
    assert results["a"] == "ok"
    assert "ERROR" in results["b"]
    assert "c" not in results


def test_pipeline_continues_when_stop_on_error_false():
    a = FakeAgent("a", "ok")
    b = FakeAgent("b", RuntimeError("boom"))
    c = FakeAgent("c", "still ran")
    results = Pipeline([a, b, c], stop_on_error=False).run("start")
    assert results["c"] == "still ran"


def test_parallel_fanout_isolates_failures():
    good = FakeAgent("good", "ok")
    bad = FakeAgent("bad", RuntimeError("fail"))
    results = Pipeline.parallel([good, bad], "x")
    assert results["good"] == "ok"
    assert results["bad"].startswith("ERROR")


def _fixed_consensus(answers: list[str]):
    """Patch StochasticConsensus to yield pre-canned answers."""
    consensus = StochasticConsensus(
        framings=["f"] * len(answers),
        base_system="s",
    )

    def fake_run(self, _question, extract=None):
        extracted = [extract(a) for a in answers] if extract else list(answers)
        from collections import Counter

        counts = Counter(e for e in extracted if not e.startswith("ERROR"))
        top = counts.most_common()
        winner, wc = top[0]
        tied = [l for l, c in top if c == wc]
        return {
            "consensus": winner,
            "tied": tied if len(tied) > 1 else None,
            "agreement": wc / len(answers),
            "all_answers": answers,
            "extracted": extracted,
            "distribution": dict(counts),
        }

    return consensus, fake_run


def test_consensus_picks_majority():
    consensus, fake_run = _fixed_consensus(["yes", "yes", "no", "yes", "no"])
    with patch.object(StochasticConsensus, "run", fake_run):
        r = consensus.run("q")
    assert r["consensus"] == "yes"
    assert r["distribution"]["yes"] == 3
    assert r["agreement"] == 0.6


def test_consensus_flags_ties():
    consensus, fake_run = _fixed_consensus(["a", "b", "a", "b"])
    with patch.object(StochasticConsensus, "run", fake_run):
        r = consensus.run("q")
    assert set(r["tied"]) == {"a", "b"}
    assert r["agreement"] == 0.5
