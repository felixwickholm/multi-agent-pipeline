# multi-agent-pipeline

Three orchestration patterns for Claude agents: **sequential pipelines**, **parallel fan-out**, and **stochastic consensus**. Built on the Anthropic SDK directly.

![tests](https://github.com/felixwickholm/multi-agent-pipeline/actions/workflows/tests.yml/badge.svg)

## Why

Most "multi-agent" workflows reduce to one of three primitives: run a chain, run a fan-out, or run the same thing N times and vote. This repo gives you those three, cleanly separated, with error isolation so one agent's failure doesn't kill the others.

## Install

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
```

## 1. Sequential pipeline

Each agent sees the previous output. Useful for draft → critique → finalize.

```python
from pipeline import Agent, Pipeline

drafter  = Agent(name="drafter",  system_prompt="...")
critic   = Agent(name="critic",   system_prompt="...")
finalize = Agent(name="finalize", system_prompt="...")

results = Pipeline([drafter, critic, finalize]).run("Your topic")
```

If an agent raises, the pipeline stops and the error is stored under that agent's name. Pass `stop_on_error=False` to let downstream agents see the error string and continue.

## 2. Parallel fan-out

All agents see the same input, run concurrently, merge to a dict.

```python
copy_a = Agent(name="bold",   system_prompt="Bold, punchy copy.")
copy_b = Agent(name="subtle", system_prompt="Subtle, classy copy.")
copy_c = Agent(name="data",   system_prompt="Data-driven copy.")

outputs = Pipeline.parallel([copy_a, copy_b, copy_c], "Landing page headline for a CRM product")
```

One agent's exception never kills the others — failed agents show up with an `ERROR:` prefix in the result.

## 3. Stochastic consensus

Spawn N agents with varied framings, take a majority vote. Reduces single-call errors on classification tasks where one agent might be wrong.

```python
from pipeline import StochasticConsensus

classifier = StochasticConsensus(
    framings=[
        "Approach this like a conservative lawyer.",
        "Approach this like a skeptical journalist.",
        "Approach this like a product manager.",
    ],
    base_system="Classify as SALES, SUPPORT, SPAM, or PERSONAL. End with 'ANSWER: <label>'.",
)

result = classifier.run(email_text, extract=my_label_extractor)
# {
#   "consensus": "sales",
#   "tied": None,
#   "agreement": 0.67,
#   "distribution": {"sales": 2, "support": 1},
#   ...
# }
```

Ties are reported explicitly via the `tied` field — you decide how to handle them.

## Run the tests

```bash
pytest -q
```

Tests use a `FakeAgent` subclass so they run offline. They cover: sequential output passing, stop-on-error behavior, parallel fan-out failure isolation, consensus majority logic, and tie detection.

## Run the examples

```bash
python examples/research_pipeline.py      # drafter → critic → editor
python examples/consensus_classifier.py   # 5 framings vote on one email
```

## When to use which

| Pattern    | Use when                                                                     |
| ---------- | ---------------------------------------------------------------------------- |
| Sequential | Each stage depends on the previous (draft → critique → finalize)             |
| Parallel   | Independent variations of the same task (A/B copy, alternative plans)        |
| Consensus  | High-stakes classification where one agent might be wrong                    |

## Design notes

- **Thread-based parallelism, not async** — simpler stack traces, easier to debug.
- **Error isolation by default** — one bad API call never crashes an ensemble.
- **Explicit tie handling** — consensus returns the `tied` list when there is no clear winner.
- **No graph DSL** — a Pipeline is a list of agents.

## License

MIT
