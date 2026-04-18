# multi-agent-pipeline

Three patterns for orchestrating multiple Claude agents: **sequential pipelines**, **parallel fan-out**, and **stochastic consensus**.

No framework, no graph DSL — just dataclasses and the Anthropic API. ~150 lines total.

## Why

Most "multi-agent" frameworks are overbuilt. In practice, 90% of multi-agent workflows are one of three things: run in sequence, run in parallel, or run the same task N times and take a vote. This repo is those three, clean.

## Install

```bash
git clone https://github.com/felixwickholm/multi-agent-pipeline
cd multi-agent-pipeline
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

## 1. Sequential pipeline

Each agent sees the previous output. Good for research → critique → synthesis.

```python
from pipeline import Agent, Pipeline

researcher = Agent(name="researcher", system_prompt="...")
critic = Agent(name="critic", system_prompt="...")
writer = Agent(name="writer", system_prompt="...")

results = Pipeline([researcher, critic, writer]).run("Your topic")
```

## 2. Parallel fan-out

All agents see the same input, run concurrently. Good for generating variations.

```python
from pipeline import Agent, Pipeline

copywriter = Agent(name="bold", system_prompt="Write bold, punchy copy.")
copywriter_b = Agent(name="subtle", system_prompt="Write subtle, classy copy.")
copywriter_c = Agent(name="data", system_prompt="Write data-driven copy.")

outputs = Pipeline.parallel(
    agents=[copywriter, copywriter_b, copywriter_c],
    shared_input="Landing page headline for a CRM product",
)
```

## 3. Stochastic consensus

Spawn N agents with varied framings, take a majority vote. Reduces single-call errors on classification tasks.

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
# {'consensus': 'sales', 'agreement': 0.80, 'distribution': {...}}
```

## Run the examples

```bash
python examples/research_pipeline.py
python examples/consensus_classifier.py
```

## Design choices

- **Parallel by default** — `Pipeline.parallel` and `StochasticConsensus` both use `ThreadPoolExecutor`
- **No LangChain, no graph compilation** — a Pipeline is a list of agents, full stop
- **Sync only** — async complicates debugging; parallelism is at the agent level via threads
- **Agreement score** — consensus returns not just the winner but how confident the ensemble was

## When to use which

| Pattern | Use when |
|---------|----------|
| Sequential | Each stage depends on the previous (research → write → edit) |
| Parallel | Independent variations of the same task (A/B copy, alternative plans) |
| Consensus | High-stakes classification where one agent might be wrong |

## License

MIT
