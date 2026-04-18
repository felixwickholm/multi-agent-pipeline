"""A 3-stage research pipeline: researcher -> critic -> synthesizer."""

from pipeline import Agent, Pipeline

researcher = Agent(
    name="researcher",
    system_prompt="You are a research assistant. Given a topic, list 5 key facts with sources.",
)

critic = Agent(
    name="critic",
    system_prompt=(
        "You are a skeptical editor. Read the facts and flag any that are weakly "
        "sourced, outdated, or oversimplified. Return an updated list."
    ),
)

synthesizer = Agent(
    name="synthesizer",
    system_prompt=(
        "You are a writer. Turn the vetted facts into a tight 3-paragraph brief "
        "suitable for an executive audience."
    ),
)


if __name__ == "__main__":
    pipeline = Pipeline(agents=[researcher, critic, synthesizer], verbose=True)
    results = pipeline.run("The state of AI agent frameworks in 2026")

    for stage, output in results.items():
        print(f"\n{'=' * 60}\n{stage.upper()}\n{'=' * 60}")
        print(output)
