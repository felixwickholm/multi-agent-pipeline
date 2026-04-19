"""A 3-stage writing pipeline: drafter -> critic -> editor.

Each stage sees only the previous stage's output. No web access — the
drafter works from general knowledge. If you need grounded facts,
wire in a search tool (see claude-agent-loop for that pattern).
"""

from pipeline import Agent, Pipeline

drafter = Agent(
    name="drafter",
    system_prompt=(
        "You are a writer. Given a topic, produce a rough first draft in "
        "3-5 short paragraphs. Be concrete, use specific examples from "
        "your general knowledge, and don't hedge. Mark any claim you are "
        "less than certain about with [unverified] so downstream editors "
        "can fact-check or remove it."
    ),
)

critic = Agent(
    name="critic",
    system_prompt=(
        "You are a skeptical editor. Read the draft and flag: (1) claims "
        "that are weakly supported or feel made up, (2) wishy-washy "
        "phrasing that should be cut, (3) obvious structural issues. "
        "Return a numbered list of concrete changes — do not rewrite the "
        "draft yourself."
    ),
)

editor = Agent(
    name="editor",
    system_prompt=(
        "You are a final editor. You have been given the original draft "
        "and the critic's list of issues. Apply the critic's changes, "
        "remove anything marked [unverified], and return a tight "
        "publishable version. Output only the final text."
    ),
)


if __name__ == "__main__":
    pipeline = Pipeline(agents=[drafter, critic, editor], verbose=True)
    results = pipeline.run(
        "Write a 200-word executive brief on why most 'multi-agent' "
        "frameworks are overkill for the tasks they get used for."
    )

    for stage, output in results.items():
        print(f"\n{'=' * 60}\n{stage.upper()}\n{'=' * 60}")
        print(output)
