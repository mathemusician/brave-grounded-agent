"""Web Search vs LLM Context as grounding for the same questions.

Measures what a developer evaluating the API cares about: sources returned, context size,
latency, and (with --answer) whether the model can actually answer from it.

Usage:
    python compare.py                 # retrieval metrics only (2 API calls per question)
    python compare.py --answer        # also answer each question from each context
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from brave_client import BraveClient, SearchResult

QUESTIONS = [
    "What is the current stable version of PostgreSQL and its headline feature?",
    "How do I paginate results with the Stripe API in Python?",
    "What are the rate limits for the GitHub REST API for authenticated users?",
    "Difference between BM25 and dense retrieval for RAG",
    "Which US states have passed comprehensive consumer privacy laws?",
]

ANSWER_PROMPT = """Answer the question using only the sources below. Cite sources like [1].
If the sources do not contain the answer, reply exactly: NOT IN SOURCES.

Question: {q}

Sources:
{ctx}"""


def answer(question: str, result: SearchResult) -> str:
    from agent import make_llm

    llm, model = make_llm()
    resp = llm.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": ANSWER_PROMPT.format(q=question, ctx=result.as_context())}],
    )
    return (resp.choices[0].message.content or "").strip()


def main(with_answers: bool) -> None:
    brave = BraveClient()
    rows = []
    report = ["# Web Search vs LLM Context\n"]

    for q in QUESTIONS:
        web = brave.web_search(q, count=10)
        ctx = brave.llm_context(q, count=10, max_tokens=4096)
        rows.append((q, web, ctx))
        print(f"{q[:60]:<60}  web: {len(web.sources):>2} src ~{web.approx_tokens():>5} tok "
              f"{web.latency_s:.2f}s | llm_context: {len(ctx.sources):>2} src "
              f"~{ctx.approx_tokens():>5} tok {ctx.latency_s:.2f}s")

    report.append("| Question | Web: sources / ~tokens / latency | LLM Context: sources / ~tokens / latency |")
    report.append("|---|---|---|")
    for q, web, ctx in rows:
        report.append(
            f"| {q} | {len(web.sources)} / {web.approx_tokens()} / {web.latency_s:.2f}s "
            f"| {len(ctx.sources)} / {ctx.approx_tokens()} / {ctx.latency_s:.2f}s |"
        )

    if with_answers:
        report.append("\n## Answers from each context\n")
        for q, web, ctx in rows:
            report.append(f"### {q}\n")
            report.append(f"**From Web Search snippets:**\n\n{answer(q, web)}\n")
            report.append(f"**From LLM Context:**\n\n{answer(q, ctx)}\n")

    out = Path("results/comparison.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(report) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answer", action="store_true", help="also generate answers (uses LLM credits)")
    main(parser.parse_args().answer)
