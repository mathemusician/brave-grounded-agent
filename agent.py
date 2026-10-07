"""A web-grounded research agent: the LLM decides when to search Brave, then answers with citations.

Usage:
    python agent.py "What changed in the EU AI Act timeline this year?"
    python agent.py --fresh pw "Latest Python release and its headline features"
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from brave_client import BraveClient

SYSTEM = """You answer questions using web search results from the Brave Search API.
- Call web_search whenever the answer depends on facts you are not certain of, or anything recent.
- You may search more than once with sharper queries if the first results are thin.
- Cite every factual claim with the source numbers shown in the results, like [2] or [1][3].
- Source numbers are global across all searches in this conversation.
- If the sources disagree or do not answer the question, say so plainly instead of guessing."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web via Brave. Returns numbered, pre-extracted page content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A focused search query."},
                },
                "required": ["query"],
            },
        },
    }
]


def make_llm() -> tuple[OpenAI, str]:
    """OpenRouter by default; any OpenAI-compatible endpoint works via LLM_BASE_URL."""
    client = OpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY"),
    )
    return client, os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4.5")


def run(question: str, freshness: str | None = None, max_steps: int = 4, verbose: bool = True) -> str:
    brave = BraveClient()
    llm, model = make_llm()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    sources_seen: list[str] = []

    for _ in range(max_steps):
        resp = llm.chat.completions.create(model=model, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return _with_reference_list(msg.content or "", sources_seen)

        for call in msg.tool_calls:
            query = json.loads(call.function.arguments)["query"]
            result = brave.llm_context(query, count=10, max_tokens=4096, freshness=freshness)
            if verbose:
                print(f"  searched: {query!r} -> {len(result.sources)} sources, "
                      f"~{result.approx_tokens()} tokens, {result.latency_s:.2f}s")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result.as_context(offset=len(sources_seen)) or "No relevant results.",
                }
            )
            sources_seen.extend(s.url for s in result.sources)

    return "Stopped after the step limit without a final answer."


def _with_reference_list(answer: str, urls: list[str]) -> str:
    cited = [(i, u) for i, u in enumerate(urls, 1) if f"[{i}]" in answer]
    if not cited:
        return answer
    refs = "\n".join(f"[{i}] {u}" for i, u in cited)
    return f"{answer}\n\nSources:\n{refs}"


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--fresh", help="Brave freshness filter: pd, pw, pm, py")
    args = parser.parse_args()
    print(run(args.question, freshness=args.fresh))
