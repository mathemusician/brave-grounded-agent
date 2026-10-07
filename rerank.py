"""Lexical vs semantic vs hybrid ranking over the same Brave results.

Pulls page chunks from the LLM Context endpoint, then re-ranks them three ways for a query:
  - BM25 (lexical): rewards exact term overlap. Great for names, error codes, version numbers.
  - Dense embeddings (semantic): rewards meaning. Finds "terminate" when you asked "stop".
  - Reciprocal rank fusion (hybrid): combines both rankings, which is what most RAG systems ship.

Usage:
    python rerank.py "how do I stop a python script that hangs forever"
    python rerank.py "ERR_SSL_PROTOCOL_ERROR chrome fix"
"""

from __future__ import annotations

import argparse
import re

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from brave_client import BraveClient

TOKEN = re.compile(r"[a-z0-9_]+")
MARKDOWN_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|])")


def tokenize(text: str) -> list[str]:
    # LLM Context snippets are Markdown, so ERR_SSL_PROTOCOL_ERROR often arrives as ERR\_SSL\_PROTOCOL\_ERROR.
    return TOKEN.findall(MARKDOWN_ESCAPE.sub(r"\1", text).lower())


def bm25_rank(query: str, chunks: list[str]) -> list[int]:
    scores = BM25Okapi([tokenize(c) for c in chunks]).get_scores(tokenize(query))
    return list(np.argsort(-scores))


def dense_rank(query: str, chunks: list[str], model: TextEmbedding) -> list[int]:
    q = np.array(next(iter(model.query_embed([query]))))
    docs = np.array(list(model.passage_embed(chunks)))
    sims = docs @ q / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q))
    return list(np.argsort(-sims))


def rrf(*rankings: list[int], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return sorted(scores, key=scores.get, reverse=True)


def main(query: str, top: int) -> None:
    result = BraveClient().llm_context(query, count=20, max_tokens=8192, threshold="disabled")
    chunks, origin = [], []
    for s in result.sources:
        for snip in s.snippets:
            chunks.append(" ".join(snip.split()))
            origin.append(s.url)
    if len(chunks) < 2:
        print("Not enough chunks returned to compare rankings.")
        return

    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    lexical = bm25_rank(query, chunks)
    semantic = dense_rank(query, chunks, model)
    hybrid = rrf(lexical, semantic)

    print(f"Query: {query}\n{len(chunks)} chunks from {len(result.sources)} pages\n")
    for name, ranking in [("BM25 (lexical)", lexical), ("Dense (semantic)", semantic), ("Hybrid (RRF)", hybrid)]:
        print(f"== {name} ==")
        for idx in ranking[:top]:
            print(f"  {chunks[idx][:140]}...\n    <- {origin[idx]}")
        print()

    overlap = len(set(lexical[:top]) & set(semantic[:top]))
    print(f"BM25 and dense agree on {overlap}/{top} of their top {top} chunks.")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    main(args.query, args.top)
