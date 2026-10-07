"""Minimal Brave Search API client: Web Search and LLM Context, with retries."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import requests

BASE_URL = "https://api.search.brave.com/res/v1"
RETRY_STATUSES = {429, 500, 502, 503, 504}


class BraveAPIError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"Brave API returned {status}: {body[:300]}")
        self.status = status
        self.body = body


@dataclass
class Source:
    url: str
    title: str
    snippets: list[str] = field(default_factory=list)
    age: str | None = None


@dataclass
class SearchResult:
    query: str
    sources: list[Source]
    latency_s: float

    def as_context(self, offset: int = 0) -> str:
        """Numbered context block an LLM can cite as [1], [2], ... starting after `offset`."""
        blocks = []
        for i, s in enumerate(self.sources, offset + 1):
            dated = f" ({s.age})" if s.age else ""
            body = "\n".join(f"- {snip}" for snip in s.snippets)
            blocks.append(f"[{i}] {s.title}{dated}\n{s.url}\n{body}")
        return "\n\n".join(blocks)

    def approx_tokens(self) -> int:
        return len(self.as_context()) // 4


class BraveClient:
    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 4,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No Brave API key. Set BRAVE_API_KEY or get a free key at "
                "https://api-dashboard.search.brave.com"
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()

    def _get(self, path: str, params: dict) -> dict:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        for attempt in range(self.max_retries + 1):
            resp = self.session.get(
                f"{BASE_URL}{path}", params=params, headers=headers, timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code not in RETRY_STATUSES or attempt == self.max_retries:
                raise BraveAPIError(resp.status_code, resp.text)
            time.sleep(self._backoff(resp, attempt))
        raise AssertionError("unreachable")

    @staticmethod
    def _backoff(resp: requests.Response, attempt: int) -> float:
        # Brave reports per-window values comma-separated, e.g. "1, 1000"; the first is the 1s window.
        reset = resp.headers.get("X-RateLimit-Reset", "")
        first = reset.split(",")[0].strip()
        if resp.status_code == 429 and first.isdigit():
            return max(float(first), 1.0)
        return min(2.0**attempt, 16.0)

    def web_search(self, query: str, count: int = 10, extra_snippets: bool = True) -> SearchResult:
        """Classic Web Search: titles plus short snippets, shaped for a results page."""
        start = time.perf_counter()
        data = self._get(
            "/web/search",
            {"q": query, "count": count, "extra_snippets": str(extra_snippets).lower()},
        )
        latency = time.perf_counter() - start
        sources = [
            Source(
                url=r["url"],
                title=r.get("title", ""),
                snippets=[r.get("description", "")] + r.get("extra_snippets", []),
                age=r.get("age"),
            )
            for r in data.get("web", {}).get("results", [])
        ]
        return SearchResult(query, sources, latency)

    def llm_context(
        self,
        query: str,
        count: int = 20,
        max_tokens: int = 8192,
        threshold: str | None = None,
        freshness: str | None = None,
    ) -> SearchResult:
        """LLM Context: pre-extracted page chunks ranked for a model, within a token budget."""
        params: dict = {"q": query, "count": count, "maximum_number_of_tokens": max_tokens}
        if threshold:
            params["context_threshold_mode"] = threshold
        if freshness:
            params["freshness"] = freshness
        start = time.perf_counter()
        data = self._get("/llm/context", params)
        latency = time.perf_counter() - start
        meta = data.get("sources", {})
        sources = []
        for item in data.get("grounding", {}).get("generic", []):
            age = meta.get(item["url"], {}).get("age") or []
            sources.append(
                Source(
                    url=item["url"],
                    title=item.get("title", ""),
                    snippets=item.get("snippets", []),
                    age=age[1] if len(age) > 1 else None,
                )
            )
        return SearchResult(query, sources, latency)
