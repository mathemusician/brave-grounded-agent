from unittest.mock import MagicMock

import pytest

from brave_client import BraveAPIError, BraveClient


def fake_response(status: int, json_body: dict | None = None, headers: dict | None = None, text: str = ""):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_body or {}
    resp.headers = headers or {}
    resp.text = text
    return resp


def make_client(*responses):
    session = MagicMock()
    session.get.side_effect = list(responses)
    return BraveClient(api_key="test", session=session), session


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("brave_client.time.sleep", lambda s: None)


LLM_CONTEXT_BODY = {
    "grounding": {
        "generic": [
            {"url": "https://a.com", "title": "A", "snippets": ["alpha one", "alpha two"]},
            {"url": "https://b.com", "title": "B", "snippets": ["beta"]},
        ],
        "map": [],
    },
    "sources": {
        "https://a.com": {"title": "A", "hostname": "a.com", "age": ["Mon", "2026-09-01", "1 month ago"]},
        "https://b.com": {"title": "B", "hostname": "b.com", "age": []},
    },
}


def test_llm_context_parses_grounding_and_dates():
    client, session = make_client(fake_response(200, LLM_CONTEXT_BODY))
    result = client.llm_context("q", threshold="strict")

    assert [s.url for s in result.sources] == ["https://a.com", "https://b.com"]
    assert result.sources[0].snippets == ["alpha one", "alpha two"]
    assert result.sources[0].age == "2026-09-01"
    assert result.sources[1].age is None
    params = session.get.call_args.kwargs["params"]
    assert params["context_threshold_mode"] == "strict"
    assert session.get.call_args.kwargs["headers"]["X-Subscription-Token"] == "test"


def test_context_numbering_continues_after_offset():
    client, _ = make_client(fake_response(200, LLM_CONTEXT_BODY))
    ctx = client.llm_context("q").as_context(offset=3)
    assert ctx.startswith("[4] A")
    assert "\n\n[5] B" in ctx


def test_web_search_merges_description_and_extra_snippets():
    body = {"web": {"results": [{"url": "https://x.com", "title": "X", "description": "main",
                                 "extra_snippets": ["more"], "age": "2 days ago"}]}}
    client, session = make_client(fake_response(200, body))
    result = client.web_search("q")
    assert result.sources[0].snippets == ["main", "more"]
    assert session.get.call_args.kwargs["params"]["extra_snippets"] == "true"


def test_retries_on_rate_limit_then_succeeds():
    client, session = make_client(
        fake_response(429, headers={"X-RateLimit-Reset": "1, 86400"}),
        fake_response(200, LLM_CONTEXT_BODY),
    )
    result = client.llm_context("q")
    assert len(result.sources) == 2
    assert session.get.call_count == 2


def test_rate_limit_backoff_uses_first_window():
    resp = fake_response(429, headers={"X-RateLimit-Reset": "3, 86400"})
    assert BraveClient._backoff(resp, attempt=0) == 3.0


def test_does_not_retry_auth_errors():
    client, session = make_client(fake_response(401, text="invalid token"))
    with pytest.raises(BraveAPIError) as err:
        client.llm_context("q")
    assert err.value.status == 401
    assert session.get.call_count == 1


def test_gives_up_after_max_retries():
    client, session = make_client(*[fake_response(503) for _ in range(5)])
    with pytest.raises(BraveAPIError):
        client.llm_context("q")
    assert session.get.call_count == 5


def test_missing_key_explains_where_to_get_one(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="api-dashboard.search.brave.com"):
        BraveClient()
