# brave-grounded-agent

A small, honest look at grounding LLM answers in live web results with the [Brave Search API](https://brave.com/search/api/).

Three scripts, each answering a question a developer asks when evaluating a search API for AI:

| Script | Question it answers |
|---|---|
| `agent.py` | Can an agent decide when to search, search more than once, and cite what it used? |
| `compare.py` | Should I ground on Web Search snippets or on the LLM Context endpoint? |
| `rerank.py` | If I re-rank results myself, does lexical, semantic, or hybrid ranking work best? |

## First successful call in 60 seconds

Get a free key at [api-dashboard.search.brave.com](https://api-dashboard.search.brave.com) (includes $5 of monthly credit, about 1,000 calls), then:

```bash
export BRAVE_API_KEY=your_key
curl -s "https://api.search.brave.com/res/v1/llm/context?q=what+is+retrieval+augmented+generation&maximum_number_of_tokens=2048" \
  -H "X-Subscription-Token: $BRAVE_API_KEY" | python3 -m json.tool | head -40
```

You get back `grounding.generic[]`: one entry per page with `url`, `title`, and `snippets`, already extracted and sized for a model. No scraping step.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add BRAVE_API_KEY and an OpenRouter (or any OpenAI-compatible) key
pytest -q              # client tests run offline, no key needed
```

## 1. A research agent with citations (`agent.py`)

```bash
python agent.py "Which US states passed comprehensive privacy laws in 2026?"
python agent.py --fresh pw "Latest stable Python release and headline features"
```

The model gets one tool, `web_search`, backed by the LLM Context endpoint. It decides when to call it, can refine its query and search again, and must cite every claim. Citation numbers stay unique across searches, and the final answer ends with only the sources it actually cited.

Any OpenAI-compatible model works through `LLM_BASE_URL` and `LLM_MODEL`; the default is Claude through OpenRouter.

## 2. Web Search vs LLM Context (`compare.py`)

```bash
python compare.py            # sources, context size, latency (10 API calls)
python compare.py --answer   # also answers each question from each context
```

Web Search is shaped for a results page: a title and a short description per result (plus up to 5 `extra_snippets`). LLM Context is shaped for a model: relevance-selected chunks from the page body, inside a token budget you set. The script runs both on the same questions and writes `results/comparison.md`, so the trade-off is measured rather than assumed.

## 3. Lexical vs semantic vs hybrid ranking (`rerank.py`)

```bash
python rerank.py "how do I stop a python script that hangs forever"
python rerank.py "ERR_SSL_PROTOCOL_ERROR chrome fix"
```

Pulls chunks from Brave, then ranks them three ways:

- **BM25 (lexical):** scores exact word overlap. Strong for names, error codes, and version numbers, where the exact string matters.
- **Dense embeddings (semantic):** scores meaning. Finds "terminate the process" when you asked how to "stop" it.
- **Reciprocal rank fusion (hybrid):** merges the two rankings.

A worked example on five hand-written chunks for the first query:

| Rank | BM25 | Dense | Hybrid |
|---|---|---|---|
| 1 | "How to **hang** a picture frame on drywall" | "Python scripts can be stopped with Ctrl+C" | "How to hang a picture frame..." |
| 2 | "Use signal.alarm or a timeout to terminate..." | "subprocess.run(timeout=10) kills the child..." | "Python scripts can be stopped with Ctrl+C" |

BM25 rewards the word "hang" even though the chunk is about drywall. Dense retrieval understands the question. Hybrid still keeps the bad chunk at the top, because with a tiny candidate set one confident wrong ranking dominates. Two takeaways: run the error-code query too, since exact strings are where lexical ranking usually earns its place; and hybrid is a good default, not a guarantee. The first-stage retrieval (Brave's index) matters most, which is the argument for a good search API in the first place.

## Notes from building this

Things I would tell another developer, or feed back to an API team:

- **Pick the endpoint by who reads the output.** If a model is the reader, start with LLM Context; it removes the scrape-and-chunk step entirely. Web Search is better when you need the results-page view (titles, ranking order, ages).
- **Rate-limit headers are per window and comma-separated** (for example `X-RateLimit-Reset: 1, 86400`). The first value is the 1-second window, which is the one to back off on. `brave_client.py` reads it on a 429 instead of guessing.
- **Pin `Api-Version` in production.** The LLM Context extraction pipeline changed on 2026-07-31; pinning keeps output stable while you evaluate the new one.
- **Snippets can be JSON-serialized tables or code**, not only prose. Models handle it, but post-processing code should not assume plain text.
- **Empty `grounding.generic` is a valid answer**, not an error. The agent tells the model "No relevant results" so it can say so instead of inventing sources.
- **Token budget is the main lever on latency and cost.** The agent uses 4,096 tokens per search; the docs suggest about 2,048 for simple facts and 16,384 for deep research.

## Files

```
brave_client.py   Web Search + LLM Context client: retries, backoff, typed results
agent.py          Tool-calling research agent with citations
compare.py        Web Search vs LLM Context on the same questions
rerank.py         BM25 vs dense vs hybrid re-ranking
tests/            Offline tests for parsing, numbering, retries, and errors
```
