# FloraSense AI — Agent (Week 3)

Ties the Week 1 vision model and Week 2 RAG knowledge base together with
Gemini, using LangGraph, so the assistant can handle real multi-step
questions like *"my sunflower leaves are yellow, what should I do?"*

**Using Gemini rather than Claude here specifically because Google AI
Studio has a genuine free API tier** — no credit card, no expiration —
which matters for a student/portfolio project where ongoing API costs
aren't practical during development.

## How it works

```
User question (+ optional image, location)
        │
        ▼
   ┌─────────┐   needs a tool?   ┌──────────────────────────┐
   │  agent  │ ────────────────► │ tools: classify_flower,  │
   │ (Gemini)│ ◄──────────────── │ retrieve_knowledge,      │
   └─────────┘   tool result     │ get_weather               │
        │                        └──────────────────────────┘
        │ no more tools needed
        ▼
   final grounded answer
```

This is a standard ReAct loop, built with LangGraph: Gemini decides which
tool(s) to call and in what order, based on the system prompt's guidance
(e.g. "if an image is attached, classify it first"). It can call multiple
tools in sequence before answering — e.g. classify the image, then look up
disease info for that specific species.

## Tools

| Tool | Wraps | Purpose |
|---|---|---|
| `classify_flower` | Week 1/2 FastAPI backend | Identify species from an image |
| `retrieve_knowledge` | Week 2 ChromaDB | Ground answers in real botanical facts |
| `get_weather` | Open-Meteo (free, no key) | Current conditions for location-based questions |

## Setup

```bash
pip install -r requirements.txt
```

You need **all three previous pieces running/available**:
1. The FastAPI backend from Week 2 (`uvicorn app.main:app`) — for `classify_flower`
2. The ChromaDB built in Week 2 (`chroma_db/`) — for `retrieve_knowledge`
3. A **free** Google AI Studio API key — for the agent's reasoning itself

Get one at https://aistudio.google.com/apikey — sign in with a Google
account, click "Create API key". No credit card, no billing setup.

Set your API key:
```bash
export GOOGLE_API_KEY=your-key-here      # Mac/Linux
set GOOGLE_API_KEY=your-key-here          # Windows cmd
```

Note: the free tier is rate-limited (currently ~15 requests/minute, ~1500/day
on Flash models) — plenty for development and demoing, but if you hit a 429
error, just wait a few seconds and retry.

By default the agent looks for:
- the backend at `http://localhost:8000` (override with `FLORASENSE_BACKEND_URL`)
- the vector store at `../florasense_rag/chroma_db` (override with `FLORASENSE_CHROMA_DB`)

If your folders are named/located differently, set those env vars before running.

## Run it

```bash
# Text-only question, no image
python main.py "What pollinators visit sunflowers?"

# With an image
python main.py "What's wrong with my plant?" --image path/to/flower.jpg

# With image + location (triggers the weather tool too)
python main.py "Can this grow here right now?" --image flower.jpg --location "Patiala, Punjab"
```

Verbose mode (default) prints each tool call and result as the agent makes
them, so you can see its reasoning step by step — useful for debugging and
also genuinely interesting to watch. Use `--quiet` to only print the final
answer.

## Tested vs. not-yet-tested

I tested the actual LangGraph routing logic (agent → tool → agent → final
answer, and the graph terminating correctly) using a fake LLM standing in
for Gemini, since that doesn't require a live API key. I also tested every
tool's error-handling path (missing image, backend not running, DB not
built) and fixed one real bug found this way: `retrieve_knowledge` was
loading the embedding model before checking the DB existed, causing an ugly
crash instead of a clean error message — now fixed.

What I could *not* test end-to-end here: an actual call to Gemini (needs
your API key) and the weather tool against the real Open-Meteo API (not
reachable from this sandboxed environment). Both use very standard,
well-documented APIs, so they should work — but they're worth watching
closely on your first real run.

## What's next (Week 4)

- Streamlit frontend that calls this agent
- Simple analytics logging (predictions, queries, latency) to SQLite
- Deploy to Hugging Face Spaces
