# 🌿 FloraSense AI — Intelligent Botanical Research Assistant

An end-to-end agentic AI system that identifies flower species from photos,
explains its reasoning visually, and answers grounded questions about care,
diseases, pollinators, and growing conditions — combining computer vision,
retrieval-augmented generation, and an LLM agent that decides which tools to
use for a given question.

> Built as a 4-week project: vision model → RAG knowledge base → LangGraph
> agent → Streamlit UI + analytics. Every component below was actually run
> and tested, not just designed on paper — see Results for real numbers.

---

## What it does

Ask something like *"What's wrong with my plant's leaves?"* with a photo
attached, and FloraSense:

1. **Identifies the species** from the image (EfficientNetV2-S, fine-tuned
   on the Oxford 102 Flowers dataset) — with a **calibrated confidence
   score**, so it tells you plainly when it's unsure rather than guessing
   confidently and being wrong.
2. **Shows its reasoning visually** via Grad-CAM — a heatmap over the parts
   of the image that drove the prediction.
3. **Retrieves real, sourced facts** about that species (diseases,
   pollinators, care, native range) from a knowledge base built from
   Wikipedia, rather than relying on the LLM's unverified general knowledge.
4. **Checks live weather/location** when the question depends on it (e.g.
   "can this grow here right now?").
5. **Reasons across all of that** to produce one grounded, cited answer —
   an LLM agent (via LangGraph) decides which tools to call and in what
   order, based on the actual question asked.

## Architecture

```
                          ┌─────────────────────┐
   User (Streamlit UI) ──▶│   LangGraph Agent    │
   photo + question       │   (Gemini Flash)     │
                          └──────────┬───────────┘
                                     │ decides which tool(s) to call
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                       ▼
     ┌─────────────────┐   ┌──────────────────┐    ┌──────────────────┐
     │ classify_flower  │   │ retrieve_knowledge│    │   get_weather     │
     │                  │   │                    │    │                  │
     │ FastAPI service  │   │ ChromaDB +         │    │ Open-Meteo API   │
     │ EfficientNetV2-S │   │ BGE embeddings     │    │ (free, no key)   │
     │ + Grad-CAM        │   │ (93 species,       │    │                  │
     │ + calibration     │   │  1,283 chunks,     │    │                  │
     │                  │   │  from Wikipedia)   │    │                  │
     └─────────────────┘   └──────────────────┘    └──────────────────┘
              │                      │                       │
              └──────────────────────┴───────────────────────┘
                                     ▼
                         Grounded, cited final answer
```

## Results (real, not projected)

| Component | Result |
|---|---|
| Classifier test accuracy | **85.3%** (102 species, Oxford Flowers dataset) |
| Validation accuracy | 90.1% |
| Confidence calibration | Temperature = 1.117 (near-neutral — model was already reasonably well-calibrated) |
| Knowledge base coverage | 93/102 species successfully sourced from Wikipedia, chunked into 1,283 passages |
| Agent tools | 3 (vision classifier, RAG retriever, weather) — LangGraph-orchestrated |
| Example real query | *"What's wrong with my plant?"* (uncertain AI-generated image) → correctly reported **24% confidence**, reasoned with appropriate caveats instead of stating a guess as fact |
| Example real query | *Bishop of Llandaff dahlia photo* → **69.9% confidence**, correct cultivar-level identification, cited historical + botanical facts |

*(Add your own screenshots/GIFs here — see "Demo" section below.)*

## Project structure

```
florasense-ai/
├── florasense_notebook/     # Week 1 — Colab training notebook
│   └── FloraSense_Week1_Vision_Module.ipynb
├── florasense_backend/      # Week 1-2 — FastAPI vision service
│   ├── app/
│   │   ├── main.py           # /predict, /health endpoints
│   │   ├── model_service.py  # inference + Grad-CAM + calibration
│   │   └── schemas.py
│   └── models/                # trained weights go here (not committed — see below)
├── florasense_rag/           # Week 2 — knowledge base
│   ├── fetch_corpus.py        # Wikipedia ingestion (rate-limited, resumable)
│   ├── build_vector_store.py  # chunk + embed + ChromaDB
│   └── query_test.py
├── florasense_agent/         # Week 3-4 — agent + UI
│   ├── tools.py                # classify_flower, retrieve_knowledge, get_weather
│   ├── graph.py                 # LangGraph ReAct loop
│   ├── main.py                  # CLI
│   ├── app.py                   # Streamlit UI
│   └── analytics_db.py          # SQLite logging + dashboard queries
└── README.md                  # you are here
```

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Vision model | EfficientNetV2-S (via `timm`), transfer learning | Strong accuracy/speed tradeoff, trains on free-tier Colab GPU |
| Explainability | Grad-CAM | Shows *why*, not just *what* |
| Calibration | Temperature scaling | Confidence scores that mean what they say |
| Backend | FastAPI | Fast, typed, auto-documented (`/docs`) |
| Embeddings | BGE (`bge-small-en-v1.5`), local | Free, no API cost, runs offline |
| Vector DB | ChromaDB | Simple, persistent, no server to manage |
| Agent framework | LangGraph | Explicit, debuggable tool-calling loop |
| LLM | Gemini (Flash-Lite) | Genuinely free tier, no card required — practical for a student project |
| Frontend | Streamlit | Fast to build, good enough for a real demo |
| Analytics | SQLite | Zero setup, sufficient at this scale |

## Setup

Each subfolder has its own README with exact steps — high-level order:

1. **Train the model** — run the Week 1 notebook in Colab (free T4 GPU),
   export `florasense_best.pt`, `idx_to_name.json`, `calibration.json`.
2. **Start the backend** — drop those 3 files into `florasense_backend/models/`,
   `pip install -r requirements.txt`, `uvicorn app.main:app --port 8000`.
3. **Build the knowledge base** — `python fetch_corpus.py`, then
   `python build_vector_store.py` in `florasense_rag/`.
4. **Get a free Gemini key** — https://aistudio.google.com/apikey (no card).
5. **Run the agent/UI** — copy `app.py` + `analytics_db.py` into
   `florasense_agent/`, set `GOOGLE_API_KEY` and `FLORASENSE_CHROMA_DB`,
   `streamlit run app.py`.

## Known limitations & honest notes

Being upfront about this matters more than pretending it's flawless:

- **Knowledge base coverage**: 9/102 species had no clean Wikipedia match
  (ambiguous common names like "orange dahlia" redirecting to a shared
  article) — logged in `_missing.json`, not silently dropped.
- **Corpus depth**: Wikipedia is a real but general-purpose source — a
  production system would layer in specialist sources (RHS, USDA PLANTS
  database, medicinal plant references) for deeper coverage on care/disease
  specifics.
- **Classifier scope**: trained on 102 species from a benchmark dataset, not
  a comprehensive global flora — it will confidently* (or honestly,
  low-confidently) misclassify anything outside that set.
- **Free-tier constraints**: Gemini's free tier has daily rate limits
  (varies by exact model — Flash-Lite is the most generous, ~1,000-1,500
  req/day at time of writing); fine for demo/dev use, not production traffic.
- **Not deployed publicly**: the full stack (vision model + ChromaDB + BGE
  embeddings + agent) is heavier than free hosting tiers comfortably serve
  — this repo is set up for local/demo running rather than a public URL.

## What I'd build next

- Multi-flower detection in a single image (YOLO-based)
- Specialist source ingestion beyond Wikipedia
- Manual title overrides for the 9 missed species
- A proper eval set (precision/recall on retrieval, not just vibes)
- Session memory so follow-up questions don't need re-stating context

## Demo

*(Record a 2-3 minute walkthrough and link it here. Suggested flow: show a
real photo → species ID + confidence badge + Grad-CAM → ask a follow-up
question that triggers RAG → show the agent reasoning trace expander → show
the analytics dashboard filling in live. This demonstrates every piece
working together, not just the happy path.)*

[Demo video link here]
