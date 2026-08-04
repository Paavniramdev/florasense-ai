# FloraSense AI — RAG Knowledge Base (Week 2, part 2)

Builds the retrieval layer that grounds the assistant's answers about
diseases, pollinators, care, native range, and uses — instead of it just
making things up.

## Pipeline

```
fetch_corpus.py  →  data/raw/*.json  →  build_vector_store.py  →  ChromaDB  →  query_test.py
   (Wikipedia)        (raw text)         (chunk + embed)          (storage)     (test retrieval)
```

### Why Wikipedia as the base corpus

It's real, freely-licensed (CC BY-SA — safe to store and reuse with
attribution), and plant articles typically already cover Description,
Cultivation, Uses, and Pests & diseases — which maps directly onto the
question types FloraSense needs to answer. Every stored chunk keeps its
source URL, so answers can always be traced back and cited.

**This is a starting corpus, not a finished one.** For a stronger/more
authoritative knowledge base later, add more fetch functions for
specialist sources (RHS, USDA PLANTS database, medicinal plant references)
— the fetch → chunk → embed → store pattern stays identical either way.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Fetch the raw corpus

```bash
# Uses your Week 1 idx_to_name.json for the species list (recommended —
# guarantees the RAG corpus matches exactly what your classifier can predict)
python fetch_corpus.py --labels ../backend/models/idx_to_name.json --out data/raw

# Or, with no labels file, it fetches the official 102 species names itself
python fetch_corpus.py --out data/raw
```

This takes a few minutes (rate-limited to be polite to Wikipedia's API — one
request every 0.5s). It's resume-safe: if it gets interrupted, just re-run
the same command and it'll skip species already saved.

Species with no good Wikipedia match get logged to `data/raw/_missing.json`
— check that file afterward; a handful of oddly-named species (~5-10 out of
102) commonly won't match and may need a manual title override or a
hand-written fallback doc.

## 3. Build the vector store

```bash
python build_vector_store.py --raw data/raw --db ./chroma_db
```

First run downloads the embedding model (`BAAI/bge-small-en-v1.5`, ~130MB,
runs locally — no API key, no per-query cost). This chunks each article into
~800-character passages and embeds them all into a persistent ChromaDB
collection at `./chroma_db`.

## 4. Test retrieval

```bash
python query_test.py "what diseases affect sunflowers"
python query_test.py --k 3 "which flowers attract bees"
python query_test.py "medicinal uses of chamomile"
```

You should see the most relevant passages come back with their source
species and a similarity score. If results look off-topic, the two usual
causes are: (a) that species' Wikipedia article was thin/missing sections
Wikipedia doesn't cover for it, or (b) the question is about something no
article discusses (e.g. very specific regional growing advice) — which is
exactly the gap the agent will need to fill with reasoning on top of
retrieved facts in Week 3, rather than something retrieval alone can fix.

## What's next (Week 3)

This retriever becomes one of the tools the LangGraph agent can call —
alongside the vision classifier and a weather API — so a question like *"my
sunflower leaves are yellow, what should I do?"* triggers: classify the
image → retrieve disease-related passages for that species → reason over
both to produce a grounded answer.
