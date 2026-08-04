---
title: FloraSense AI
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# 🌿 FloraSense AI

An agentic botanical research assistant: upload a flower photo, get a
species identification with a calibrated confidence score, a Grad-CAM
explanation, and grounded answers (from a Wikipedia-sourced knowledge base)
about care, diseases, pollinators, and growing conditions — with an LLM
agent deciding which tools to use for each question.

**This is the merged single-process deployment build** — the vision model,
RAG retriever, and agent all run inside this one Streamlit app (rather than
as separate services), since that's what free-tier hosting like this
requires.

See the full project (training notebook, standalone FastAPI backend, and
architecture writeup) at: [link your GitHub repo here]

## Setup for deploying this Space yourself

1. Copy your trained model files into `models/`:
   `florasense_best.pt`, `idx_to_name.json`, `calibration.json`
2. Copy your built vector store folder in as `chroma_db/`
3. Add `GOOGLE_API_KEY` as a repository secret (Settings → Repository secrets)
   — get a free key at https://aistudio.google.com/apikey
4. Push — Spaces will build and launch automatically

## Notes

- Free tier is CPU-only, so classification + Grad-CAM will be slower than
  running locally on GPU (training) — expect a few extra seconds per query.
- The Space may sleep after inactivity and take ~30-60s to wake up on the
  next visit — that's normal free-tier behavior, not a bug.
- First query after a cold start will also pause briefly while the BGE
  embedding model downloads/loads.
