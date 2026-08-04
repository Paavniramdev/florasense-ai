"""
build_vector_store.py — chunks the raw Wikipedia docs, embeds them with a
local BGE model (no API key, no per-call cost), and stores everything in a
persistent ChromaDB collection ready for retrieval.

Usage:
    python build_vector_store.py --raw data/raw --db ./chroma_db
"""
import argparse
import glob
import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "florasense_knowledge"

# BGE models are trained expecting this instruction prefix on *queries* only
# (not on the passages being indexed) — it measurably improves retrieval
# quality and costs nothing to include.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120):
    """Simple sliding-window character chunker with sentence-boundary snapping.

    Character-based (not token-based) to avoid a tokenizer dependency here;
    800 chars / ~120 overlap keeps chunks small enough for good retrieval
    precision while still containing full sentences most of the time.
    """
    text = " ".join(text.split())  # normalize whitespace
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # snap to the nearest sentence end so we don't cut mid-sentence
            snap = text.rfind(". ", start, end)
            if snap != -1 and snap > start + chunk_size // 2:
                end = snap + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end - overlap > start else end
    return chunks


def load_raw_docs(raw_dir: str):
    docs = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        if path.endswith("_missing.json"):
            continue
        with open(path) as f:
            docs.append(json.load(f))
    return docs


def build_store(raw_dir: str, db_dir: str, batch_size: int = 64):
    docs = load_raw_docs(raw_dir)
    print(f"Loaded {len(docs)} raw species documents from {raw_dir}")
    if not docs:
        raise SystemExit("No raw docs found — run fetch_corpus.py first.")

    print(f"Loading embedding model ({EMBED_MODEL_NAME})... first run will download it.")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=db_dir)
    # Fresh build each run — simplest way to avoid stale/duplicate chunks
    # while the corpus is still being iterated on.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    all_chunks, all_metadatas, all_ids = [], [], []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append(
                {
                    "species": doc["species"],
                    "wiki_title": doc.get("wiki_title", doc["species"]),
                    "source_url": doc.get("url", ""),
                    "chunk_index": i,
                }
            )
            all_ids.append(f"{doc['species'].replace(' ', '_')}_{i}")

    print(f"Chunked into {len(all_chunks)} passages. Embedding in batches of {batch_size}...")

    for start in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[start : start + batch_size]
        batch_meta = all_metadatas[start : start + batch_size]
        batch_ids = all_ids[start : start + batch_size]

        embeddings = embedder.encode(batch_chunks, normalize_embeddings=True).tolist()

        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_chunks,
            metadatas=batch_meta,
        )
        print(f"  embedded {min(start + batch_size, len(all_chunks))}/{len(all_chunks)}")

    print(f"\nDone. Collection '{COLLECTION_NAME}' has {collection.count()} chunks stored at {db_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--db", default="./chroma_db")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    build_store(args.raw, args.db, args.batch_size)
