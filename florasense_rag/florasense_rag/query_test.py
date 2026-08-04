"""
query_test.py — quick sanity-check tool for the retriever, before wiring it
into the LangGraph agent in Week 3.

Usage:
    python query_test.py --db ./chroma_db "What pollinators visit sunflowers?"
    python query_test.py --db ./chroma_db --k 3 "diseases that affect roses"
"""
import argparse

import chromadb
from sentence_transformers import SentenceTransformer

from build_vector_store import EMBED_MODEL_NAME, COLLECTION_NAME, BGE_QUERY_PREFIX


def retrieve(question: str, db_dir: str, k: int = 5):
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(COLLECTION_NAME)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    query_embedding = embedder.encode([BGE_QUERY_PREFIX + question], normalize_embeddings=True).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=k)

    hits = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        hits.append({"text": doc, "species": meta["species"], "source_url": meta["source_url"], "score": 1 - dist})
    return hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--db", default="./chroma_db")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    hits = retrieve(args.question, args.db, args.k)
    print(f"\nTop {len(hits)} results for: {args.question!r}\n")
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] {hit['species']}  (similarity={hit['score']:.3f})")
        print(f"    {hit['text'][:250]}...")
        print(f"    source: {hit['source_url']}\n")
