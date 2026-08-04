"""
tools.py — the three tools the FloraSense agent can choose between.

DEPLOYMENT VERSION: classify_flower calls the vision model directly in this
same Python process (via model_service.py), instead of making an HTTP call
to a separate FastAPI server. This matters for single-service hosting (e.g.
Hugging Face Spaces free tier), where running 3 separate services isn't
practical — everything needs to live in one process.

If you're running locally with the separate FastAPI backend instead, use the
HTTP-based tools.py from florasense_agent/ rather than this one.
"""
import json
import os

import chromadb
import requests
from langchain_core.tools import tool
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer

import model_service

# --- Config -----------------------------------------------------------------
CHROMA_DB_PATH = os.getenv("FLORASENSE_CHROMA_DB", "./chroma_db")
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "florasense_knowledge"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Lazily initialized so importing this module doesn't immediately try to
# load a 130MB model or open a DB connection before it's actually needed.
_embedder = None
_chroma_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection


# --- Tool 1: Vision classifier -----------------------------------------------
@tool
def classify_flower(image_path: str) -> str:
    """Identify the flower species in an uploaded image.

    Use this FIRST whenever the user has attached or referenced an image of a
    flower/plant. Returns the predicted species, a calibrated confidence
    score, and the top-5 alternative guesses. If confidence is low, treat the
    identification as uncertain rather than fact when answering the user.

    Args:
        image_path: local filesystem path to the image file.
    """
    if not os.path.exists(image_path):
        return json.dumps({"error": f"Image file not found at {image_path}"})

    if not model_service.flora_model.loaded:
        try:
            model_service.flora_model.load()
        except Exception as e:
            return json.dumps({"error": f"Could not load vision model: {e}"})

    try:
        pil_image = Image.open(image_path)
        pil_image.load()
    except UnidentifiedImageError:
        return json.dumps({"error": "Could not decode image file."})

    try:
        species, confidence, is_confident, top5, _gradcam_b64 = model_service.flora_model.predict(pil_image)
    except Exception as e:
        return json.dumps({"error": f"Inference failed: {e}"})

    return json.dumps(
        {
            "predicted_species": species,
            "confidence": confidence,
            "is_confident": is_confident,
            "top5": [{"species": s, "confidence": c} for s, c in top5],
        }
    )


# --- Tool 2: RAG retriever ---------------------------------------------------
@tool
def retrieve_knowledge(query: str, species: str = "", k: int = 5) -> str:
    """Search the botanical knowledge base for information relevant to a question.

    Use this to answer questions about a flower's care, diseases, pollinators,
    native range, or uses. If you already know the species (e.g. from
    classify_flower or because the user named it), pass it in `species` to
    bias retrieval and include it in the query text.

    Args:
        query: the question or topic to search for, e.g. "diseases affecting this plant".
        species: optional species name to focus the search on.
        k: number of passages to retrieve (default 5).
    """
    search_text = f"{species}: {query}" if species else query

    try:
        collection = _get_collection()
    except Exception as e:
        return json.dumps({"error": f"Could not reach knowledge base: {e}"})

    try:
        embedder = _get_embedder()
        query_embedding = embedder.encode([BGE_QUERY_PREFIX + search_text], normalize_embeddings=True).tolist()
    except Exception as e:
        return json.dumps({"error": f"Could not load embedding model: {e}"})

    where_filter = {"species": species} if species else None
    results = collection.query(query_embeddings=query_embedding, n_results=k, where=where_filter)

    if not results["documents"][0]:
        # Species filter may be too strict (e.g. name mismatch) — retry without it.
        results = collection.query(query_embeddings=query_embedding, n_results=k)

    passages = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        passages.append(
            {
                "text": doc,
                "species": meta["species"],
                "source_url": meta.get("source_url", ""),
                "relevance": round(1 - dist, 3),
            }
        )
    return json.dumps(passages)


# --- Tool 3: Weather ---------------------------------------------------------
@tool
def get_weather(location: str) -> str:
    """Get current weather conditions for a location.

    Use this for questions about whether a plant can grow somewhere right
    now, or gardening timing questions ("should I water today", "is it too
    cold to plant this"). Uses Open-Meteo (free, no API key).

    Args:
        location: a place name, e.g. "Patiala, Punjab" or "London".
    """
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
            timeout=10,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        results = geo_data.get("results")
        if not results:
            return json.dumps({"error": f"Could not find location: {location}"})

        lat, lon = results[0]["latitude"], results[0]["longitude"]
        resolved_name = f"{results[0].get('name', location)}, {results[0].get('country', '')}".strip(", ")

        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
        current = weather_data.get("current", {})

    except requests.RequestException as e:
        return json.dumps({"error": f"Weather lookup failed: {e}"})

    return json.dumps(
        {
            "location": resolved_name,
            "temperature_c": current.get("temperature_2m"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "weather_code": current.get("weather_code"),
            "observed_at": current.get("time"),
        }
    )


ALL_TOOLS = [classify_flower, retrieve_knowledge, get_weather]
