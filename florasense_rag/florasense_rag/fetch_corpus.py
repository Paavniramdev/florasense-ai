"""
fetch_corpus.py — builds the raw knowledge base for FloraSense AI's RAG layer.

For each of the 102 flower species, this pulls the full plain-text Wikipedia
article (not just the summary) via Wikipedia's official API. Wikipedia content
is CC BY-SA licensed, so it's safe to store and re-serve locally with
attribution (each saved doc keeps its source URL).

Why Wikipedia as the base corpus:
- Real, verifiable, freely-licensed text (not hallucinated)
- Plant articles typically have Description / Cultivation / Uses / Pests &
  diseases / Distribution sections — which map directly onto the question
  types this assistant needs to answer (care, diseases, native range, uses)
- Free and requires no API key

This is a *starting* corpus. For production quality you'd want to layer in
specialist sources (e.g. RHS, USDA plant database, medicinal plant
references) — the ingestion pattern here (fetch -> save raw JSON -> chunk ->
embed) stays the same regardless of source, so extending it later is just
adding another fetch function.

Usage:
    python fetch_corpus.py --labels ../backend/models/idx_to_name.json --out data/raw
    # or, with no labels file, it fetches the official 102 species names itself:
    python fetch_corpus.py --out data/raw
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "FloraSenseAI-Research/1.0 (educational project; contact: student)"}
CATEGORIES_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html"


def fetch_official_species_names():
    """Same technique used in the Week 1 training notebook — pulls the
    authoritative 102 species names directly from Oxford's own page."""
    req = urllib.request.Request(CATEGORIES_URL, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    cells = re.findall(r"<td[^>]*>\s*(.*?)\s*</td>", html, flags=re.S)
    names = []
    for c in cells:
        c = re.sub(r"<[^>]+>", "", c).strip().replace("&amp;", "&")
        if c and not c.isdigit():
            names.append(c)
    return names


def load_species_names(labels_path: str):
    if labels_path and os.path.exists(labels_path):
        with open(labels_path) as f:
            raw = json.load(f)
        # idx_to_name.json is {"0": "pink primrose", ...}
        return [raw[str(i)] for i in range(len(raw))]
    print("No labels file given/found — fetching official names from Oxford's site.")
    return fetch_official_species_names()


def wiki_query(params: dict, max_retries: int = 5) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"{WIKI_API}?{query}"
    req = urllib.request.Request(url, headers=HEADERS)

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Respect Retry-After if Wikipedia sends one, otherwise back off
                # exponentially (2s, 4s, 8s, 16s, 32s).
                wait = int(e.headers.get("Retry-After", 2 ** (attempt + 1)))
                print(f"    rate-limited (429) — waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            wait = 2 ** (attempt + 1)
            print(f"    network error ({e}) — waiting {wait}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


def fetch_wikipedia_extract(title: str):
    """Tries a few title variants since flower common names are sometimes
    ambiguous (e.g. "canna lily" vs "canna (plant)")."""
    candidates = [title, f"{title} (plant)", f"{title} flower"]

    for candidate in candidates:
        data = wiki_query(
            {
                "action": "query",
                "format": "json",
                "prop": "extracts|info",
                "explaintext": 1,
                "redirects": 1,
                "inprop": "url",
                "titles": candidate,
            }
        )
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":
                continue  # page not found, try next candidate
            extract = page.get("extract", "").strip()
            if len(extract) > 200:  # avoid disambiguation stubs
                return {
                    "species": title,
                    "wiki_title": page.get("title", candidate),
                    "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{candidate}"),
                    "text": extract,
                }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="", help="Path to idx_to_name.json from Week 1 export")
    parser.add_argument("--out", default="data/raw", help="Output directory for raw JSON docs")
    parser.add_argument("--sleep", type=float, default=1.5, help="Seconds between requests (be polite to Wikipedia)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    species_names = load_species_names(args.labels)
    print(f"Fetching Wikipedia articles for {len(species_names)} species...")

    fetched, missing = 0, []
    for i, name in enumerate(species_names, 1):
        safe_name = name.strip().lower().replace(" ", "_").replace("/", "_")
        out_path = os.path.join(args.out, f"{safe_name}.json")

        if os.path.exists(out_path):
            fetched += 1
            continue  # resume-friendly: skip already-fetched species

        result = fetch_wikipedia_extract(name)
        if result:
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            fetched += 1
            print(f"  [{i}/{len(species_names)}] OK  — {name} ({len(result['text'])} chars)")
        else:
            missing.append(name)
            print(f"  [{i}/{len(species_names)}] MISS — {name} (no good Wikipedia match)")

        time.sleep(args.sleep)

    print(f"\nDone. Fetched {fetched}/{len(species_names)}.")
    if missing:
        print(f"{len(missing)} species had no good match — you may want to add these manually:")
        for m in missing:
            print(f"  - {m}")
        with open(os.path.join(args.out, "_missing.json"), "w") as f:
            json.dump(missing, f, indent=2)


if __name__ == "__main__":
    main()
