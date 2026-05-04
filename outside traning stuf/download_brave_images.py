"""Download real and AI-generated images via Bing Image Search API.

For each topic in search tearms.txt:
  - Searches "{topic}" with date filter before 2020 -> saves to real/
  - Searches one of 4 AI tool variants with date filter 2023-present -> saves to ai/

AI query variants rotate evenly across topics:
  "Midjourney style {topic}"
  "DALL-E 3 generated {topic}"
  "Stable Diffusion {topic}"
  "AI generated {topic}"

Usage:
    & .\.venv311\Scripts\python.exe .\training\download_bing_images.py
"""

from __future__ import annotations

import os
import time
import hashlib
import concurrent.futures
from pathlib import Path

import requests
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

BRAVE_API_KEY = "BSAYJhRNgXpCa6GkTZOi6Y7E9CA2z7e"

SEARCH_TERMS_FILE = Path(r"C:\Stuff\coding\New classifier\search tearms 4.txt")
OUT_ROOT = Path(r"W:\Datasets\AI immage classifier 3.0 datasets\web-scrape-3")

IMAGES_PER_TOPIC  = 500
MIN_SIZE_BYTES    = 10_000
MIN_DIM           = 400
WORKERS           = 64
SLEEP_BETWEEN_SEARCHES = 0.3

REAL_FRESHNESS = "2010-01-01to2019-12-31"
AI_FRESHNESS   = "2023-01-01to2025-12-31"

# Rotating AI query prefixes — each topic gets one of these
AI_PREFIXES = [
    "Midjourney",
    "DALLE3",
    "Stable Diffusion",
    "AI generated",
]

# ── Brave Image Search ───────────────────────────────────────────────────────

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/images/search"


def brave_search(query: str, count: int = 100, offset: int = 0, freshness: str = "") -> list[dict]:
    """Brave Images API — max 100 results per call."""
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {
        "q": query,
        "count": min(count, 100),
        "offset": offset,
        "safesearch": "off",
        "freshness": freshness,
    }
    try:
        r = requests.get(BRAVE_ENDPOINT, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        print(f"  Search error for '{query}': {e}")
        try:
            print(f"  Response body: {r.text[:300]}")
        except Exception:
            pass
        return []


def fetch_all_results(query: str, target: int, freshness: str = "") -> list[str]:
    """Single page fetch — Brave only has ~100 unique results per query anyway."""
    batch = brave_search(query, count=100, offset=0, freshness=freshness)
    urls = []
    for item in batch:
        url = item.get("properties", {}).get("url", "") or item.get("url", "")
        if url:
            urls.append(url)
    print(f"    found {len(urls)} URLs", flush=True)
    return urls[:target]


# ── Download ──────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def url_to_filename(url: str, idx: int) -> str:
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in IMAGE_EXTS:
        ext = ".jpg"
    return f"{idx:06d}_{h}{ext}"


def download_image(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        r = requests.get(url, timeout=5, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.content
        if len(data) < MIN_SIZE_BYTES:
            return False
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data))
        if img.width < MIN_DIM or img.height < MIN_DIM:
            return False
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def download_batch(urls: list[str], folder: Path, start_idx: int) -> int:
    folder.mkdir(parents=True, exist_ok=True)

    def fetch(args):
        i, url = args
        dest = folder / url_to_filename(url, start_idx + i)
        return download_image(url, dest)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(fetch, args): args for args in enumerate(urls)}
        with tqdm(total=len(urls), unit="img", leave=False) as pbar:
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())
                pbar.update(1)
                pbar.set_postfix(saved=sum(results))
    return sum(results)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_topics() -> list[str]:
    lines = SEARCH_TERMS_FILE.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def process_topic(topic: str, topic_idx: int, total: int) -> None:
    real_dir = OUT_ROOT / "real"
    ai_dir   = OUT_ROOT / "ai"

    existing_real = len(list(real_dir.glob("*.*"))) if real_dir.exists() else 0
    existing_ai   = len(list(ai_dir.glob("*.*")))   if ai_dir.exists()   else 0

    # Pick AI prefix by rotating through the 4 options
    ai_prefix = AI_PREFIXES[topic_idx % len(AI_PREFIXES)]

    print(f"\n[{topic_idx+1}/{total}] {topic}")

    # Real — date context baked into query
    real_query = f"{topic}"
    print(f"  Real: '{real_query}' (2010-2019)")
    real_urls = fetch_all_results(real_query, IMAGES_PER_TOPIC, freshness=REAL_FRESHNESS)
    print(f"  Found {len(real_urls)} URLs, downloading...")
    n_real = download_batch(real_urls, real_dir, existing_real)
    print(f"  Saved {n_real} real images  (total real: {existing_real + n_real})")

    time.sleep(SLEEP_BETWEEN_SEARCHES)

    # AI — tool prefix + recent context baked into query
    ai_query = f"{ai_prefix} {topic}"
    print(f"  AI:   '{ai_query}' (2023-2025)")
    ai_urls = fetch_all_results(ai_query, IMAGES_PER_TOPIC, freshness=AI_FRESHNESS)
    print(f"  Found {len(ai_urls)} URLs, downloading...")
    n_ai = download_batch(ai_urls, ai_dir, existing_ai)
    print(f"  Saved {n_ai} AI images  (total ai: {existing_ai + n_ai})")


if __name__ == "__main__":
    if not BRAVE_API_KEY:
        raise SystemExit("ERROR: Set your Brave API key in BRAVE_API_KEY at the top of this script.")

    topics = load_topics()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Output:  {OUT_ROOT}")
    print(f"Topics:  {len(topics)}")
    print(f"Target:  {IMAGES_PER_TOPIC} real + {IMAGES_PER_TOPIC} AI per topic")
    print(f"Est. total: ~{len(topics) * IMAGES_PER_TOPIC * 2:,} images")
    print(f"Real freshness: {REAL_FRESHNESS}")
    print(f"AI freshness:   {AI_FRESHNESS}")
    print(f"AI prefixes: {AI_PREFIXES}\n")

    for i, topic in enumerate(topics):
        process_topic(topic, i, len(topics))

    real_count = len(list((OUT_ROOT / "real").glob("*.*")))
    ai_count   = len(list((OUT_ROOT / "ai").glob("*.*")))
    print(f"\nAll done.  real={real_count}  ai={ai_count}")
