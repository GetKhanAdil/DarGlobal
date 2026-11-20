"""Scraper for DarGlobal (darglobal.co.uk) and Wasalt (wasalt.sa).

Collects publicly available project/company data and writes it to
app/data/knowledge_base.json, which the chatbot ships with and queries at
runtime. Uses only requests + BeautifulSoup, so it needs no paid API key.

Run:  python scraper/scrape.py

Note: both sites are partly JavaScript-rendered. This script extracts what is
available in the server-rendered HTML (project cards with name + location,
company copy, FAQs). The committed knowledge_base.json was produced from this
pipeline and then lightly curated into clean, deduplicated documents. Re-run
to refresh; review the output before shipping.
"""
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

OUT_PATH = Path(__file__).parent.parent / "app" / "data" / "knowledge_base.json"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
TIMEOUT = 30


def fetch(url: str) -> BeautifulSoup | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"[warn] failed to fetch {url}: {e}")
        return None


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def scrape_darglobal_projects() -> list[dict]:
    """Extract DarGlobal project cards (name + location) from /projects."""
    docs = []
    soup = fetch("https://darglobal.co.uk/projects")
    if not soup:
        return docs
    seen = set()
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if "darglobal.co.uk" not in href and not href.startswith("/"):
            continue
        name_el = a.find("strong") or a.find("b")
        if not name_el:
            continue
        name = clean(name_el.get_text())
        if not name or name in seen:
            continue
        # The card text usually contains the location line before the name.
        card_text = clean(a.get_text(" "))
        location = clean(card_text.replace(name, ""))
        url = href if href.startswith("http") else f"https://darglobal.co.uk{href}"
        seen.add(name)
        docs.append(
            {
                "id": "dg-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
                "source": "DarGlobal",
                "title": name,
                "url": url,
                "text": f"{name} is a DarGlobal luxury real estate project. Location: {location}.",
            }
        )
    print(f"[info] DarGlobal: {len(docs)} project cards")
    return docs


def scrape_darglobal_home() -> list[dict]:
    """Extract DarGlobal company overview paragraphs from the homepage."""
    docs = []
    soup = fetch("https://darglobal.co.uk/")
    if not soup:
        return docs
    paragraphs = [clean(p.get_text(" ")) for p in soup.find_all("p")]
    blob = " ".join(p for p in paragraphs if len(p) > 60)[:3000]
    if blob:
        docs.append(
            {
                "id": "dg-overview",
                "source": "DarGlobal",
                "title": "About DarGlobal",
                "url": "https://darglobal.co.uk/",
                "text": blob,
            }
        )
    return docs


def scrape_wasalt() -> list[dict]:
    """Extract Wasalt overview + FAQ content from the homepage."""
    docs = []
    soup = fetch("https://wasalt.sa/en")
    if not soup:
        return docs
    paragraphs = [clean(p.get_text(" ")) for p in soup.find_all("p")]
    blob = " ".join(p for p in paragraphs if len(p) > 50)[:3000]
    if blob:
        docs.append(
            {
                "id": "wa-overview",
                "source": "Wasalt",
                "title": "About Wasalt",
                "url": "https://wasalt.sa/en",
                "text": blob,
            }
        )
    print(f"[info] Wasalt: {len(docs)} documents")
    return docs


def main():
    docs = []
    docs += scrape_darglobal_home()
    docs += scrape_darglobal_projects()
    docs += scrape_wasalt()

    if not docs:
        print("[error] scraped nothing; keeping existing knowledge_base.json")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {len(docs)} documents to {OUT_PATH}")


if __name__ == "__main__":
    main()
