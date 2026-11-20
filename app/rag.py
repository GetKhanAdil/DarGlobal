"""Lightweight retrieval over the scraped knowledge base using BM25.

No embeddings / vector DB needed: the knowledge base is small (a few dozen
curated documents), so BM25 keyword ranking is fast, dependency-light and
good enough to ground the chatbot's answers.
"""
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_DATA_PATH = Path(__file__).parent / "data" / "knowledge_base.json"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    # Very light plural normalization so "brands" matches "brand",
    # "villas" matches "villa", "projects" matches "project", etc.
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower())]


class KnowledgeBase:
    def __init__(self, path: Path = _DATA_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self.docs: list[dict] = json.load(f)
        # Index on title + text so a query matching a title still ranks well.
        corpus = [_tokenize(f"{d['title']} {d['text']}") for d in self.docs]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Return the top-k most relevant documents for the query."""
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.docs, scores), key=lambda x: x[1], reverse=True
        )
        # Keep only docs with a positive score (an actual keyword match).
        return [doc for doc, score in ranked[:k] if score > 0]

    def build_context(self, query: str, k: int = 4) -> tuple[str, list[dict]]:
        """Return a formatted context block and the source docs used."""
        hits = self.search(query, k=k)
        if not hits:
            return "", []
        blocks = []
        for d in hits:
            blocks.append(
                f"[{d['source']} - {d['title']}]\n{d['text']}\nSource: {d['url']}"
            )
        return "\n\n".join(blocks), hits


# Singleton loaded once at import time.
kb = KnowledgeBase()
