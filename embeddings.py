"""
Embedder abstraction. LocalEmbedder wraps sentence-transformers all-MiniLM-L6-v2
(384-dim, no API key, ~80 MB download on first use).

The Embedder protocol lets tests swap in a fake without hitting the network or disk.

Install: pip install sentence-transformers
"""
from __future__ import annotations

from typing import List


class Embedder:
    """Protocol-style base. Subclass or duck-type to replace."""
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class LocalEmbedder(Embedder):
    """Lazy-loaded so import doesn't trigger model download."""

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)

    def embed(self, text: str) -> List[float]:
        self._load()
        return self._model.encode(text, normalize_embeddings=True).tolist()


_default_embedder: LocalEmbedder | None = None


def get_default_embedder() -> LocalEmbedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = LocalEmbedder()
    return _default_embedder
