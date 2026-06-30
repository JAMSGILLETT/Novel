"""
Chroma vector store wrapper.

One collection per entity type, all stories share collections — story_id is
stored in metadata and used as a `where` filter on every query.

Distance space: cosine. Chroma returns distance = 1 - cosine_similarity for
normalized embeddings, so score = 1 - distance maps back to [0, 1].

Persistence: co-located with novelgen.db as ./novelgen_chroma/

Install: pip install chromadb==0.4.24
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

CHROMA_PATH = Path(__file__).parent / "novelgen_chroma"

# These collections are shared across all stories — no story_id filter applied.
GLOBAL_ENTITY_TYPES = {"world_rule", "world_lore"}

COLLECTION_FOR = {
    "character": "characters",
    "plotline": "plotlines",
    "location": "locations",
    "world_rule": "world_rules",
    "world_lore": "world_lore",
    "chapter_summary": "chapter_summaries",
}


def get_chroma_client(path: Optional[Path] = None):
    import chromadb
    return chromadb.PersistentClient(path=str(path or CHROMA_PATH))


def get_ephemeral_client():
    """For tests only — in-memory, no disk."""
    import chromadb
    return chromadb.EphemeralClient()


def _collection(client, entity_type: str):
    return client.get_or_create_collection(
        COLLECTION_FOR[entity_type],
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Text representations (what gets embedded for each entity type)
# ---------------------------------------------------------------------------

def character_text(c) -> str:
    parts = [c.name, c.personality]
    if c.goals:
        parts.append("Goals: " + "; ".join(c.goals))
    if c.current_objectives:
        parts.append("Objectives: " + "; ".join(c.current_objectives))
    return ". ".join(p for p in parts if p)


def plotline_text(p) -> str:
    parts = [p.name, p.progress_stage]
    if p.next_possible_developments:
        parts.append("Next: " + "; ".join(p.next_possible_developments))
    return ". ".join(p for p in parts if p)


def location_text(loc) -> str:
    parts = [loc.name, loc.description]
    if loc.tone:
        parts.append("Tone: " + loc.tone)
    return ". ".join(p for p in parts if p)


def world_rule_text(r) -> str:
    return f"{r.title}: {r.content}"


def world_lore_text(l) -> str:
    return f"{l.title}: {l.content}"


def chapter_summary_text(s) -> str:
    return s.medium_summary


ENTITY_TEXT_FN = {
    "character": character_text,
    "plotline": plotline_text,
    "location": location_text,
    "world_rule": world_rule_text,
    "world_lore": world_lore_text,
    "chapter_summary": chapter_summary_text,
}


# ---------------------------------------------------------------------------
# Upsert and search
# ---------------------------------------------------------------------------

def upsert_entity(
    client,
    entity_type: str,
    entity_id: str,
    embedding: List[float],
    text: str,
    story_id: Optional[str] = None,
) -> None:
    col = _collection(client, entity_type)
    meta: dict = {"entity_id": entity_id}
    if entity_type not in GLOBAL_ENTITY_TYPES and story_id is not None:
        meta["story_id"] = story_id
    col.upsert(
        ids=[entity_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[meta],
    )


def vector_search(
    client,
    entity_type: str,
    query_embedding: List[float],
    story_id: str,
    threshold: float,
    max_results: int,
) -> List[Tuple[str, float]]:
    """
    Returns (entity_id, score) pairs where score >= threshold, ordered by score desc.
    score is cosine similarity in [0, 1].
    Global entity types (world_rule, world_lore) are searched without a story_id filter.
    """
    col = _collection(client, entity_type)
    total = col.count()
    if total == 0:
        return []

    n = min(max_results, total)
    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": n,
        "include": ["metadatas", "distances"],
    }
    if entity_type not in GLOBAL_ENTITY_TYPES:
        query_kwargs["where"] = {"story_id": story_id}

    try:
        results = col.query(**query_kwargs)
    except Exception:
        return []

    hits: List[Tuple[str, float]] = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        score = 1.0 - dist
        if score >= threshold:
            hits.append((meta["entity_id"], score))

    hits.sort(key=lambda x: x[1], reverse=True)
    return hits
