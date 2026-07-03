"""
Node 10: Persistence.

Pure logic — no LLM. Applies the reconciled_patches from Node 9 to SQLite
and re-embeds changed entities in Chroma so the next chapter's context
builder sees an updated world.

For each patch:
  1. Load the current entity from SQLite.
  2. Apply non-None fields from the patch (additive fields like knowledge_added
     are appended, not replaced).
  3. Write the updated entity back to SQLite.
  4. Re-embed the updated entity in Chroma.

Also persists:
  - chapter_summary  → chapter_summaries table
  - chapter_number   → chapters table (so Node 1 can detect it next run)
  - pov_state        → from POVPatch, if present
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import db
import vector_store as vs
from embeddings import get_default_embedder
from schema import (
    Character, ChapterGraphState, ChapterSummary, Location, Plotline, POVState,
    CharacterPatch, PlotlinePatch, LocationPatch, POVPatch,
)


# ---------------------------------------------------------------------------
# Patch appliers — load current entity, merge patch fields, return updated
# ---------------------------------------------------------------------------

def _apply_character_patch(
    patch: CharacterPatch,
    story_id: str,
    db_path: Optional[Path],
    _p=print,
) -> Optional[Character]:
    c = db.get_character_by_id(patch.entity_id, story_id, db_path)
    if c is None:
        _p(f"    [warn] Character {patch.entity_id} not found in DB — skipping patch")
        return None

    updates: dict = {}
    if patch.emotional_state is not None:
        updates["emotional_state"] = patch.emotional_state
    if patch.current_location_id is not None:
        updates["current_location_id"] = patch.current_location_id
    if patch.personality is not None:
        updates["personality"] = patch.personality
    if patch.goals is not None:
        updates["goals"] = patch.goals
    if patch.relationships is not None:
        updates["relationships"] = {**c.relationships, **patch.relationships}
    if patch.reputation is not None:
        updates["reputation"] = {**c.reputation, **patch.reputation}
    if patch.current_objectives is not None:
        updates["current_objectives"] = patch.current_objectives
    if patch.is_alive is not None:
        updates["is_alive"] = patch.is_alive
        if not patch.is_alive:
            updates["current_objectives"] = []
            updates["emotional_state"] = "deceased"
    if patch.knowledge_added:
        updates["knowledge"] = list(dict.fromkeys(c.knowledge + patch.knowledge_added))
    if patch.secrets_added:
        updates["secrets"] = list(dict.fromkeys(c.secrets + patch.secrets_added))

    updates["updated_at"] = datetime.utcnow()
    return c.model_copy(update=updates)


def _apply_plotline_patch(
    patch: PlotlinePatch,
    story_id: str,
    db_path: Optional[Path],
    chapter_number: int,
    _p=print,
) -> Optional[Plotline]:
    p = db.get_plotline_by_id(patch.entity_id, story_id, db_path)
    if p is None:
        _p(f"    [warn] Plotline {patch.entity_id} not found in DB — skipping patch")
        return None

    updates: dict = {}
    if patch.status is not None:
        updates["status"] = patch.status
    if patch.progress_stage is not None:
        updates["progress_stage"] = patch.progress_stage
    if patch.current_tension is not None:
        updates["current_tension"] = max(0, min(10, patch.current_tension))
    if patch.next_possible_developments is not None:
        updates["next_possible_developments"] = patch.next_possible_developments
    if patch.involved_character_ids_added:
        combined = list(dict.fromkeys(p.involved_character_ids + patch.involved_character_ids_added))
        updates["involved_character_ids"] = combined

    # Any patch at all means this thread was addressed this chapter — resets the stale-plotline clock.
    updates["last_touched_chapter"] = chapter_number
    updates["updated_at"] = datetime.utcnow()
    return p.model_copy(update=updates)


def _apply_location_patch(
    patch: LocationPatch,
    story_id: str,
    db_path: Optional[Path],
    _p=print,
) -> Optional[Location]:
    loc = db.get_location_by_id(patch.entity_id, story_id, db_path)
    if loc is None:
        _p(f"    [warn] Location {patch.entity_id} not found in DB — skipping patch")
        return None

    updates: dict = {}
    if patch.description is not None:
        updates["description"] = patch.description
    if patch.political_control is not None:
        updates["political_control"] = patch.political_control
    if patch.tone is not None:
        updates["tone"] = patch.tone
    if patch.npcs_present is not None:
        updates["npcs_present"] = patch.npcs_present
    if patch.recent_events_added:
        # Keep last 10 events to avoid unbounded growth
        combined = list(dict.fromkeys(loc.recent_events + patch.recent_events_added))
        updates["recent_events"] = combined[-10:]
    if patch.secrets_added:
        updates["secrets"] = list(dict.fromkeys(loc.secrets + patch.secrets_added))

    updates["updated_at"] = datetime.utcnow()
    return loc.model_copy(update=updates)


def _apply_pov_patch(
    patch: POVPatch,
    story_id: str,
    db_path: Optional[Path],
    _p=print,
) -> Optional[POVState]:
    pov = db.get_pov_state(story_id, db_path)
    if pov is None:
        _p(f"    [warn] POV state not found in DB — skipping patch")
        return None

    updates: dict = {}
    if patch.pov_character_id is not None:
        updates["pov_character_id"] = patch.pov_character_id
    if patch.location_id is not None:
        updates["location_id"] = patch.location_id
    if patch.companions is not None:
        updates["companions"] = patch.companions
    if patch.emotional_state is not None:
        updates["emotional_state"] = patch.emotional_state
    if patch.goals is not None:
        updates["goals"] = patch.goals
    if patch.inventory_added:
        updates["inventory"] = list(dict.fromkeys(pov.inventory + patch.inventory_added))
    if patch.inventory_removed:
        current = updates.get("inventory", pov.inventory)
        updates["inventory"] = [i for i in current if i not in patch.inventory_removed]
    if patch.injuries_added:
        updates["injuries"] = list(dict.fromkeys(pov.injuries + patch.injuries_added))
    if patch.knowledge_added:
        updates["knowledge"] = list(dict.fromkeys(pov.knowledge + patch.knowledge_added))

    return pov.model_copy(update=updates)


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_persistence_node(
    chroma_client=None,
    embedder=None,
    db_path: Optional[Path] = None,
    print_fn=print,
) -> Callable[[ChapterGraphState], dict]:
    _p = print_fn

    def _chroma():
        if chroma_client is not None:
            return chroma_client
        raise RuntimeError("persistence node requires a chroma_client — pass one to make_persistence_node()")

    def _embedder():
        return embedder if embedder is not None else get_default_embedder()

    def node(state: ChapterGraphState) -> dict:
        _db = db_path
        story_id = state.story_id
        chroma = _chroma()
        emb = _embedder()

        def embed_and_upsert(entity_type, entity_id, text, sid=None):
            vec = emb.embed(text)
            vs.upsert_entity(chroma, entity_type, entity_id, vec, text, story_id=sid)

        # Whole chapter's worth of writes commit or roll back together — a crash
        # partway through (or an embedding failure) can't leave the DB half-updated.
        conn = db.get_connection(_db)
        try:
            # --- Apply reconciled patches ---
            for patch in state.reconciled_patches:
                if isinstance(patch, CharacterPatch):
                    updated = _apply_character_patch(patch, story_id, _db, _p=_p)
                    if updated:
                        db.upsert_character(updated, story_id, _db, conn=conn)
                        embed_and_upsert("character", updated.id, vs.character_text(updated), sid=story_id)
                        _p(f"    Persisted character: {updated.name}")

                elif isinstance(patch, PlotlinePatch):
                    updated = _apply_plotline_patch(patch, story_id, _db, state.chapter_number, _p=_p)
                    if updated:
                        db.upsert_plotline(updated, story_id, _db, conn=conn)
                        embed_and_upsert("plotline", updated.id, vs.plotline_text(updated), sid=story_id)
                        _p(f"    Persisted plotline: {updated.name}")

                elif isinstance(patch, LocationPatch):
                    updated = _apply_location_patch(patch, story_id, _db, _p=_p)
                    if updated:
                        db.upsert_location(updated, story_id, _db, conn=conn)
                        embed_and_upsert("location", updated.id, vs.location_text(updated), sid=story_id)
                        _p(f"    Persisted location: {updated.name}")

                elif isinstance(patch, POVPatch):
                    updated = _apply_pov_patch(patch, story_id, _db, _p=_p)
                    if updated:
                        db.upsert_pov_state(updated, story_id, _db, conn=conn)
                        _p(f"    Persisted POV state")

            # --- Persist newly discovered entities ---
            for c in state.new_characters:
                db.upsert_character(c, story_id, _db, conn=conn)
                embed_and_upsert("character", c.id, vs.character_text(c), sid=story_id)
                _p(f"    Created new character: {c.name}")

            for loc in state.new_locations:
                db.upsert_location(loc, story_id, _db, conn=conn)
                embed_and_upsert("location", loc.id, vs.location_text(loc), sid=story_id)
                _p(f"    Created new location: {loc.name}")

            for p in state.new_plotlines:
                p = p.model_copy(update={"last_touched_chapter": state.chapter_number})
                db.upsert_plotline(p, story_id, _db, conn=conn)
                embed_and_upsert("plotline", p.id, vs.plotline_text(p), sid=story_id)
                _p(f"    Created new plotline: {p.name}")

            for r in state.new_world_rules:
                db.upsert_world_rule(r, _db, conn=conn)
                embed_and_upsert("world_rule", r.id, vs.world_rule_text(r))
                _p(f"    Created new world rule: {r.title}")

            for l in state.new_world_lore:
                db.upsert_world_lore(l, _db, conn=conn)
                embed_and_upsert("world_lore", l.id, vs.world_lore_text(l))
                _p(f"    Created new world lore: {l.title}")

            # --- Persist chapter summary ---
            if state.chapter_summary:
                db.upsert_chapter_summary(state.chapter_summary, story_id, _db, conn=conn)
                _p(f"    Persisted chapter {state.chapter_summary.chapter_number} summary")

            # --- Mechanical outline updates (new beats/arcs, completed beats) ---
            from node_outline_manager import apply_mechanical_outline_updates
            apply_mechanical_outline_updates(state, story_id, _db, conn=conn)

            # --- Register chapter number so Node 1 sees it next run ---
            conn.execute(
                "INSERT OR IGNORE INTO chapters (story_id, chapter_number) VALUES (?, ?)",
                (story_id, state.chapter_number),
            )

            conn.commit()
            _p(f"    Registered chapter {state.chapter_number}")
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        return {}

    return node
