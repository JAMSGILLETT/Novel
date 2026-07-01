"""
Node 2: Context Builder.

Produces a ContextPack via two passes:

MANDATORY PASS (always included regardless of relevance score):
  - All WorldRules (global hard constraints, never skip)
  - All active Plotlines (status="active") — open threads must never be dropped
  - POV character only (from POVState.pov_character_id)
  - POV character's current Location (from POVState.location_id)
  - Last chapter summary (anchors continuity)
  - Full character roster (id/name/alive for every character ever introduced,
    so the Story Planner can reference off-screen characters by name)

  All other characters (including companions) are pulled by vector search only —
  they must be relevant to the current prompt or last summary to appear.

VECTOR PASS (relevance-ranked, filtered by per-type thresholds):
  - Characters, plotlines, locations, world_lore searched by embedding
  - Query text = user_input + last chapter summary (if it exists), so that
    a bare "continue" still finds entities relevant to where the story left off
    rather than matching nothing
  - Results merged with mandatory set; duplicates dropped (mandatory wins)

DEPENDENCY GRAPH PASS:
  - For every entity found so far, look up canon_rules in SQLite
  - Inject specified entities unconditionally (bypassing threshold)
  - Record as DependencyGraphHit so downstream nodes know why it appeared

STALE-PLOTLINE AUDIT:
  - Pure chapter-count check (no LLM, no extra DB call) over active_plotlines
  - Any active plotline untouched by a patch for STALE_PLOTLINE_THRESHOLD+
    chapters is recorded on ContextPack.stale_plotlines — logged by the
    pipeline and surfaced to the Story Planner as a soft nudge, never a hard
    requirement

Factory pattern: make_context_builder_node() injects dependencies so tests
can pass a FakeEmbedder and EphemeralClient without touching production singletons.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import db
import vector_store as vs
from embeddings import Embedder, get_default_embedder
from schema import (
    CanonRule, Character, CharacterRosterEntry, ChapterSummary,
    ContextPack, DependencyGraphHit,
    Location, Plotline, WorldLore, WorldRule,
    ChapterGraphState,
)

STALE_PLOTLINE_THRESHOLD = 8  # chapters an active plotline can go untouched before it's flagged as going quiet


def make_context_builder_node(
    embedder: Optional[Embedder] = None,
    chroma_client=None,
    db_path: Optional[Path] = None,
) -> Callable[[ChapterGraphState], dict]:

    def _embedder() -> Embedder:
        return embedder if embedder is not None else get_default_embedder()

    def _client():
        return chroma_client if chroma_client is not None else vs.get_chroma_client()

    def node(state: ChapterGraphState) -> dict:
        story_id = state.story_id
        cfg = state.retrieval_config
        client = _client()

        # ------------------------------------------------------------------ #
        # MANDATORY PASS                                                       #
        # ------------------------------------------------------------------ #

        all_world_rules: List[WorldRule] = db.get_all_world_rules(db_path)
        all_world_lore: List[WorldLore] = db.get_all_world_lore(db_path)

        all_plotlines = db.get_all_plotlines(story_id, db_path)
        active_plotlines: List[Plotline] = [
            p for p in all_plotlines if p.status == "active"
        ]

        # Stale-plotline audit: active threads nobody's touched in a while.
        # Pure chapter-count check, no extra DB call — active_plotlines is already loaded above.
        stale_plotlines: List[Plotline] = [
            p for p in active_plotlines
            if state.chapter_number - p.last_touched_chapter >= STALE_PLOTLINE_THRESHOLD
        ]

        pov = db.get_pov_state(story_id, db_path)

        mandatory_characters: List[Character] = []
        mandatory_char_ids: Set[str] = set()
        if pov and pov.pov_character_id:
            c = db.get_character_by_id(pov.pov_character_id, story_id, db_path)
            if c:
                mandatory_characters.append(c)
                mandatory_char_ids.add(pov.pov_character_id)

        mandatory_locations: List[Location] = []
        mandatory_loc_ids: Set[str] = set()
        if pov and pov.location_id:
            loc = db.get_location_by_id(pov.location_id, story_id, db_path)
            if loc:
                mandatory_locations.append(loc)
                mandatory_loc_ids.add(loc.id)

        last_summary = db.get_last_chapter_summary(story_id, state.chapter_number, db_path)

        # Character roster — all characters, lightweight
        all_characters = db.get_all_characters(story_id, db_path)
        roster: List[CharacterRosterEntry] = [
            CharacterRosterEntry(
                id=c.id,
                name=c.name,
                is_alive=c.is_alive,
                current_location_id=c.current_location_id,
            )
            for c in all_characters
        ]

        # ------------------------------------------------------------------ #
        # VECTOR PASS                                                          #
        # ------------------------------------------------------------------ #
        # Blend user_input with last chapter summary so that a bare "continue"
        # is anchored to where the story left off rather than matching nothing.
        query_text = state.user_input
        if last_summary:
            query_text = query_text + "\n" + last_summary.medium_summary

        query_vec = _embedder().embed(query_text)
        all_scores: Dict[str, float] = {}

        def _search(entity_type: str) -> List[Tuple[str, float]]:
            threshold = cfg.vector_hit_threshold.get(entity_type, 0.75)
            max_res = cfg.max_results_per_type.get(entity_type, 5)
            hits = vs.vector_search(client, entity_type, query_vec, story_id, threshold, max_res)
            all_scores.update({eid: score for eid, score in hits})
            return hits

        char_hits = _search("character")
        plot_hits = _search("plotline")
        loc_hits = _search("location")
        # world_rule and world_lore skipped — all loaded in mandatory pass above

        # Fetch full entities, skip anything already in mandatory set
        mandatory_plot_ids: Set[str] = {p.id for p in active_plotlines}

        vector_characters: List[Character] = _fetch_new(
            char_hits, mandatory_char_ids,
            lambda eid: db.get_character_by_id(eid, story_id, db_path),
        )
        vector_plotlines: List[Plotline] = _fetch_new(
            plot_hits, mandatory_plot_ids,
            lambda eid: db.get_plotline_by_id(eid, story_id, db_path),
        )
        vector_locations: List[Location] = _fetch_new(
            loc_hits, mandatory_loc_ids,
            lambda eid: db.get_location_by_id(eid, story_id, db_path),
        )

        # NAME-MATCH PASS: if a character's name appears literally in the user
        # input, include them regardless of vector score. Covers the common case
        # where the user types "Kael confronts Mira" and Mira misses the threshold.
        user_input_lower = state.user_input.lower()
        all_char_ids_so_far = mandatory_char_ids | {c.id for c in vector_characters}
        name_matched_characters: List[Character] = []
        for c in all_characters:
            if c.id in all_char_ids_so_far:
                continue
            if c.name.lower() in user_input_lower:
                name_matched_characters.append(c)
                all_char_ids_so_far.add(c.id)

        # Merge
        characters = mandatory_characters + vector_characters + name_matched_characters
        plotlines = active_plotlines + vector_plotlines
        locations = mandatory_locations + vector_locations
        world_lore = all_world_lore

        # ------------------------------------------------------------------ #
        # DEPENDENCY GRAPH PASS                                                #
        # ------------------------------------------------------------------ #
        all_found_ids = (
            [c.id for c in characters]
            + [p.id for p in plotlines]
            + [l.id for l in locations]
            + [r.id for r in all_world_rules]
            + [w.id for w in world_lore]
        )
        injected_ids: Set[str] = set(all_found_ids)
        canon_rules = db.get_canon_rules_triggered_by(story_id, all_found_ids, db_path)

        dep_hits: List[DependencyGraphHit] = []
        for rule in canon_rules:
            if rule.inject_entity_id in injected_ids:
                continue
            entity = _fetch_by_type(rule, story_id, db_path)
            if entity is None:
                continue
            injected_ids.add(rule.inject_entity_id)
            _inject_entity(entity, rule.inject_entity_type,
                           characters, plotlines, locations, all_world_rules, world_lore)
            dep_hits.append(DependencyGraphHit(
                rule_id=rule.rule_id,
                reason=rule.reason,
                content=_entity_summary(entity),
            ))

        # ------------------------------------------------------------------ #
        # ASSEMBLE                                                             #
        # ------------------------------------------------------------------ #
        pack = ContextPack(
            pov_state=pov,
            character_roster=roster,
            active_characters=characters,
            active_plotlines=plotlines,
            nearby_locations=locations,
            relevant_world_rules=all_world_rules,
            relevant_world_lore=world_lore,
            last_chapter_summary=last_summary,
            dependency_graph_hits=dep_hits,
            vector_search_scores=all_scores,
            stale_plotlines=stale_plotlines,
        )
        return {"context_pack": pack}

    return node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_all(hits: List[Tuple[str, float]], getter) -> list:
    out = []
    for eid, _ in hits:
        e = getter(eid)
        if e is not None:
            out.append(e)
    return out


def _fetch_new(
    hits: List[Tuple[str, float]],
    already_have: Set[str],
    getter,
) -> list:
    """Fetch entities from hits, skipping IDs already in already_have."""
    out = []
    for eid, _ in hits:
        if eid in already_have:
            continue
        e = getter(eid)
        if e is not None:
            already_have.add(eid)
            out.append(e)
    return out


def _fetch_by_type(rule: CanonRule, story_id: str, db_path: Optional[Path]):
    t = rule.inject_entity_type
    eid = rule.inject_entity_id
    if t == "character":
        return db.get_character_by_id(eid, story_id, db_path)
    if t == "plotline":
        return db.get_plotline_by_id(eid, story_id, db_path)
    if t == "location":
        return db.get_location_by_id(eid, story_id, db_path)
    if t == "world_rule":
        return db.get_world_rule_by_id(eid, db_path)
    if t == "world_lore":
        return db.get_world_lore_by_id(eid, db_path)
    return None


def _inject_entity(entity, entity_type: str, characters, plotlines,
                   locations, world_rules, world_lore) -> None:
    if entity_type == "character":
        characters.append(entity)
    elif entity_type == "plotline":
        plotlines.append(entity)
    elif entity_type == "location":
        locations.append(entity)
    elif entity_type == "world_rule":
        world_rules.append(entity)
    elif entity_type == "world_lore":
        world_lore.append(entity)


def _entity_summary(entity) -> str:
    if hasattr(entity, "name") and hasattr(entity, "personality"):
        return f"Character: {entity.name}"
    if hasattr(entity, "name") and hasattr(entity, "progress_stage"):
        return f"Plotline: {entity.name} ({entity.progress_stage})"
    if hasattr(entity, "name") and hasattr(entity, "description"):
        return f"Location: {entity.name}"
    if hasattr(entity, "title") and hasattr(entity, "rule_type"):
        return f"WorldRule: {entity.title}"
    if hasattr(entity, "title") and hasattr(entity, "category"):
        return f"WorldLore: {entity.title}"
    return str(entity)


# Production singleton
context_builder_node = make_context_builder_node()
