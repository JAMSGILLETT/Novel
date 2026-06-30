"""
Tests for Node 2: Context Builder.

No network calls, no model downloads:
  - FakeEmbedder returns a fixed 384-dim unit vector.
  - EphemeralClient() for Chroma (in-memory).
  - Throwaway SQLite in a tempfile.

Run: python test_context_builder.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import db
import vector_store as vs
from embeddings import Embedder
from node_context_builder import make_context_builder_node
from schema import (
    CanonRule, Character, ChapterSummary, Location,
    Plotline, POVState, WorldLore, WorldRule, ChapterGraphState,
)

DIM = 384


class FakeEmbedder(Embedder):
    def embed(self, text: str) -> List[float]:
        vec = [0.0] * DIM
        vec[0] = 1.0
        return vec


def _ephemeral_client():
    return vs.get_ephemeral_client()


def _tmp_db() -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return Path(f.name)


def _upsert_to_chroma(client, entity_type, entity_id, text, story_id=None):
    embedder = FakeEmbedder()
    vs.upsert_entity(client, entity_type, entity_id,
                     embedder.embed(text), text, story_id=story_id)


def _seed(db_path, client, story_id):
    db.init_db(db_path)

    rule = WorldRule(rule_type="magic_system", title="Bloodbinding",
                     content="Magic requires a blood price")
    lore = WorldLore(category="history", title="The Founding War",
                     content="The city was built on a battlefield")
    alice = Character(name="Alice", personality="Cautious", goals=["Find Marcus"])
    plot = Plotline(name="The Search", progress_stage="rising_action", current_tension=6,
                    involved_character_ids=[alice.id], status="active")
    loc = Location(name="Riverside Market", description="A crowded trade hub")
    pov = POVState(location_id=loc.id, companions=[alice.id])
    summary = ChapterSummary(chapter_number=1, short_summary="Ch1 short",
                              medium_summary="Alice arrived at the market seeking clues about Marcus")

    db.upsert_world_rule(rule, db_path)
    db.upsert_world_lore(lore, db_path)
    db.upsert_character(alice, story_id, db_path)
    db.upsert_plotline(plot, story_id, db_path)
    db.upsert_location(loc, story_id, db_path)
    db.upsert_pov_state(pov, story_id, db_path)
    db.upsert_chapter_summary(summary, story_id, db_path)

    _upsert_to_chroma(client, "world_rule", rule.id, vs.world_rule_text(rule))
    _upsert_to_chroma(client, "world_lore", lore.id, vs.world_lore_text(lore))
    _upsert_to_chroma(client, "character", alice.id, vs.character_text(alice), story_id=story_id)
    _upsert_to_chroma(client, "plotline", plot.id, vs.plotline_text(plot), story_id=story_id)
    _upsert_to_chroma(client, "location", loc.id, vs.location_text(loc), story_id=story_id)

    return rule, lore, alice, plot, loc, pov, summary


def test_mandatory_always_present():
    """Active plotlines, world rules, POV location, POV companions, and last summary
    are always present regardless of the vector search query."""
    story_id = "story-mandatory-001"
    db_path = _tmp_db()
    client = _ephemeral_client()
    rule, lore, alice, plot, loc, pov, summary = _seed(db_path, client, story_id)

    node = make_context_builder_node(
        embedder=FakeEmbedder(), chroma_client=client, db_path=db_path
    )
    # Generic "continue" — would pull nothing semantically on its own
    state = ChapterGraphState(
        story_id=story_id, chapter_number=2,
        input_mode="continuation", user_input="continue",
    )
    result = node(state)
    pack = result["context_pack"]

    # World rules always present
    assert len(pack.relevant_world_rules) == 1
    assert pack.relevant_world_rules[0].title == "Bloodbinding"
    # Active plotline always present
    assert any(p.name == "The Search" for p in pack.active_plotlines)
    # POV companion (Alice) always present in active_characters
    assert any(c.name == "Alice" for c in pack.active_characters)
    # POV location always present
    assert any(l.name == "Riverside Market" for l in pack.nearby_locations)
    # Last chapter summary always present
    assert pack.last_chapter_summary is not None
    assert pack.last_chapter_summary.chapter_number == 1
    print("test_mandatory_always_present OK")


def test_character_roster_always_complete():
    """Roster contains all characters, including ones not retrieved by vector search."""
    story_id = "story-roster-001"
    db_path = _tmp_db()
    client = _ephemeral_client()
    db.init_db(db_path)

    alice = Character(name="Alice", personality="Cautious")
    bob = Character(name="Bob", personality="Reckless")
    old_mentor = Character(name="The Mentor", personality="Wise")

    # Only Alice goes into Chroma; Bob and Mentor are SQLite-only (off-screen)
    db.upsert_character(alice, story_id, db_path)
    db.upsert_character(bob, story_id, db_path)
    db.upsert_character(old_mentor, story_id, db_path)
    _upsert_to_chroma(client, "character", alice.id,
                      vs.character_text(alice), story_id=story_id)

    node = make_context_builder_node(
        embedder=FakeEmbedder(), chroma_client=client, db_path=db_path
    )
    state = ChapterGraphState(
        story_id=story_id, chapter_number=2,
        input_mode="continuation", user_input="Alice walks through the market",
    )
    result = node(state)
    pack = result["context_pack"]

    roster_names = {e.name for e in pack.character_roster}
    assert "Alice" in roster_names
    assert "Bob" in roster_names
    assert "The Mentor" in roster_names, "Off-screen character must still appear in roster"
    print("test_character_roster_always_complete OK —",
          f"roster has {len(pack.character_roster)} characters, "
          f"active_characters has {len(pack.active_characters)}")


def test_continue_query_blended_with_summary():
    """'continue' blended with last chapter summary finds relevant entities."""
    story_id = "story-blend-001"
    db_path = _tmp_db()
    client = _ephemeral_client()
    rule, lore, alice, plot, loc, pov, summary = _seed(db_path, client, story_id)

    node = make_context_builder_node(
        embedder=FakeEmbedder(), chroma_client=client, db_path=db_path
    )
    state = ChapterGraphState(
        story_id=story_id, chapter_number=2,
        input_mode="continuation", user_input="continue",
    )
    result = node(state)
    pack = result["context_pack"]

    # With blended query, Alice (in Chroma) should be found
    assert any(c.name == "Alice" for c in pack.active_characters)
    assert pack.last_chapter_summary is not None
    print("test_continue_query_blended_with_summary OK — "
          f"found {len(pack.active_characters)} characters despite bare 'continue'")


def test_cold_start_has_world_rules_no_pov():
    """Cold start: world rules present, pov_state None, empty roster, no summary."""
    story_id = "story-cold-001"
    db_path = _tmp_db()
    client = _ephemeral_client()
    db.init_db(db_path)

    rule = WorldRule(rule_type="physics", title="No teleportation",
                     content="Objects cannot teleport")
    db.upsert_world_rule(rule, db_path)
    _upsert_to_chroma(client, "world_rule", rule.id, vs.world_rule_text(rule))

    node = make_context_builder_node(
        embedder=FakeEmbedder(), chroma_client=client, db_path=db_path
    )
    state = ChapterGraphState(
        story_id=story_id, chapter_number=1,
        input_mode="cold_start", user_input="Begin the story",
    )
    result = node(state)
    pack = result["context_pack"]

    assert pack.pov_state is None
    assert pack.active_characters == []
    assert pack.character_roster == []
    assert len(pack.relevant_world_rules) == 1
    assert pack.last_chapter_summary is None
    print("test_cold_start_has_world_rules_no_pov OK")


def test_dependency_graph_injection():
    """Entity reachable only via canon rule is injected and recorded."""
    story_id = "story-dep-001"
    db_path = _tmp_db()
    client = _ephemeral_client()
    db.init_db(db_path)

    alice = Character(name="Alice", personality="Cautious")
    bob = Character(name="Bob", personality="Reckless")
    db.upsert_character(alice, story_id, db_path)
    db.upsert_character(bob, story_id, db_path)
    _upsert_to_chroma(client, "character", alice.id,
                      vs.character_text(alice), story_id=story_id)
    # Bob NOT in Chroma — only reachable via canon rule

    rule = CanonRule(
        rule_id="dep-alice-bob", story_id=story_id,
        trigger_entity_type="character", trigger_entity_id=alice.id,
        inject_entity_type="character", inject_entity_id=bob.id,
        reason="Bob owes Alice a debt",
    )
    db.insert_canon_rule(rule, db_path)

    node = make_context_builder_node(
        embedder=FakeEmbedder(), chroma_client=client, db_path=db_path
    )
    state = ChapterGraphState(
        story_id=story_id, chapter_number=2,
        input_mode="continuation",
        user_input="Alice confronts someone at the docks",
    )
    result = node(state)
    pack = result["context_pack"]

    char_names = [c.name for c in pack.active_characters]
    assert "Alice" in char_names
    assert "Bob" in char_names, f"Bob should be injected via canon rule, got: {char_names}"
    assert len(pack.dependency_graph_hits) == 1
    assert pack.dependency_graph_hits[0].rule_id == "dep-alice-bob"
    print("test_dependency_graph_injection OK —",
          pack.dependency_graph_hits[0].reason)


def test_no_duplicate_injection():
    """Entity found by mandatory pass is not duplicated by canon rule."""
    story_id = "story-dup-001"
    db_path = _tmp_db()
    client = _ephemeral_client()
    db.init_db(db_path)

    alice = Character(name="Alice", personality="Cautious")
    db.upsert_character(alice, story_id, db_path)
    _upsert_to_chroma(client, "character", alice.id,
                      vs.character_text(alice), story_id=story_id)

    rule = CanonRule(
        rule_id="self-ref", story_id=story_id,
        trigger_entity_type="character", trigger_entity_id=alice.id,
        inject_entity_type="character", inject_entity_id=alice.id,
        reason="Self-reference test",
    )
    db.insert_canon_rule(rule, db_path)

    node = make_context_builder_node(
        embedder=FakeEmbedder(), chroma_client=client, db_path=db_path
    )
    state = ChapterGraphState(
        story_id=story_id, chapter_number=2,
        input_mode="continuation",
        user_input="Alice walks through the market",
    )
    result = node(state)
    pack = result["context_pack"]

    assert len(pack.active_characters) == 1
    assert len(pack.dependency_graph_hits) == 0
    print("test_no_duplicate_injection OK")


if __name__ == "__main__":
    test_mandatory_always_present()
    test_character_roster_always_complete()
    test_continue_query_blended_with_summary()
    test_cold_start_has_world_rules_no_pov()
    test_dependency_graph_injection()
    test_no_duplicate_injection()
    print("\nAll context builder tests passed.")
