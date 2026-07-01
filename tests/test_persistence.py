"""
Tests for node_persistence.py:
  - the is_alive fix (a CharacterPatch with is_alive=False must actually
    persist as Character.is_alive=False, and the context builder's roster
    must reflect it)
  - the single-transaction guarantee (a failure partway through must roll
    back every write from that chapter, not leave a half-updated DB)
"""
from __future__ import annotations

import pytest

import db as db_module
from node_context_builder import make_context_builder_node
from node_persistence import make_persistence_node
from schema import Character, CharacterPatch, ChapterGraphState


def _make_state(story_id, chapter_number, reconciled_patches):
    return ChapterGraphState(
        story_id=story_id,
        chapter_number=chapter_number,
        user_input="x",
        reconciled_patches=reconciled_patches,
    )


class _FakeEmbedder:
    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def embed(self, text):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated embedding failure")
        return [0.1, 0.2, 0.3, 0.4]


def test_is_alive_patch_persists_as_false(tmp_db_path, ephemeral_chroma):
    story_id = "s1"
    char = Character(name="Kael", personality="bold", is_alive=True)
    db_module.upsert_character(char, story_id, tmp_db_path)

    state = _make_state(story_id, 5, [CharacterPatch(entity_id=char.id, is_alive=False)])
    node = make_persistence_node(chroma_client=ephemeral_chroma, embedder=_FakeEmbedder(), db_path=tmp_db_path)
    node(state)

    reloaded = db_module.get_character_by_id(char.id, story_id, tmp_db_path)
    assert reloaded.is_alive is False
    assert reloaded.emotional_state == "deceased"


def test_dead_character_shows_deceased_in_roster(tmp_db_path, ephemeral_chroma):
    story_id = "s1"
    char = Character(name="Kael", personality="bold", is_alive=True)
    db_module.upsert_character(char, story_id, tmp_db_path)

    state = _make_state(story_id, 5, [CharacterPatch(entity_id=char.id, is_alive=False)])
    persistence_node = make_persistence_node(chroma_client=ephemeral_chroma, embedder=_FakeEmbedder(), db_path=tmp_db_path)
    persistence_node(state)

    context_state = ChapterGraphState(story_id=story_id, chapter_number=6, user_input="continue")
    context_node = make_context_builder_node(embedder=_FakeEmbedder(), chroma_client=ephemeral_chroma, db_path=tmp_db_path)
    result = context_node(context_state)
    roster = result["context_pack"].character_roster
    entry = next(e for e in roster if e.id == char.id)
    assert entry.is_alive is False


def test_persistence_is_atomic_rolls_back_on_failure(tmp_db_path, ephemeral_chroma):
    story_id = "s1"
    char1 = Character(name="Kael", personality="bold", emotional_state="calm")
    char2 = Character(name="Mira", personality="sly", emotional_state="calm")
    db_module.upsert_character(char1, story_id, tmp_db_path)
    db_module.upsert_character(char2, story_id, tmp_db_path)

    patches = [
        CharacterPatch(entity_id=char1.id, emotional_state="furious"),
        CharacterPatch(entity_id=char2.id, emotional_state="terrified"),
    ]
    state = _make_state(story_id, 5, patches)

    # Fails on the second embed call — char1's write happens first but must
    # still be rolled back since nothing commits until the whole node succeeds.
    node = make_persistence_node(chroma_client=ephemeral_chroma, embedder=_FakeEmbedder(fail_after=1), db_path=tmp_db_path)
    with pytest.raises(RuntimeError):
        node(state)

    reloaded1 = db_module.get_character_by_id(char1.id, story_id, tmp_db_path)
    reloaded2 = db_module.get_character_by_id(char2.id, story_id, tmp_db_path)
    assert reloaded1.emotional_state == "calm", "char1's write should have rolled back with the rest of the transaction"
    assert reloaded2.emotional_state == "calm"

    # And the chapter must not have been registered either
    assert db_module.get_latest_chapter_number(story_id, tmp_db_path) is None
