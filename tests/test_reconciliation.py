"""
Tests for node_reconciliation.py — pure logic, no LLM, so this is the
highest-value place to test thoroughly.
"""
from __future__ import annotations

from node_reconciliation import reconcile
from schema import CharacterPatch, ChapterGraphState, PlotlinePatch


def _state_with_patches(patches):
    return ChapterGraphState(
        story_id="s1", chapter_number=1, user_input="x", memory_patches=patches,
    )


def test_is_alive_conflict_dead_wins():
    patches = [
        CharacterPatch(entity_id="char-1", is_alive=True),
        CharacterPatch(entity_id="char-1", is_alive=False),
    ]
    result = reconcile(_state_with_patches(patches))
    merged = result["reconciled_patches"][0]
    assert merged.is_alive is False
    assert len(result["reconciliation_conflicts"]) == 1
    assert result["reconciliation_conflicts"][0].resolved_by_rule == "dead_wins"


def test_is_alive_conflict_dead_wins_regardless_of_order():
    patches = [
        CharacterPatch(entity_id="char-1", is_alive=False),
        CharacterPatch(entity_id="char-1", is_alive=True),
    ]
    result = reconcile(_state_with_patches(patches))
    merged = result["reconciled_patches"][0]
    assert merged.is_alive is False


def test_location_conflict_last_writer_wins():
    patches = [
        CharacterPatch(entity_id="char-1", current_location_id="loc-a"),
        CharacterPatch(entity_id="char-1", current_location_id="loc-b"),
    ]
    result = reconcile(_state_with_patches(patches))
    merged = result["reconciled_patches"][0]
    assert merged.current_location_id == "loc-b"
    assert result["reconciliation_conflicts"][0].resolved_by_rule == "last_writer_wins"


def test_character_additive_fields_union_without_duplicates():
    patches = [
        CharacterPatch(entity_id="char-1", knowledge_added=["fact-a", "fact-b"]),
        CharacterPatch(entity_id="char-1", knowledge_added=["fact-b", "fact-c"]),
    ]
    result = reconcile(_state_with_patches(patches))
    merged = result["reconciled_patches"][0]
    assert merged.knowledge_added == ["fact-a", "fact-b", "fact-c"]


def test_plotline_status_conflict_most_advanced_wins():
    patches = [
        PlotlinePatch(entity_id="plot-1", status="active"),
        PlotlinePatch(entity_id="plot-1", status="resolved"),
    ]
    result = reconcile(_state_with_patches(patches))
    merged = result["reconciled_patches"][0]
    assert merged.status == "resolved"
    assert result["reconciliation_conflicts"][0].resolved_by_rule == "most_advanced_status"


def test_plotline_status_dormant_loses_to_active():
    patches = [
        PlotlinePatch(entity_id="plot-1", status="dormant"),
        PlotlinePatch(entity_id="plot-1", status="active"),
    ]
    result = reconcile(_state_with_patches(patches))
    merged = result["reconciled_patches"][0]
    assert merged.status == "active"


def test_single_patch_per_entity_passes_through_unchanged_no_conflicts():
    patches = [CharacterPatch(entity_id="char-1", emotional_state="relieved")]
    result = reconcile(_state_with_patches(patches))
    assert len(result["reconciled_patches"]) == 1
    assert result["reconciled_patches"][0].emotional_state == "relieved"
    assert result["reconciliation_conflicts"] == []


def test_different_entities_do_not_interfere():
    patches = [
        CharacterPatch(entity_id="char-1", emotional_state="happy"),
        CharacterPatch(entity_id="char-2", emotional_state="sad"),
    ]
    result = reconcile(_state_with_patches(patches))
    by_id = {p.entity_id: p for p in result["reconciled_patches"]}
    assert by_id["char-1"].emotional_state == "happy"
    assert by_id["char-2"].emotional_state == "sad"
