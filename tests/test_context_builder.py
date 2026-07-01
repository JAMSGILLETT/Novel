"""
Tests for node_context_builder.py:
  - the mandatory pass always includes the POV character and their location
  - the stale-plotline audit flags active plotlines untouched for
    STALE_PLOTLINE_THRESHOLD+ chapters, and leaves recently-touched ones alone
"""
from __future__ import annotations

import db as db_module
from node_context_builder import make_context_builder_node, STALE_PLOTLINE_THRESHOLD
from schema import Character, Location, Plotline, POVState, ChapterGraphState


class _FakeEmbedder:
    def embed(self, text):
        return [0.1, 0.2, 0.3, 0.4]


def _build_state(story_id, chapter_number):
    return ChapterGraphState(story_id=story_id, chapter_number=chapter_number, user_input="continue")


def test_mandatory_pass_includes_pov_character_and_location(tmp_db_path, ephemeral_chroma):
    story_id = "s1"
    char = Character(name="Kael", personality="bold")
    loc = Location(name="Docks", description="foggy pier")
    db_module.upsert_character(char, story_id, tmp_db_path)
    db_module.upsert_location(loc, story_id, tmp_db_path)
    db_module.upsert_pov_state(POVState(location_id=loc.id, pov_character_id=char.id), story_id, tmp_db_path)

    node = make_context_builder_node(embedder=_FakeEmbedder(), chroma_client=ephemeral_chroma, db_path=tmp_db_path)
    result = node(_build_state(story_id, 5))
    pack = result["context_pack"]

    assert [c.id for c in pack.active_characters] == [char.id]
    assert [l.id for l in pack.nearby_locations] == [loc.id]


def test_stale_plotline_flagged_after_threshold(tmp_db_path, ephemeral_chroma):
    story_id = "s1"
    stale_plot = Plotline(
        name="The Ledger", status="active", progress_stage="rising", current_tension=5,
        last_touched_chapter=1,
    )
    fresh_plot = Plotline(
        name="The Rival", status="active", progress_stage="brewing", current_tension=3,
        last_touched_chapter=9,
    )
    db_module.upsert_plotline(stale_plot, story_id, tmp_db_path)
    db_module.upsert_plotline(fresh_plot, story_id, tmp_db_path)

    chapter_number = 1 + STALE_PLOTLINE_THRESHOLD  # exactly at the threshold for stale_plot
    node = make_context_builder_node(embedder=_FakeEmbedder(), chroma_client=ephemeral_chroma, db_path=tmp_db_path)
    result = node(_build_state(story_id, chapter_number))
    pack = result["context_pack"]

    stale_ids = {p.id for p in pack.stale_plotlines}
    assert stale_plot.id in stale_ids
    assert fresh_plot.id not in stale_ids


def test_no_stale_plotlines_when_all_recently_touched(tmp_db_path, ephemeral_chroma):
    story_id = "s1"
    plot = Plotline(
        name="The Ledger", status="active", progress_stage="rising", current_tension=5,
        last_touched_chapter=4,
    )
    db_module.upsert_plotline(plot, story_id, tmp_db_path)

    node = make_context_builder_node(embedder=_FakeEmbedder(), chroma_client=ephemeral_chroma, db_path=tmp_db_path)
    result = node(_build_state(story_id, 5))
    assert result["context_pack"].stale_plotlines == []
