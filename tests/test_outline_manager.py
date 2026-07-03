"""
Tests for node_outline_manager.py:
  - mechanical (no-LLM) updates: new plotlines become beats, a resolved
    plotline marks its beat completed, new characters get placeholder arcs
  - the periodic LLM revision only fires on the configured chapter interval
"""
from __future__ import annotations

import db as db_module
from node_outline_manager import (
    OUTLINE_REVISION_INTERVAL,
    apply_mechanical_outline_updates,
    make_outline_manager_node,
    maybe_revise_outline,
)
from schema import (
    Character, ChapterGraphState, Plotline, PlotlinePatch, StoryBeat, StoryOutline,
)


class _ExplodingClient:
    """Raises if called — used to prove an LLM call was skipped."""
    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        raise AssertionError("LLM should not have been called")


def test_new_plotline_becomes_a_beat(tmp_db_path):
    story_id = "s1"
    db_module.upsert_story_outline(StoryOutline(story_id=story_id), tmp_db_path)

    plot = Plotline(name="The Ledger", status="active", progress_stage="just introduced", current_tension=3)
    state = ChapterGraphState(
        story_id=story_id, chapter_number=3, user_input="x", new_plotlines=[plot],
    )
    apply_mechanical_outline_updates(state, story_id, tmp_db_path)

    outline = db_module.get_story_outline(story_id, tmp_db_path)
    assert len(outline.beats) == 1
    assert outline.beats[0].related_plotline_id == plot.id
    assert outline.beats[0].status == "upcoming"


def test_resolved_plotline_marks_its_beat_completed(tmp_db_path):
    story_id = "s1"
    plot_id = "plot-123"
    outline = StoryOutline(
        story_id=story_id,
        beats=[StoryBeat(description="The ledger surfaces", status="in_progress", related_plotline_id=plot_id)],
    )
    db_module.upsert_story_outline(outline, tmp_db_path)

    state = ChapterGraphState(
        story_id=story_id, chapter_number=10, user_input="x",
        reconciled_patches=[PlotlinePatch(entity_id=plot_id, status="resolved")],
    )
    apply_mechanical_outline_updates(state, story_id, tmp_db_path)

    reloaded = db_module.get_story_outline(story_id, tmp_db_path)
    assert reloaded.beats[0].status == "completed"


def test_new_character_gets_placeholder_arc(tmp_db_path):
    story_id = "s1"
    db_module.upsert_story_outline(StoryOutline(story_id=story_id), tmp_db_path)

    char = Character(name="Mira", personality="sly")
    state = ChapterGraphState(
        story_id=story_id, chapter_number=7, user_input="x", new_characters=[char],
    )
    apply_mechanical_outline_updates(state, story_id, tmp_db_path)

    outline = db_module.get_story_outline(story_id, tmp_db_path)
    assert len(outline.character_arcs) == 1
    assert outline.character_arcs[0].character_id == char.id
    assert "chapter 7" in outline.character_arcs[0].current_stage


def test_mechanical_update_is_a_noop_when_nothing_changed(tmp_db_path):
    story_id = "s1"
    outline = StoryOutline(story_id=story_id, premise="Original premise", version=1)
    db_module.upsert_story_outline(outline, tmp_db_path)

    state = ChapterGraphState(story_id=story_id, chapter_number=4, user_input="x")
    apply_mechanical_outline_updates(state, story_id, tmp_db_path)

    reloaded = db_module.get_story_outline(story_id, tmp_db_path)
    assert reloaded.premise == "Original premise"


def test_hand_authored_premise_gets_beats_on_cold_start(tmp_db_path, fake_ollama_json):
    """An outline created in the World Bible tab (premise but no beats) must
    still trigger beat generation on cold start — keeping the author's premise."""
    story_id = "s1"
    db_module.upsert_story_outline(
        StoryOutline(story_id=story_id, premise="My hand-written premise", theme="Loyalty"),
        tmp_db_path,
    )
    client = fake_ollama_json({
        "premise": "LLM-invented premise that must NOT win",
        "theme": "LLM theme",
        "planned_ending": "Generated ending",
        "beats": [{"description": "Generated beat"}],
        "character_arcs": [],
    })
    node = make_outline_manager_node(ollama_client=client, db_path=tmp_db_path)
    state = ChapterGraphState(story_id=story_id, chapter_number=1, user_input="go", input_mode="cold_start")

    outline = node(state)["story_outline"]
    assert outline.premise == "My hand-written premise"
    assert outline.theme == "Loyalty"
    assert outline.planned_ending == "Generated ending"  # author left it blank — LLM fills it
    assert len(outline.beats) == 1
    # persisted, so the next run loads it without another LLM call
    assert db_module.get_story_outline(story_id, tmp_db_path).premise == "My hand-written premise"


def test_existing_outline_loads_without_llm_mid_story(tmp_db_path):
    """Mid-story (continuation), an outline is loaded as-is even without beats."""
    story_id = "s1"
    db_module.upsert_story_outline(StoryOutline(story_id=story_id, premise="Edited mid-story"), tmp_db_path)
    node = make_outline_manager_node(ollama_client=_ExplodingClient(), db_path=tmp_db_path)
    state = ChapterGraphState(story_id=story_id, chapter_number=5, user_input="x", input_mode="continuation")
    assert node(state)["story_outline"].premise == "Edited mid-story"


def test_seeded_outline_with_beats_loads_without_llm_on_cold_start(tmp_db_path):
    """Seed scripts pre-create full outlines (with beats) — no LLM init call."""
    story_id = "s1"
    db_module.upsert_story_outline(
        StoryOutline(story_id=story_id, premise="Seeded", beats=[StoryBeat(description="b1")]),
        tmp_db_path,
    )
    node = make_outline_manager_node(ollama_client=_ExplodingClient(), db_path=tmp_db_path)
    state = ChapterGraphState(story_id=story_id, chapter_number=1, user_input="x", input_mode="cold_start")
    assert node(state)["story_outline"].premise == "Seeded"


def test_revision_skipped_off_interval_boundary(tmp_db_path):
    story_id = "s1"
    db_module.upsert_story_outline(StoryOutline(story_id=story_id, premise="P"), tmp_db_path)

    off_boundary_chapter = OUTLINE_REVISION_INTERVAL - 1
    state = ChapterGraphState(story_id=story_id, chapter_number=off_boundary_chapter, user_input="x")

    result = maybe_revise_outline(state, db_path=tmp_db_path, ollama_client=_ExplodingClient(), print_fn=lambda *a: None)
    assert result is None


def test_revision_fires_on_interval_boundary(tmp_db_path, fake_ollama_json):
    story_id = "s1"
    db_module.upsert_story_outline(StoryOutline(story_id=story_id, premise="Old premise", version=1), tmp_db_path)

    payload = {
        "premise": "Old premise",
        "theme": "Trust and betrayal",
        "planned_ending": "The truth about the ledger comes out",
        "beats": [{"description": "The ledger is found", "status": "completed"}],
        "character_arcs": [],
    }
    client = fake_ollama_json(payload)

    state = ChapterGraphState(story_id=story_id, chapter_number=OUTLINE_REVISION_INTERVAL, user_input="x")
    result = maybe_revise_outline(state, db_path=tmp_db_path, ollama_client=client, print_fn=lambda *a: None)

    assert result is not None
    assert result.version == 2
    assert result.last_revised_chapter == OUTLINE_REVISION_INTERVAL
    assert result.theme == "Trust and betrayal"
    assert len(result.beats) == 1
