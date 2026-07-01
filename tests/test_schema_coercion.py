"""
Tests for schema.py's lenient-input coercion — the layer that absorbs small
local models' habit of returning list fields as JSON strings or
single-key-dict envelopes instead of real JSON arrays.
"""
from __future__ import annotations

from schema import CharacterConstraint, StoryPlan, _coerce_str_list


def test_coerce_str_list_passes_through_real_list():
    assert _coerce_str_list(["a", "b"]) == ["a", "b"]


def test_coerce_str_list_parses_json_string():
    assert _coerce_str_list('["a", "b"]') == ["a", "b"]


def test_coerce_str_list_repairs_malformed_json_string():
    # single-quoted / trailing comma — not strict JSON, needs json_repair
    result = _coerce_str_list("['a', 'b',]")
    assert result == ["a", "b"]


def test_coerce_str_list_unwraps_single_key_dict():
    assert _coerce_str_list({"items": ["a", "b"]}) == ["a", "b"]


def test_coerce_str_list_dict_with_no_list_values_returns_values():
    # No list-valued key to unwrap — falls back to the dict's values
    assert _coerce_str_list({"a": 1, "b": 2}) == [1, 2]


def test_story_plan_coerces_scenes_from_json_string():
    plan = StoryPlan(
        scenes='["Scene one", "Scene two"]',
        pacing_notes="tense",
        conflicts=[],
        narrative_goals=[],
        character_constraints=[],
    )
    assert plan.scenes == ["Scene one", "Scene two"]


def test_story_plan_target_word_count_defaults_and_bounds():
    plan = StoryPlan(
        scenes=["a"], pacing_notes="", conflicts=[], narrative_goals=[], character_constraints=[],
    )
    assert plan.target_word_count == 1000


def test_character_constraint_default_id_is_overwritable():
    # 8B models often omit character_id; reasoner is expected to overwrite it
    cc = CharacterConstraint()
    assert cc.character_id == ""
