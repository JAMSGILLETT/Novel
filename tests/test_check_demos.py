"""Tests for DSPy-lite few-shot demo selection (check_demos)."""
from __future__ import annotations

from schema import ContextPack, WorldRule
from check_demos import select_demos, format_demos, DEMO_BANK


def _pack(rule_types):
    return ContextPack(
        pov_state=None, character_roster=[], active_characters=[],
        active_plotlines=[], nearby_locations=[],
        relevant_world_rules=[
            WorldRule(id=f"r{i}", rule_type=rt, title="t", content="c")
            for i, rt in enumerate(rule_types)
        ],
        relevant_world_lore=[], last_chapter_summary=None,
    )


def test_always_includes_clean_and_craft():
    demos = select_demos(_pack([]))
    roles = {d.role for d in demos}
    assert "clean" in roles, "a clean-pass demo must always be present (anti over-flagging)"
    assert "craft" in roles


def test_matching_canon_demo_is_preferred():
    # A magic-system rule is present → the magic-system canon demo should be the
    # canon example chosen, not an unrelated one.
    demos = select_demos(_pack(["magic_system"]))
    canon = [d for d in demos if d.role == "canon"]
    assert canon and "magic_system" in canon[0].rule_tags


def test_respects_k_cap():
    assert len(select_demos(_pack(["magic_system", "social_rule"]), k=2)) == 2


def test_format_is_nonempty_and_shows_json():
    block = format_demos(select_demos(_pack(["magic_system"])))
    assert "WORKED EXAMPLES" in block
    assert '"canon_passed"' in block  # demos render the correct JSON verdict
    assert "CHAPTER PROSE" in block


def test_format_empty_when_no_demos():
    assert format_demos([]) == ""
