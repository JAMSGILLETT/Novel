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


# ---------------------------------------------------------------------------
# Mining real verdicts into demos
# ---------------------------------------------------------------------------

def test_demo_from_verdict_only_trusts_clean_passes():
    from check_demos import demo_from_verdict
    passed = {"passed": True, "world_rules": "[X] no magic", "prose": "She ran fast.",
              "verdict_json": "{}"}
    flagged = {"passed": False, "world_rules": "[X] no magic", "prose": "She cast a spell.",
               "verdict_json": "{}"}
    assert demo_from_verdict(passed) is not None
    assert demo_from_verdict(flagged) is None, "flagged verdicts must not be mined as demos"


def test_mined_clean_pass_preferred_over_seed():
    history = [{"passed": True, "world_rules": "[MAGIC_SYSTEM] no magic",
                "prose": "Sela counted the guards' footsteps and slipped through the gap.",
                "verdict_json": "{}"}]
    demos = select_demos(_pack(["magic_system"]), history=history)
    clean = [d for d in demos if d.role == "clean"]
    assert clean and "Sela counted the guards" in clean[0].prose


def test_check_verdict_db_roundtrip(tmp_db_path):
    import db
    db.init_db(tmp_db_path)
    db.save_check_verdict("s1", 1, True, "[X] rule", "clean prose", '{"canon_passed": true}', tmp_db_path)
    db.save_check_verdict("s1", 2, False, "[X] rule", "bad prose", '{"canon_passed": false}', tmp_db_path)
    rows = db.get_recent_check_verdicts("s1", limit=10, db_path=tmp_db_path)
    assert [r["chapter_number"] for r in rows] == [2, 1]  # newest first
    assert rows[1]["passed"] is True and rows[0]["passed"] is False
    # Idempotent upsert: re-saving chapter 1 doesn't duplicate.
    db.save_check_verdict("s1", 1, False, "[X] rule", "revised", "{}", tmp_db_path)
    rows = db.get_recent_check_verdicts("s1", db_path=tmp_db_path)
    assert len(rows) == 2 and rows[1]["passed"] is False
