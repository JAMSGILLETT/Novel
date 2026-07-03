"""Control-flow tests for the craft Self-Refine node (node_self_refine).

The LLM call is stubbed, so these exercise stop conditions and safety — not
prose quality. Note node_self_refine binds `chat_text` at import, so we patch
the name on that module, not on llm_client.
"""
from __future__ import annotations

import node_self_refine
from schema import ChapterGraphState, StoryPlan


def _state(prose: str) -> ChapterGraphState:
    plan = StoryPlan(scenes=["s"], pacing_notes="", conflicts=[], narrative_goals=[],
                     character_constraints=[], target_word_count=1000)
    return ChapterGraphState(story_id="t", chapter_number=1, user_input="x",
                             context_pack=None, story_plan=plan, chapter_prose=prose)


def _stub_chat(monkeypatch, outputs):
    calls = {"n": 0}
    it = iter(outputs)

    def fake(*a, **k):
        calls["n"] += 1
        return next(it)

    monkeypatch.setattr(node_self_refine, "chat_text", fake)
    return calls


def test_disabled_by_env_returns_unchanged(monkeypatch):
    monkeypatch.setenv("NOVELGEN_SELF_REFINE", "0")
    calls = _stub_chat(monkeypatch, [])
    node = node_self_refine.make_self_refine_node(print_fn=lambda *_: None)
    out = node(_state("original prose here"))
    assert out["chapter_prose"] == "original prose here"
    assert calls["n"] == 0, "disabled must make no LLM call"


def test_clean_sentinel_stops_immediately(monkeypatch):
    monkeypatch.setenv("NOVELGEN_SELF_REFINE", "1")
    calls = _stub_chat(monkeypatch, ["NO_CHANGES"])
    node = node_self_refine.make_self_refine_node(print_fn=lambda *_: None)
    out = node(_state("already good prose that needs nothing"))
    assert out["chapter_prose"] == "already good prose that needs nothing"
    assert calls["n"] == 1


def test_applies_improvement_then_stops_on_sentinel(monkeypatch):
    monkeypatch.setenv("NOVELGEN_SELF_REFINE", "1")
    improved = "A markedly improved and clearly different paragraph of prose entirely."
    _stub_chat(monkeypatch, [improved, "NO_CHANGES"])
    node = node_self_refine.make_self_refine_node(print_fn=lambda *_: None)
    out = node(_state("weak prose"))
    assert out["chapter_prose"] == improved


def test_respects_max_iters(monkeypatch):
    monkeypatch.setenv("NOVELGEN_SELF_REFINE", "1")
    # Each call returns a distinct, sufficiently-changed draft; cap must stop it.
    outs = [f"draft version number {i} with plenty of new distinct words here" for i in range(10)]
    calls = _stub_chat(monkeypatch, outs)
    node = node_self_refine.make_self_refine_node(max_iters=2, print_fn=lambda *_: None)
    node(_state("start"))
    assert calls["n"] == 2, "must not exceed max_iters LLM calls"


def test_negligible_change_stops(monkeypatch):
    monkeypatch.setenv("NOVELGEN_SELF_REFINE", "1")
    prose = "the quick brown fox jumps over the lazy dog again and again today"
    calls = _stub_chat(monkeypatch, [prose, prose])  # identical output = no-op
    node = node_self_refine.make_self_refine_node(print_fn=lambda *_: None)
    out = node(_state(prose))
    assert out["chapter_prose"] == prose
    assert calls["n"] == 1, "an unchanged draft should stop after the first call"


def test_llm_error_keeps_current_draft(monkeypatch):
    monkeypatch.setenv("NOVELGEN_SELF_REFINE", "1")

    def boom(*a, **k):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(node_self_refine, "chat_text", boom)
    node = node_self_refine.make_self_refine_node(print_fn=lambda *_: None)
    out = node(_state("keep me"))
    assert out["chapter_prose"] == "keep me"


def test_db_setting_toggles_enabled(monkeypatch, tmp_db_path):
    import db
    db.init_db(tmp_db_path)
    monkeypatch.delenv("NOVELGEN_SELF_REFINE", raising=False)
    # Default (unset) → enabled.
    assert node_self_refine.is_enabled(tmp_db_path) is True
    # Stored "0" → disabled.
    db.set_setting("self_refine", "0", tmp_db_path)
    assert node_self_refine.is_enabled(tmp_db_path) is False
    # Stored "1" → enabled again.
    db.set_setting("self_refine", "1", tmp_db_path)
    assert node_self_refine.is_enabled(tmp_db_path) is True


def test_env_zero_overrides_db_setting(monkeypatch, tmp_db_path):
    import db
    db.init_db(tmp_db_path)
    db.set_setting("self_refine", "1", tmp_db_path)  # stored ON
    monkeypatch.setenv("NOVELGEN_SELF_REFINE", "0")  # env forces OFF
    assert node_self_refine.is_enabled(tmp_db_path) is False


def test_empty_prose_is_noop(monkeypatch):
    monkeypatch.setenv("NOVELGEN_SELF_REFINE", "1")
    calls = _stub_chat(monkeypatch, [])
    node = node_self_refine.make_self_refine_node(print_fn=lambda *_: None)
    out = node(_state(""))
    assert out["chapter_prose"] == ""
    assert calls["n"] == 0
