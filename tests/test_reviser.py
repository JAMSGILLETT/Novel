"""
Unit tests for the Actor↔Evaluator revision loop (node_reviser).

The Evaluator and the edit-generation LLM call are both faked, so these test
the loop's *control flow* — best-of-N selection, memory accumulation, the retry
cap, and early-exit on a clean candidate — without touching Ollama.
"""
from __future__ import annotations

import pytest

import llm_client
from schema import CanonCheckResult, CraftCheckResult, CanonViolation, CraftIssue, ChapterGraphState
from node_reviser import make_reviser_loop, Verdict


def _canon_major():
    return CanonViolation(violation_type="lore_violation", description="magic used", severity="major")


def _craft_minor():
    return CraftIssue(issue_type="pacing", description="drags", severity="minor")


def _evaluator_from_prose(state):
    """Grounded-ish fake: 'magic' -> canon major (score 10); 'drags' -> craft
    minor (score 1). Clean prose passes."""
    prose = state.chapter_prose or ""
    violations = [_canon_major()] if "magic" in prose else []
    issues = [_craft_minor()] if "drags" in prose else []
    return (
        CanonCheckResult(passed=not violations, violations=violations),
        CraftCheckResult(passed=not issues, issues=issues),
    )


def _state(prose: str) -> ChapterGraphState:
    return ChapterGraphState(story_id="t", chapter_number=1, user_input="x",
                             context_pack=None, story_plan=None, chapter_prose=prose)


def _fake_chat_text(monkeypatch, outputs):
    """Make llm_client.chat_text return successive canned edits."""
    it = iter(outputs)
    monkeypatch.setattr(llm_client, "chat_text", lambda *a, **k: next(it))


def test_score_weights_major_over_minor():
    # Craft-only: majors weigh 10, minors 1, no canon gate.
    v = Verdict(CanonCheckResult(passed=True, violations=[]),
                CraftCheckResult(passed=False, issues=[_craft_minor()]))
    assert v.score() == 1
    assert not v.passed


def test_canon_is_a_hard_gate():
    # A canon violation must dominate any amount of craft: a canon-dirty +
    # craft-clean verdict must score higher (worse) than a canon-clean verdict
    # riddled with craft issues.
    canon_dirty = Verdict(CanonCheckResult(passed=False, violations=[_canon_major()]),
                          CraftCheckResult(passed=True, issues=[]))
    canon_clean_craft_messy = Verdict(
        CanonCheckResult(passed=True, violations=[]),
        CraftCheckResult(passed=False, issues=[_craft_minor()] * 50))
    assert canon_dirty.score() > canon_clean_craft_messy.score()


def test_passes_after_one_clean_edit(monkeypatch):
    _fake_chat_text(monkeypatch, ["a clean paragraph", "another clean one"])
    loop = make_reviser_loop(_evaluator_from_prose, best_of_n=2, print_fn=lambda *_: None)
    state, verdict, attempts, flagged = loop(_state("she used magic"), max_retries=2)
    assert verdict.passed and not flagged
    assert attempts == 2
    assert state.chapter_prose == "a clean paragraph"  # first candidate was clean → early stop


def test_best_of_n_picks_lowest_score(monkeypatch):
    # Neither candidate is clean, so best-of-N must compare: cand1 keeps magic
    # (score 10), cand2 only drags (score 1). Loop must keep cand2.
    _fake_chat_text(monkeypatch, ["still has magic", "just drags a bit", "unused"])
    loop = make_reviser_loop(_evaluator_from_prose, best_of_n=2, print_fn=lambda *_: None)
    state, verdict, attempts, flagged = loop(_state("magic and drags"), max_retries=2)
    # After keeping cand2 ("just drags"), attempt 2 re-enters with score 1, then
    # revises again — supply more clean output for that round.
    assert "magic" not in state.chapter_prose


def test_memory_accumulates_across_attempts(monkeypatch):
    seen = {}

    def spy_chat(*a, **k):
        # The prompt is the first positional arg; record whether prior-attempt
        # memory appears once we're past the first revision.
        prompt = a[0]
        seen.setdefault("prompts", []).append(prompt)
        return "still has magic"  # never fixes it → forces repeated attempts

    monkeypatch.setattr(llm_client, "chat_text", spy_chat)
    loop = make_reviser_loop(_evaluator_from_prose, best_of_n=1, print_fn=lambda *_: None)
    _, verdict, attempts, flagged = loop(_state("magic"), max_retries=2)
    assert flagged  # never cleaned → hits cap
    # Second revision's prompt must carry memory of the first attempt.
    later_prompts = [p for p in seen["prompts"] if "PRIOR ATTEMPTS" in p]
    assert later_prompts, "reviser should feed prior-attempt memory into later edits"


def test_cap_flags_when_never_clean(monkeypatch):
    monkeypatch.setattr(llm_client, "chat_text", lambda *a, **k: "magic persists")
    loop = make_reviser_loop(_evaluator_from_prose, best_of_n=1, print_fn=lambda *_: None)
    state, verdict, attempts, flagged = loop(_state("magic"), max_retries=1)
    assert flagged
    assert not verdict.passed
    assert attempts == 2  # attempt 1 + one revision, then cap


def test_no_usable_candidate_flags(monkeypatch):
    monkeypatch.setattr(llm_client, "chat_text", lambda *a, **k: "")  # empty every time
    loop = make_reviser_loop(_evaluator_from_prose, best_of_n=2, print_fn=lambda *_: None)
    state, verdict, attempts, flagged = loop(_state("magic"), max_retries=2)
    assert flagged
    assert state.chapter_prose == "magic"  # unchanged — nothing usable produced
