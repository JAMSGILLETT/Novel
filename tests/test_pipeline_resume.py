"""
Crash-recovery integration tests for pipeline.run_chapter.

Runs the REAL run_chapter with every LLM node factory stubbed out, simulates a
crash partway through, then re-runs and asserts the pipeline resumes after the
last completed node instead of regenerating the chapter from scratch.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import db as db_module
from schema import (
    CanonCheckResult, ChapterSummary, ContextPack, CraftCheckResult,
    StoryOutline, StoryPlan,
)


# ---------------------------------------------------------------------------
# Stub node factories — each records how many times its node ran, so tests can
# assert which pipeline stages executed vs. were skipped on resume.
# ---------------------------------------------------------------------------

def _stub_context_pack() -> ContextPack:
    return ContextPack(
        pov_state=None,
        character_roster=[],
        active_characters=[],
        active_plotlines=[],
        nearby_locations=[],
        relevant_world_rules=[],
        relevant_world_lore=[],
        last_chapter_summary=None,
    )


def _stub_plan() -> StoryPlan:
    return StoryPlan(
        scenes=["Scene one"], pacing_notes="steady", conflicts=["conflict"],
        narrative_goals=["goal"], character_constraints=[],
    )


class StubNodes:
    """Installs stub factories for every node run_chapter imports, and counts
    calls per stage. `crash_at` makes that stage raise on its first-ever call
    (simulating the process dying there)."""

    def __init__(self, monkeypatch, db_path: Path, crash_at: str | None = None):
        self.calls: dict[str, int] = {}
        self.crash_at = crash_at
        self._db_path = db_path
        mp = monkeypatch

        def count(stage: str):
            self.calls[stage] = self.calls.get(stage, 0) + 1
            if stage == self.crash_at:
                self.crash_at = None  # crash only once
                raise RuntimeError(f"simulated crash in {stage}")

        import node_outline_manager, node_context_builder
        import node_unified_writer, node_combined_check
        import node_chapter_summarizer, node_memory_extractor
        import node_reconciliation, node_persistence
        import llm_client, vector_store, pipeline

        mp.setattr(node_outline_manager, "make_outline_manager_node",
                   lambda **kw: lambda s: (count("outline"), {"story_outline": StoryOutline(story_id=s.story_id)})[1])
        mp.setattr(node_outline_manager, "maybe_revise_outline",
                   lambda s, **kw: (count("revise_outline"), None)[1])
        mp.setattr(node_context_builder, "make_context_builder_node",
                   lambda **kw: lambda s: (count("context"), {"context_pack": _stub_context_pack()})[1])
        # Unified writer replaces the old planner + character-reasoner + writer
        # nodes: one call returns both the plan and the prose. Revisions come
        # back through the same node with violation_feedback set.
        mp.setattr(node_unified_writer, "make_unified_writer_node",
                   lambda **kw: lambda s, violation_feedback=None: (count("write"),
                       {"story_plan": _stub_plan(), "chapter_prose": "Stub prose.",
                        "character_reasonings": []})[1])
        # Combined check replaces the separate canon + craft nodes: one call
        # returns both results. Count both stage keys so resume assertions still
        # see "canon" and "craft" ran exactly once.
        mp.setattr(node_combined_check, "make_combined_check_node",
                   lambda **kw: lambda s: (count("canon"), count("craft"),
                       (CanonCheckResult(passed=True), CraftCheckResult(passed=True)))[2])
        mp.setattr(node_chapter_summarizer, "make_chapter_summarizer_node",
                   lambda **kw: lambda s: (count("summarize"), {"chapter_summary": ChapterSummary(
                       chapter_number=s.chapter_number, short_summary="short", medium_summary="medium")})[1])
        mp.setattr(node_memory_extractor, "make_memory_extractor_node",
                   lambda **kw: lambda s: (count("extract"), {"memory_patches": []})[1])
        mp.setattr(node_reconciliation, "make_reconciliation_node",
                   lambda **kw: lambda s: (count("reconcile"), {"reconciled_patches": [], "reconciliation_conflicts": []})[1])

        def _persist_factory(**kw):
            def node(s):
                count("persist")
                # mimic the real node's essential effect: register the chapter
                conn = db_module.get_connection(self._db_path)
                conn.execute("INSERT OR IGNORE INTO chapters (story_id, chapter_number) VALUES (?, ?)",
                             (s.story_id, s.chapter_number))
                conn.commit()
                conn.close()
                return {}
            return node
        mp.setattr(node_persistence, "make_persistence_node", _persist_factory)

        mp.setattr(pipeline, "_update_hierarchical_summary",
                   lambda s, **kw: (count("book_summary"), ("stub book summary", []))[1])
        mp.setattr(llm_client, "get_model", lambda: "stub-model")
        mp.setattr(vector_store, "get_chroma_client", lambda path: None)


def _run(tmp_path: Path, db_path: Path, user_input: str = "next chapter"):
    from pipeline import run_chapter
    return run_chapter(
        user_input=user_input,
        story_id="test-story",
        db_path=db_path,
        chroma_path=tmp_path / "chroma",
        manuscripts_dir=tmp_path / "manuscripts",
        book_title="Test Book",
        print_fn=lambda line: None,
    )


def test_crash_then_resume_skips_completed_nodes(monkeypatch, tmp_db_path, tmp_path):
    # Run 1: crash in the chapter summarizer (stage 8 of 12)
    stubs = StubNodes(monkeypatch, tmp_db_path, crash_at="summarize")
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(tmp_path, tmp_db_path)

    ckpt = db_module.get_chapter_checkpoint("test-story", tmp_db_path)
    assert ckpt is not None, "crash should leave a checkpoint behind"
    assert ckpt["chapter_number"] == 1
    assert ckpt["last_stage"] == "craft", "last COMPLETED stage before the crash"

    # Run 2: fresh process (new stub counters), same DB — must resume
    stubs2 = StubNodes(monkeypatch, tmp_db_path)
    state = _run(tmp_path, tmp_db_path)

    for skipped in ("outline", "context", "write", "canon", "craft"):
        assert skipped not in stubs2.calls, f"{skipped} already completed — must not re-run"
    for ran in ("summarize", "book_summary", "extract", "reconcile", "persist"):
        assert stubs2.calls.get(ran) == 1, f"{ran} was not completed — must run on resume"

    # Restored state carried the prose through; chapter completed and cleaned up
    assert state.chapter_prose == "Stub prose."
    assert state.chapter_number == 1
    assert db_module.get_chapter_checkpoint("test-story", tmp_db_path) is None
    assert db_module.get_latest_chapter_number("test-story", tmp_db_path) == 1
    out = tmp_path / "manuscripts" / "Test Book" / "Chapter 1.txt"
    assert out.exists() and "Stub prose." in out.read_text(encoding="utf-8")


def test_crash_after_persist_still_writes_manuscript(monkeypatch, tmp_db_path, tmp_path):
    # Crash in maybe_revise_outline: persistence already registered the chapter,
    # but the manuscript file save happened; checkpoint still present.
    stubs = StubNodes(monkeypatch, tmp_db_path, crash_at="revise_outline")
    with pytest.raises(RuntimeError):
        _run(tmp_path, tmp_db_path)
    ckpt = db_module.get_chapter_checkpoint("test-story", tmp_db_path)
    assert ckpt is not None and ckpt["last_stage"] == "persist"
    # `latest` now equals the checkpointed chapter — resume must finish it,
    # not start chapter 2.
    stubs2 = StubNodes(monkeypatch, tmp_db_path)
    state = _run(tmp_path, tmp_db_path)
    assert state.chapter_number == 1
    assert stubs2.calls.get("persist") is None, "persist already completed"
    assert db_module.get_chapter_checkpoint("test-story", tmp_db_path) is None


def test_clean_run_leaves_no_checkpoint(monkeypatch, tmp_db_path, tmp_path):
    stubs = StubNodes(monkeypatch, tmp_db_path)
    state = _run(tmp_path, tmp_db_path)
    assert state.chapter_number == 1
    assert db_module.get_chapter_checkpoint("test-story", tmp_db_path) is None
    # every stage ran exactly once
    assert all(n == 1 for n in stubs.calls.values()), stubs.calls


def test_full_state_survives_serialization_roundtrip(monkeypatch, tmp_db_path, tmp_path):
    """A realistic mid-pipeline state (context pack, plan, prose, check results)
    must round-trip through the checkpoint JSON without error or data loss."""
    from schema import ChapterGraphState

    state = ChapterGraphState(
        story_id="test-story", chapter_number=2, user_input="go",
        input_mode="continuation",
        context_pack=_stub_context_pack(),
        story_plan=_stub_plan(),
        chapter_prose="Prose here.",
        canon_check_result=CanonCheckResult(passed=True),
        craft_check_result=CraftCheckResult(passed=True),
        chapter_summary=ChapterSummary(chapter_number=2, short_summary="s", medium_summary="m"),
        story_outline=StoryOutline(story_id="test-story"),
    )
    db_module.save_chapter_checkpoint("test-story", 2, "summarize", state.model_dump_json(), tmp_db_path)
    ckpt = db_module.get_chapter_checkpoint("test-story", tmp_db_path)
    restored = ChapterGraphState.model_validate_json(ckpt["state_json"])
    assert restored == state
