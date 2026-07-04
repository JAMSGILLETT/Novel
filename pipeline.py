"""
Pipeline runner — story-agnostic replacement for run_demo.py.

Usage:
    from pipeline import run_chapter

    run_chapter(
        user_input="Kael confronts Mira",
        story_id="my-story",
        db_path=Path("my_story.db"),
        chroma_path=Path("my_story_chroma"),
        manuscripts_dir=Path("manuscripts"),
        print_fn=print,   # swap for GUI queue writer
    )
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from schema import ChapterGraphState, MAX_CANON_CHECK_RETRIES, MAX_CRAFT_CHECK_RETRIES

# Ordered stage keys for run_chapter — one per stage() banner. Each completed
# stage checkpoints the full pipeline state to the DB (chapter_checkpoints), so
# a crash mid-chapter resumes after the last finished node instead of starting
# the whole chapter over.
_STAGES = [
    "outline", "context", "write", "self_refine",
    "canon", "craft", "summarize", "book_summary",
    "extract", "reconcile", "persist",
]
_TOTAL_STAGES = len(_STAGES)  # for progress %


class GenerationCancelled(Exception):
    """Raised inside run_chapter when should_cancel() returns True between stages."""


def _elapsed(t0: float) -> str:
    secs = time.time() - t0
    return f"{secs:.1f}s" if secs < 60 else f"{int(secs//60)}m {int(secs%60)}s"


def _section(p, title: str) -> None:
    """Prints a node/stage banner — collapses the repeated 60-char rule blocks."""
    p(f"\n{'='*60}")
    p(title)
    p(f"{'='*60}")


def _backup_database(db_path: Path, chapter_number: int, print_fn=print, chroma_path=None) -> None:
    """Snapshot the DB (and vector store) to backups/ before this chapter's
    writes begin. Never blocks generation — a failure here only warns, since
    losing a backup is far less costly than losing the DB. (db.create_backup
    checkpoints, snapshots Chroma, and prunes.)"""
    try:
        import db as db_module
        db_module.create_backup(db_path, tag=f"ch{chapter_number}", chroma_path=chroma_path)
    except Exception as e:
        print_fn(f"  [warn] Database backup failed (continuing anyway): {e}")


def _save_chapter(state: ChapterGraphState, manuscripts_dir: Path, book_title: str) -> Path:
    """Save chapter to manuscripts/{book_title}/Chapter {n}.txt. Returns the file path."""
    book_dir = manuscripts_dir / book_title
    book_dir.mkdir(parents=True, exist_ok=True)
    prose = state.chapter_prose or ""
    word_count = len(prose.split())
    flags = []
    if state.flagged_for_review:
        flags.append("CANON")
    if state.flagged_for_craft_review:
        flags.append("CRAFT")
    flagged = f" [FLAGGED FOR REVIEW: {', '.join(flags)}]" if flags else ""
    header = (
        f"CHAPTER {state.chapter_number}{flagged}\n"
        f"{'='*60}\n"
        f"({word_count} words)\n\n"
    )
    out_file = book_dir / f"Chapter {state.chapter_number}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(header + prose + "\n")
    return out_file


def _update_book_summary(
    current: Optional[str],
    chapter_summary,
    model: str,
    story_id: str,
    db_path: Optional[Path] = None,
    print_fn=print,
) -> str:
    """
    Efficiently update the running novel summary by feeding the LLM only the
    existing summary + new chapter summary — never re-reads all prose.
    """
    from llm_client import chat_text
    from prompt_templates import get_template

    events_text = "\n".join(f"  • {e}" for e in chapter_summary.timeline_events)

    if current:
        tpl = get_template("book_summary_continuation", story_id, db_path)
        prompt = tpl.format(
            current_summary=current,
            chapter_number=chapter_summary.chapter_number,
            medium_summary=chapter_summary.medium_summary,
            timeline_events=events_text,
        )
    else:
        tpl = get_template("book_summary_first", story_id, db_path)
        prompt = tpl.format(
            chapter_number=chapter_summary.chapter_number,
            medium_summary=chapter_summary.medium_summary,
            timeline_events=events_text,
        )

    try:
        result = chat_text(prompt, model=model, max_tokens=700, timeout=300, label="Book summary")
        if result:
            print_fn(f"  Book summary updated ({len(result.split())} words)")
            return result
    except Exception as e:
        print_fn(f"  [warn] Book summary update failed: {e}")

    # Fallback: just append the new chapter summary
    base = current or ""
    new_line = f"\nChapter {chapter_summary.chapter_number}: {chapter_summary.medium_summary}"
    return (base + new_line).strip()


_ACT_SUMMARY_OUTPUT_FORMAT = """

=== OUTPUT FORMAT (respond with ONLY this JSON) ===
{
  "summary": "150 to 200 word compact summary here.",
  "key_events": ["concrete permanent fact 1", "concrete permanent fact 2"]
}"""


def _compress_to_act_summary(
    rolling_summary: str, model: str, story_id: str, db_path: Optional[Path] = None, print_fn=print,
) -> tuple[str, list[str]]:
    """One LLM call: compress a rolling summary into a permanent, compact act
    summary. This is never re-compressed again, so early acts don't get
    crushed the way a single ever-growing rolling summary eventually would."""
    from llm_client import chat_json
    from prompt_templates import get_template

    tpl = get_template("act_summary_compression", story_id, db_path)
    prompt = tpl.format(rolling_summary=rolling_summary) + _ACT_SUMMARY_OUTPUT_FORMAT

    try:
        data = chat_json(prompt, model=model, max_tokens=700, timeout=300,
                         label="Act summary compression")
        summary = (data.get("summary") or "").strip()
        key_events = data.get("key_events") or []
        if summary:
            print_fn(f"  Act summary compressed ({len(summary.split())} words, {len(key_events)} key events)")
            return summary, key_events
    except Exception as e:
        print_fn(f"  [warn] Act summary compression failed: {e}")

    # Fallback: use the rolling summary as-is, no key events extracted
    return rolling_summary.strip(), []


def _update_hierarchical_summary(
    state: ChapterGraphState,
    story_id: str,
    db_path: Path,
    model: str,
    print_fn=print,
    skip_rolling_update: bool = False,
) -> tuple[str, list[str]]:
    """Updates the rolling summary as before, then — every
    OUTLINE_REVISION_INTERVAL chapters — compresses it into a new permanent
    ActSummary and resets the rolling summary for the next act.
    Returns (new_rolling_summary, new_act_summaries_list).

    skip_rolling_update: if True, state.book_summary was already updated by
    the chapter summarizer node in one combined LLM call — skip the separate
    _update_book_summary call."""
    from node_outline_manager import OUTLINE_REVISION_INTERVAL
    from schema import ActSummary
    import db as db_module

    if skip_rolling_update:
        print_fn("  Rolling summary already updated by summarizer — skipping separate call")
        new_rolling = state.book_summary or ""
    else:
        new_rolling = _update_book_summary(
            current=state.book_summary,
            chapter_summary=state.chapter_summary,
            model=model,
            story_id=story_id,
            db_path=db_path,
            print_fn=print_fn,
        )
    act_summaries = list(state.act_summaries)

    if state.chapter_number > 0 and state.chapter_number % OUTLINE_REVISION_INTERVAL == 0:
        act_number = state.chapter_number // OUTLINE_REVISION_INTERVAL
        chapter_start = (act_number - 1) * OUTLINE_REVISION_INTERVAL + 1
        print_fn(f"  Act boundary reached (chapter {state.chapter_number}) — compressing act {act_number}...")
        compact_summary, key_events = _compress_to_act_summary(new_rolling, model, story_id, db_path, print_fn)
        act = ActSummary(
            act_number=act_number,
            chapter_start=chapter_start,
            chapter_end=state.chapter_number,
            summary=compact_summary,
            key_events=key_events,
        )
        db_module.insert_act_summary(story_id, act, db_path)
        act_summaries.append(compact_summary)
        new_rolling = ""  # rolling summary resets; the act summary now carries this period permanently

    return new_rolling, act_summaries


def run_chapter(
    user_input: str,
    story_id: str,
    db_path: Path,
    chroma_path: Path,
    manuscripts_dir: Path,
    book_title: str = "My Novel",
    print_fn: Callable[[str], None] = print,
    should_cancel: Optional[Callable[[], bool]] = None,
    progress_fn: Optional[Callable[[int, int, str], None]] = None,
    debug: bool = False,
) -> ChapterGraphState:
    """
    Run the full pipeline for one chapter.
    print_fn receives each status line — swap in a GUI writer for streaming output.
    should_cancel, if given, is polled between stages; when it returns True the
    run aborts by raising GenerationCancelled (no partial chapter is saved).
    progress_fn(step, total, label), if given, is called as each stage begins.
    debug=True routes all internal node prints through print_fn (visible in GUI).
    Returns the final ChapterGraphState.
    """
    p = print_fn  # shorthand
    node_p = p if debug else print  # internal node output: GUI in debug mode, stdout otherwise

    # One helper per stage: poll for cancellation, report progress, print banner.
    _stage_no = [0]

    def stage(title: str) -> None:
        if should_cancel is not None and should_cancel():
            # Deliberate abort — drop the crash checkpoint so the next run
            # starts a fresh chapter (Stop means "discard", crash means "resume").
            try:
                db_module.delete_chapter_checkpoint(story_id, db_path)
            except Exception:
                pass
            raise GenerationCancelled()
        _stage_no[0] += 1
        if progress_fn is not None:
            progress_fn(_stage_no[0], _TOTAL_STAGES, title)
        _section(p, title)

    import db as db_module
    import vector_store as vs
    from node_outline_manager import make_outline_manager_node, maybe_revise_outline
    from node_context_builder import make_context_builder_node, STALE_PLOTLINE_THRESHOLD
    from llm_client import get_model
    _MODEL = get_model()  # snapshot the active model for this run's banners/summaries
    from node_chapter_summarizer import make_chapter_summarizer_node
    from node_memory_extractor import make_memory_extractor_node
    from node_reconciliation import make_reconciliation_node
    from node_persistence import make_persistence_node

    db_module.init_db(db_path)
    db_module.DB_PATH = db_path

    # Load running book summary so Node 3 and Node 4 have full novel context
    book_summary = db_module.get_book_summary(story_id, db_path)
    act_summaries = [a.summary for a in db_module.get_all_act_summaries(story_id, db_path)]

    state = ChapterGraphState(
        story_id=story_id,
        chapter_number=0,
        user_input=user_input,
        book_summary=book_summary,
        act_summaries=act_summaries,
    )

    chroma_client = vs.get_chroma_client(chroma_path)

    # ── Chapter number & mode (pure DB facts, no LLM) ─────────────
    latest = db_module.get_latest_chapter_number(story_id, db_path)
    state = state.model_copy(update={
        "input_mode": "cold_start" if latest is None else "continuation",
        "chapter_number": (latest or 0) + 1,
    })

    # ── Crash recovery: resume an interrupted chapter from its checkpoint ──
    # A checkpoint row only survives a crash or unexpected error — chapters that
    # finish (or are cancelled with Stop) delete theirs. If one exists for the
    # chapter we're about to generate, restore the saved state and skip every
    # node that already completed.
    done_stages: set = set()
    ckpt = db_module.get_chapter_checkpoint(story_id, db_path)
    if ckpt is not None:
        # Also resumable: crash after Node 10 registered the chapter (so
        # `latest` already includes it) but before the manuscript file was
        # written — only the post-persistence steps remain.
        resumable = ckpt["chapter_number"] == state.chapter_number or (
            ckpt["chapter_number"] == latest and ckpt["last_stage"] == _STAGES[-1]
        )
        if resumable:
            try:
                state = ChapterGraphState.model_validate_json(ckpt["state_json"])
                done_stages = set(_STAGES[: _STAGES.index(ckpt["last_stage"]) + 1])
            except Exception as e:
                p(f"  [warn] Couldn't load the crash checkpoint — starting the chapter fresh: {e}")
                db_module.delete_chapter_checkpoint(story_id, db_path)
        else:
            # Belongs to an older, already-completed chapter — stale, discard.
            db_module.delete_chapter_checkpoint(story_id, db_path)

    p(f"  Mode: {state.input_mode}  |  Chapter: {state.chapter_number}")
    if done_stages:
        _stage_no[0] = len(done_stages)  # progress % accounts for skipped nodes
        p(f"  Resuming chapter {state.chapter_number} from checkpoint — "
          f"{len(done_stages)}/{len(_STAGES)} node(s) already completed, "
          f"continuing after '{ckpt['last_stage']}'")
        if user_input.strip() and user_input.strip() != (state.user_input or "").strip():
            p("  [note] Finishing the interrupted chapter with its original input; "
              "your new input will apply to the next chapter.")

    def _run(key: str) -> bool:
        return key not in done_stages

    def _ckpt(key: str, snapshot: ChapterGraphState) -> None:
        """Persist the full pipeline state after a completed node. Best-effort:
        a failure only costs crash recovery for this node, never the chapter."""
        try:
            db_module.save_chapter_checkpoint(
                story_id, snapshot.chapter_number, key, snapshot.model_dump_json(), db_path,
            )
        except Exception as e:
            p(f"  [warn] Checkpoint save failed (crash recovery unavailable for this node): {e}")

    # Back up story.db before this chapter writes anything (now that we know the
    # chapter number). Skipped on resume — the pre-chapter snapshot already
    # exists, and a new one would capture mid-chapter state under the same tag.
    if not done_stages:
        _backup_database(db_path, state.chapter_number, print_fn=p, chroma_path=chroma_path)

    # ── Outline Manager (load, or generate on cold start) ─────────
    if _run("outline"):
        stage("OUTLINE MANAGER")
        t0 = time.time()
        outline_node = make_outline_manager_node(db_path=db_path)
        result = outline_node(state)
        state = state.model_copy(update=result)
        outline = state.story_outline
        p(f"  Premise: {outline.premise or '(not yet set)'}")
        p(f"  Beats: {len(outline.beats)}  |  Character arcs: {len(outline.character_arcs)}  |  v{outline.version}")
        p(f"  Done in {_elapsed(t0)}")
        _ckpt("outline", state)

    # ── Node 2: Context Builder ───────────────────────────────────
    if _run("context"):
        stage("NODE 2 — CONTEXT BUILDER")
        t0 = time.time()
        context_node = make_context_builder_node(chroma_client=chroma_client, db_path=db_path)
        result = context_node(state)
        state = state.model_copy(update=result)
        pack = state.context_pack
        p(f"  Characters: {[c.name for c in pack.active_characters]}")
        p(f"  Plotlines:  {[pl.name for pl in pack.active_plotlines]}")
        p(f"  Locations:  {[l.name for l in pack.nearby_locations]}")
        if pack.stale_plotlines:
            p(f"  [nudge] Stale plotlines (quiet {STALE_PLOTLINE_THRESHOLD}+ chapters): "
              f"{[pl.name for pl in pack.stale_plotlines]}")
        p(f"  Done in {_elapsed(t0)}")
        _ckpt("context", state)

    # ── Nodes 3+4+5: Unified Writer ──────────────────────────────
    # One LLM call: the model thinks through the plan and character motivations
    # internally (via its reasoning/think block), then produces both a structured
    # plan (needed by canon/craft checks) and the chapter prose.
    from node_unified_writer import make_unified_writer_node
    writer_node = make_unified_writer_node(db_path=db_path, print_fn=node_p)
    if _run("write"):
        stage(f"NODES 3-5 — UNIFIED WRITER  [{_MODEL}]")
        t0 = time.time()
        result = writer_node(state)
        state = state.model_copy(update=result)
        plan = state.story_plan
        if plan:
            p(f"  Plan: {len(plan.scenes)} scene(s) | {len(plan.conflicts)} conflict(s)")
            for i, scene in enumerate(plan.scenes, 1):
                p(f"    {i}. {scene}")
        word_count = len((state.chapter_prose or "").split())
        p(f"  {word_count} words written")
        p(f"  Done in {_elapsed(t0)}")
        _ckpt("write", state)

    # ── Craft Self-Refine (cheap, craft-only, before the grounded gate) ──
    # The model improves its own draft's craft before the expensive grounded
    # Evaluator + best-of-N reviser run, so fewer costly revision rounds fire.
    # Canon is untouched here — it stays the external grounded gate below.
    from node_self_refine import make_self_refine_node
    if _run("self_refine"):
        stage(f"SELF-REFINE (craft)  [{_MODEL}]")
        t0 = time.time()
        result = make_self_refine_node(db_path=db_path, print_fn=node_p)(state)
        state = state.model_copy(update=result)
        p(f"  {len((state.chapter_prose or '').split())} words after craft self-refine")
        p(f"  Done in {_elapsed(t0)}")
        _ckpt("self_refine", state)

    # ── Node 6: Combined Canon + Craft Check ──────────────────────
    # Single LLM call evaluates both continuity and writing quality together.
    # If the prose fails either check, the writer revises and the combined
    # check re-runs on the new prose. Both "canon" and "craft" stage keys map
    # to this one block so resume-from-checkpoint still works.
    from node_combined_check import make_combined_check_node
    from node_reviser import make_reviser_loop

    # Evaluator: grounded combined check. Actor: best-of-N local-edit reviser
    # with memory across attempts (node_reviser). The reviser re-scores each
    # candidate with this same evaluator, so canon stays grounded throughout.
    combined_check = make_combined_check_node(db_path=db_path)
    reviser_loop = make_reviser_loop(combined_check, print_fn=p)

    if _run("canon") or _run("craft"):
        stage(f"NODE 6 — CANON + CRAFT CHECK  [{_MODEL}]")
        t0 = time.time()
        max_retries = max(MAX_CANON_CHECK_RETRIES, MAX_CRAFT_CHECK_RETRIES)
        state, combined_result, attempts, flagged = reviser_loop(state, max_retries)
        canon_result  = combined_result.canon
        craft_result  = combined_result.craft
        canon_flagged = not canon_result.passed and flagged
        craft_flagged = not craft_result.passed and flagged
        state = state.model_copy(update={
            "canon_check_result":      canon_result,
            "canon_check_attempts":    attempts,
            "flagged_for_review":      canon_flagged,
            "flagged_violations":      canon_result.violations if canon_flagged else [],
            "craft_check_result":      craft_result,
            "craft_check_attempts":    attempts,
            "flagged_for_craft_review": craft_flagged,
            "craft_issues":            craft_result.issues if craft_flagged else [],
        })
        if canon_flagged:
            p(f"  *** CANON FLAGGED — {len(state.flagged_violations)} violation(s) remain ***")
            for v in state.flagged_violations:
                p(f"    [{v.severity}] {v.description}")
        if craft_flagged:
            p(f"  *** CRAFT FLAGGED — {len(state.craft_issues)} issue(s) remain ***")
            for v in state.craft_issues:
                p(f"    [{v.severity}] {v.description}")
        if not flagged:
            p(f"  Passed after {attempts} attempt(s)")

        # Record the final verdict so the few-shot demo bank can grow from real
        # history (check_demos mines this). Best-effort — never blocks a chapter.
        try:
            from schema import CombinedCheckResult
            from node_combined_check import _fmt_world_rules
            verdict = CombinedCheckResult(
                canon_passed=canon_result.passed, violations=canon_result.violations,
                craft_passed=craft_result.passed, issues=craft_result.issues,
            )
            db_module.save_check_verdict(
                story_id, state.chapter_number, verdict.canon_passed and verdict.craft_passed,
                _fmt_world_rules(state.context_pack), state.chapter_prose or "",
                verdict.model_dump_json(), db_path,
            )
        except Exception as e:
            p(f"  [warn] Couldn't record check verdict for demo history: {e}")

        p(f"  Done in {_elapsed(t0)}")
        _ckpt("canon", state)
        _ckpt("craft", state)

    # ── Node 7: Chapter Summarizer ────────────────────────────────
    _summarizer_updated_book = False
    if _run("summarize"):
        stage(f"NODE 7 — CHAPTER SUMMARIZER  [{_MODEL}]")
        t0 = time.time()
        result = make_chapter_summarizer_node(db_path=db_path, print_fn=node_p)(state)
        _summarizer_updated_book = "book_summary" in result
        state = state.model_copy(update=result)
        p(f"  {state.chapter_summary.short_summary}")
        p(f"  Done in {_elapsed(t0)}")
        _ckpt("summarize", state)

    # ── Hierarchical summary update (rolling + permanent act summaries) ──
    if _run("book_summary"):
        stage("BOOK SUMMARY UPDATE")
        t0 = time.time()
        new_book_summary, new_act_summaries = _update_hierarchical_summary(
            state, story_id=story_id, db_path=db_path, model=_MODEL, print_fn=p,
            skip_rolling_update=_summarizer_updated_book,
        )
        state = state.model_copy(update={"book_summary": new_book_summary, "act_summaries": new_act_summaries})
        db_module.upsert_book_summary(story_id, new_book_summary, db_path)
        p(f"  Done in {_elapsed(t0)}")
        _ckpt("book_summary", state)

    # ── Node 8: Memory Extractor ──────────────────────────────────
    if _run("extract"):
        stage(f"NODE 8 — MEMORY EXTRACTOR  [{_MODEL}]")
        t0 = time.time()
        extractor_result = make_memory_extractor_node(db_path=db_path, print_fn=node_p)(state)
        state = state.model_copy(update={
            "memory_patches":  state.memory_patches + extractor_result["memory_patches"],
            "new_characters":  extractor_result.get("new_characters", []),
            "new_locations":   extractor_result.get("new_locations", []),
            "new_plotlines":   extractor_result.get("new_plotlines", []),
            "new_world_rules": extractor_result.get("new_world_rules", []),
            "new_world_lore":  extractor_result.get("new_world_lore", []),
        })
        p(f"  {len(state.memory_patches)} patch(es), "
          f"{sum(len(extractor_result.get(k,[])) for k in ('new_characters','new_locations','new_plotlines','new_world_rules','new_world_lore'))} new entity/entities")
        p(f"  Done in {_elapsed(t0)}")
        _ckpt("extract", state)

    # ── Node 9: Reconciliation ────────────────────────────────────
    if _run("reconcile"):
        stage("NODE 9 — RECONCILIATION")
        t0 = time.time()
        result = make_reconciliation_node(print_fn=node_p)(state)
        state = state.model_copy(update=result)
        p(f"  {len(state.reconciled_patches)} patch(es), "
          f"{len(state.reconciliation_conflicts)} conflict(s) resolved")
        p(f"  Done in {_elapsed(t0)}")
        _ckpt("reconcile", state)

    # ── Node 10: Persistence ──────────────────────────────────────
    if _run("persist"):
        stage("NODE 10 — PERSISTENCE")
        t0 = time.time()
        make_persistence_node(chroma_client=chroma_client, db_path=db_path, print_fn=node_p)(state)
        p(f"  World state saved")
        p(f"  Done in {_elapsed(t0)}")
        _ckpt("persist", state)

    # ── Save chapter file ─────────────────────────────────────────
    out_file = _save_chapter(state, manuscripts_dir, book_title)
    p(f"\n  Saved: {out_file}")

    # ── Periodic outline revision (every OUTLINE_REVISION_INTERVAL chapters) ──
    revised_outline = maybe_revise_outline(state, db_path=db_path, print_fn=p)
    if revised_outline is not None:
        state = state.model_copy(update={"story_outline": revised_outline})

    # Chapter fully complete — the crash checkpoint is no longer needed.
    db_module.delete_chapter_checkpoint(story_id, db_path)

    return state
