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

BACKUP_RETENTION = 10  # keep the last N timestamped story.db backups


def _elapsed(t0: float) -> str:
    secs = time.time() - t0
    return f"{secs:.1f}s" if secs < 60 else f"{int(secs//60)}m {int(secs%60)}s"


def _section(p, title: str) -> None:
    """Prints a node/stage banner — collapses the repeated 60-char rule blocks."""
    p(f"\n{'='*60}")
    p(title)
    p(f"{'='*60}")


def _backup_database(db_path: Path, chapter_number: int, print_fn=print) -> None:
    """Copies db_path to backups/ before this chapter's writes begin, and prunes
    old backups past BACKUP_RETENTION. Never blocks generation — a failure here
    only warns, since losing a backup is far less costly than losing the DB."""
    if not db_path.exists():
        return
    try:
        import shutil
        from datetime import datetime

        backups_dir = db_path.parent / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = backups_dir / f"{db_path.stem}_ch{chapter_number}_{stamp}.db"
        shutil.copy2(db_path, dest)

        existing = sorted(
            backups_dir.glob(f"{db_path.stem}_ch*_*.db"),
            key=lambda p: p.stat().st_mtime,
        )
        for stale in existing[:-BACKUP_RETENTION]:
            stale.unlink(missing_ok=True)
    except Exception as e:
        print_fn(f"  [warn] Database backup failed (continuing anyway): {e}")


def _run_revision_loop(
    state: ChapterGraphState,
    check_fn: Callable[[ChapterGraphState], object],
    writer_node: Callable[..., dict],
    max_retries: int,
    label: str,
    print_fn=print,
    first_result=None,
):
    """Runs check_fn(state) repeatedly, asking writer_node to revise the prose
    on failure, up to max_retries times. Works for both CanonCheckResult
    (.violations) and CraftCheckResult (.issues) — pipeline.py owns this loop
    for both checks so neither check node has to privately couple itself to
    the writer.

    first_result: a pre-computed check result for attempt 1 (used when the first
    pass was already run concurrently upstream, so it isn't re-run here). Only
    valid when it was computed against the current prose.

    Returns (state_with_revised_prose, result, attempts, flagged: bool).
    """
    p = print_fn
    attempts = 0
    current_state = state

    while True:
        attempts += 1
        p(f"  {label} check attempt {attempts}/{max_retries + 1}...")
        if attempts == 1 and first_result is not None:
            result = first_result
        else:
            result = check_fn(current_state)
        problems = getattr(result, "violations", None)
        if problems is None:
            problems = getattr(result, "issues", [])

        if result.passed:
            p(f"  {label} check PASSED")
            return current_state, result, attempts, False

        p(f"  {label} check FAILED — {len(problems)} issue(s):")
        descriptions = []
        for v in problems:
            kind = getattr(v, "violation_type", None) or getattr(v, "issue_type", "?")
            severity = getattr(v, "severity", "?")
            desc = getattr(v, "description", "")
            p(f"    [{severity}] {kind}: {desc}")
            descriptions.append(f"[{severity}] {kind}: {desc}")

        if attempts > max_retries:
            p(f"  Retry cap reached ({max_retries} retries) — flagging for review and publishing as-is")
            return current_state, result, attempts, True

        p(f"  Requesting revision from writer (attempt {attempts + 1})...")
        writer_result = writer_node(current_state, violation_feedback=descriptions)
        current_state = current_state.model_copy(update={"chapter_prose": writer_result["chapter_prose"]})


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
) -> tuple[str, list[str]]:
    """Updates the rolling summary as before, then — every
    OUTLINE_REVISION_INTERVAL chapters — compresses it into a new permanent
    ActSummary and resets the rolling summary for the next act.
    Returns (new_rolling_summary, new_act_summaries_list)."""
    from node_outline_manager import OUTLINE_REVISION_INTERVAL
    from schema import ActSummary
    import db as db_module

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
) -> ChapterGraphState:
    """
    Run the full 10-node pipeline for one chapter.
    print_fn receives each status line — swap in a GUI writer for streaming output.
    Returns the final ChapterGraphState.
    """
    p = print_fn  # shorthand

    import db as db_module
    import vector_store as vs
    from node_input_router import make_input_router_node
    from node_outline_manager import make_outline_manager_node, maybe_revise_outline
    from node_context_builder import make_context_builder_node, STALE_PLOTLINE_THRESHOLD
    from node_story_planner import make_story_planner_node
    from llm_client import MODEL as _MODEL
    from node_character_reasoner import make_character_reasoner_node
    from node_story_writer import make_story_writer_node
    from node_canon_check import make_canon_check_node
    from node_craft_check import make_craft_check_node
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

    # ── Node 1: Input Router ──────────────────────────────────────
    _section(p, "NODE 1 — INPUT ROUTER")
    t0 = time.time()
    result = make_input_router_node(db_path=db_path)(state)
    state = state.model_copy(update=result)
    p(f"  Mode: {state.input_mode}  |  Chapter: {state.chapter_number}")
    p(f"  Done in {_elapsed(t0)}")

    # Back up story.db before this chapter writes anything (now that we know the chapter number)
    _backup_database(db_path, state.chapter_number, print_fn=p)

    # ── Outline Manager (load, or generate on cold start) ─────────
    _section(p, "OUTLINE MANAGER")
    t0 = time.time()
    outline_node = make_outline_manager_node(db_path=db_path)
    result = outline_node(state)
    state = state.model_copy(update=result)
    outline = state.story_outline
    p(f"  Premise: {outline.premise or '(not yet set)'}")
    p(f"  Beats: {len(outline.beats)}  |  Character arcs: {len(outline.character_arcs)}  |  v{outline.version}")
    p(f"  Done in {_elapsed(t0)}")

    # ── Node 2: Context Builder ───────────────────────────────────
    _section(p, "NODE 2 — CONTEXT BUILDER")
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

    # ── Node 3: Story Planner ─────────────────────────────────────
    _section(p, f"NODE 3 — STORY PLANNER  [{_MODEL}]")
    t0 = time.time()
    result = make_story_planner_node(db_path=db_path)(state)
    state = state.model_copy(update=result)
    plan = state.story_plan
    p(f"  Scenes: {len(plan.scenes)}  |  Conflicts: {len(plan.conflicts)}")
    for i, scene in enumerate(plan.scenes, 1):
        p(f"    {i}. {scene}")
    p(f"  Done in {_elapsed(t0)}")

    # ── Node 4: Character Reasoner ────────────────────────────────
    _section(p, f"NODE 4 — CHARACTER REASONER  [{_MODEL}]")
    t0 = time.time()
    result = make_character_reasoner_node(db_path=db_path)(state)
    state = state.model_copy(update={
        "character_reasonings": state.character_reasonings + result["character_reasonings"]
    })
    char_map = {c.id: c.name for c in pack.active_characters}
    for r in state.character_reasonings:
        p(f"  {char_map.get(r.character_id, '?')}: {r.dialogue_intent}")
    p(f"  Done in {_elapsed(t0)}")

    # ── Node 5: Story Writer ──────────────────────────────────────
    _section(p, f"NODE 5 — STORY WRITER  [{_MODEL}]")
    t0 = time.time()
    writer_node = make_story_writer_node(db_path=db_path)
    result = writer_node(state)
    state = state.model_copy(update=result)
    word_count = len((state.chapter_prose or "").split())
    p(f"  {word_count} words written")
    p(f"  Done in {_elapsed(t0)}")

    # ── Node 6 & Craft: Canon + Craft checks ──────────────────────
    # The two checks read the same prose and are independent, so their first
    # pass runs concurrently (two LLM calls at once). If canon then has to
    # revise the prose, the craft first-pass is stale and is discarded so craft
    # still evaluates the canon-approved text (preserving the original ordering).
    _section(p, f"NODE 6 — CANON CHECK  [{_MODEL}]")
    t0 = time.time()
    canon_check = make_canon_check_node(db_path=db_path)
    craft_check = make_craft_check_node(db_path=db_path)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_canon = ex.submit(canon_check, state)
        fut_craft = ex.submit(craft_check, state)
        canon_first = fut_canon.result()
        craft_first = fut_craft.result()

    state, canon_result, canon_attempts, canon_flagged = _run_revision_loop(
        state, canon_check, writer_node, MAX_CANON_CHECK_RETRIES, "Canon",
        print_fn=p, first_result=canon_first,
    )
    state = state.model_copy(update={
        "canon_check_result": canon_result,
        "canon_check_attempts": canon_attempts,
        "flagged_for_review": canon_flagged,
        "flagged_violations": canon_result.violations if canon_flagged else [],
    })
    if state.flagged_for_review:
        p(f"  *** FLAGGED — {len(state.flagged_violations)} violation(s) remain ***")
        for v in state.flagged_violations:
            p(f"    [{v.severity}] {v.description}")
    else:
        p(f"  Passed after {state.canon_check_attempts} attempt(s)")
    p(f"  Done in {_elapsed(t0)}")

    # ── Craft Check ────────────────────────────────────────────────
    _section(p, f"CRAFT CHECK  [{_MODEL}]")
    t0 = time.time()
    # The concurrent craft first-pass is only valid if canon left prose untouched
    # (i.e. canon passed on its first attempt without a rewrite).
    craft_seed = craft_first if canon_attempts == 1 else None
    state, craft_result, craft_attempts, craft_flagged = _run_revision_loop(
        state, craft_check, writer_node, MAX_CRAFT_CHECK_RETRIES, "Craft",
        print_fn=p, first_result=craft_seed,
    )
    state = state.model_copy(update={
        "craft_check_result": craft_result,
        "craft_check_attempts": craft_attempts,
        "flagged_for_craft_review": craft_flagged,
        "craft_issues": craft_result.issues if craft_flagged else [],
    })
    if state.flagged_for_craft_review:
        p(f"  *** FLAGGED — {len(state.craft_issues)} craft issue(s) remain ***")
        for v in state.craft_issues:
            p(f"    [{v.severity}] {v.description}")
    else:
        p(f"  Passed after {state.craft_check_attempts} attempt(s)")
    p(f"  Done in {_elapsed(t0)}")

    # ── Node 7: Chapter Summarizer ────────────────────────────────
    _section(p, f"NODE 7 — CHAPTER SUMMARIZER  [{_MODEL}]")
    t0 = time.time()
    result = make_chapter_summarizer_node(db_path=db_path)(state)
    state = state.model_copy(update=result)
    p(f"  {state.chapter_summary.short_summary}")
    p(f"  Done in {_elapsed(t0)}")

    # ── Hierarchical summary update (rolling + permanent act summaries) ──
    _section(p, "BOOK SUMMARY UPDATE")
    t0 = time.time()
    new_book_summary, new_act_summaries = _update_hierarchical_summary(
        state, story_id=story_id, db_path=db_path, model=_MODEL, print_fn=p,
    )
    state = state.model_copy(update={"book_summary": new_book_summary, "act_summaries": new_act_summaries})
    db_module.upsert_book_summary(story_id, new_book_summary, db_path)
    p(f"  Done in {_elapsed(t0)}")

    # ── Node 8: Memory Extractor ──────────────────────────────────
    _section(p, f"NODE 8 — MEMORY EXTRACTOR  [{_MODEL}]")
    t0 = time.time()
    extractor_result = make_memory_extractor_node(db_path=db_path)(state)
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

    # ── Node 9: Reconciliation ────────────────────────────────────
    _section(p, "NODE 9 — RECONCILIATION")
    t0 = time.time()
    result = make_reconciliation_node()(state)
    state = state.model_copy(update=result)
    p(f"  {len(state.reconciled_patches)} patch(es), "
      f"{len(state.reconciliation_conflicts)} conflict(s) resolved")
    p(f"  Done in {_elapsed(t0)}")

    # ── Node 10: Persistence ──────────────────────────────────────
    _section(p, "NODE 10 — PERSISTENCE")
    t0 = time.time()
    make_persistence_node(chroma_client=chroma_client, db_path=db_path)(state)
    p(f"  World state saved")
    p(f"  Done in {_elapsed(t0)}")

    # ── Save chapter file ─────────────────────────────────────────
    out_file = _save_chapter(state, manuscripts_dir, book_title)
    p(f"\n  Saved: {out_file}")

    # ── Periodic outline revision (every OUTLINE_REVISION_INTERVAL chapters) ──
    revised_outline = maybe_revise_outline(state, db_path=db_path, print_fn=p)
    if revised_outline is not None:
        state = state.model_copy(update={"story_outline": revised_outline})

    return state
