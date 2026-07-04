"""
Feedback-driven revision of chapters the author has already accepted:

  regenerate_chapter   — rewrite a whole chapter's prose, then refresh its summary
  regenerate_paragraph — rewrite one selected passage in place

Both are lightweight, single-story operations. They reuse the shared LLM client
(and, for the summary, the chapter-summarizer node) but deliberately do NOT re-run
the planning / canon / craft pipeline or re-derive world state — they revise prose,
so the stored world entities are left untouched. World rules and the running
summaries are fed in as context so a revision stays consistent with canon.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import db
from llm_client import chat_text
from pipeline import GenerationCancelled


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------

def _ctx_blocks(story_id: str, db_path, chapter_number: int):
    rules = db.get_all_world_rules(db_path)
    rules_text = "\n".join(f"  • [{r.rule_type}] {r.title}: {r.content}" for r in rules) or "  (none defined)"
    book = db.get_book_summary(story_id, db_path) or "  (this is early in the story)"
    prev = db.get_last_chapter_summary(story_id, chapter_number, db_path)
    prev_text = prev.medium_summary if prev else "  (no prior chapter)"
    return rules_text, book, prev_text


# ---------------------------------------------------------------------------
# Whole-chapter revision
# ---------------------------------------------------------------------------

_CHAPTER_TEMPLATE = """You are revising an existing chapter of a novel based on the author's feedback.

=== WORLD RULES (must never be violated) ===
{rules}

=== STORY SO FAR ===
{book}

=== PREVIOUS CHAPTER ===
{prev}

=== AUTHOR FEEDBACK — what to change ===
{feedback}

=== CURRENT CHAPTER (revise this whole chapter) ===
{body}

Rewrite the ENTIRE chapter, incorporating the feedback. Keep what already works and
change what the feedback asks for. Preserve the established POV, voice, and tense, and
stay consistent with the world rules and prior events. Output ONLY the revised chapter
prose — no title, no headings, no commentary."""


def regenerate_chapter(
    *, story_id: str, db_path, chroma_path, manuscripts_dir, book_title: str,
    chapter_path, feedback: str, print_fn=print,
    should_cancel: Optional[Callable[[], bool]] = None, progress_fn=None,
) -> Path:
    """Rewrite the chapter file's prose per `feedback`, preserving its header
    line (chapter number + FLAGGED tags) and recomputing the word count, then
    refresh the stored chapter summary so downstream context stays aligned."""
    chapter_path = Path(chapter_path)
    text = chapter_path.read_text(encoding="utf-8")
    header, _, body = text.partition("\n\n")
    first_line = header.split("\n", 1)[0]
    try:
        chapter_number = int(chapter_path.stem.split()[-1])
    except (ValueError, IndexError):
        chapter_number = 0

    if should_cancel is not None and should_cancel():
        raise GenerationCancelled()  # nothing written yet
    if progress_fn:
        progress_fn(1, 2, "Rewriting chapter prose")

    rules, book, prev = _ctx_blocks(story_id, db_path, chapter_number)
    prompt = _CHAPTER_TEMPLATE.format(rules=rules, book=book, prev=prev,
                                      feedback=feedback, body=body.strip())
    print_fn("  Rewriting chapter prose with your feedback…")
    new_body = chat_text(prompt, max_tokens=4096, timeout=900, label="Chapter revision", stream=True).strip()
    if not new_body:
        raise RuntimeError("model returned empty prose")
    wc = len(new_body.split())
    chapter_path.write_text(f"{first_line}\n{'='*60}\n({wc:,} words)\n\n{new_body}\n", encoding="utf-8")
    print_fn(f"  Rewrote chapter ({wc:,} words)")

    # Prose is saved; from here on a cancel just skips the (optional) summary refresh.
    if should_cancel is not None and should_cancel():
        print_fn("  Stopped — prose saved; summary left unchanged.")
        return chapter_path

    if progress_fn:
        progress_fn(2, 2, "Updating chapter summary")
    print_fn("  Updating chapter summary…")
    try:
        _resummarize(story_id, db_path, chapter_number, new_body)
        print_fn("  Chapter summary updated")
    except Exception as e:
        print_fn(f"  [warn] Summary update failed (prose still saved): {e}")
    return chapter_path


def _resummarize(story_id: str, db_path, chapter_number: int, prose: str) -> None:
    from schema import ChapterGraphState
    from node_chapter_summarizer import make_chapter_summarizer_node
    state = ChapterGraphState(story_id=story_id, chapter_number=chapter_number,
                              user_input="", chapter_prose=prose)
    result = make_chapter_summarizer_node(db_path=db_path)(state)
    db.upsert_chapter_summary(result["chapter_summary"], story_id, db_path)


# ---------------------------------------------------------------------------
# Single-passage revision
# ---------------------------------------------------------------------------

_PARAGRAPH_TEMPLATE = """You are revising a single passage of a novel chapter based on the author's feedback.

=== SURROUNDING TEXT (for voice and continuity — do NOT rewrite this) ===
…{before}
>>> PASSAGE TO REVISE >>>
{paragraph}
<<< END PASSAGE <<<
{after}…

=== AUTHOR FEEDBACK ===
{feedback}

Rewrite ONLY the marked passage. Keep the same POV, voice, and tense, and make it flow
naturally with the text around it. Output ONLY the rewritten passage — no commentary."""


def regenerate_paragraph(
    *, story_id: str, db_path, chapter_body: str, paragraph: str, feedback: str, print_fn=print,
) -> str:
    """Return a rewritten version of `paragraph`, given the surrounding prose for
    voice/continuity. The caller substitutes it back into the chapter."""
    idx = chapter_body.find(paragraph)
    if idx >= 0:
        before = chapter_body[max(0, idx - 600):idx]
        after = chapter_body[idx + len(paragraph): idx + len(paragraph) + 600]
    else:
        before = after = ""
    prompt = _PARAGRAPH_TEMPLATE.format(before=before, paragraph=paragraph.strip(),
                                        after=after, feedback=feedback)
    print_fn("  Revising the selected passage…")
    new_para = chat_text(prompt, max_tokens=1200, timeout=300, label="Passage revision", stream=True).strip()
    if not new_para:
        raise RuntimeError("model returned empty passage")
    return new_para
