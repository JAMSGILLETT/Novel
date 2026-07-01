"""
Outline Manager.

Maintains the StoryOutline (premise, theme, planned ending, beats, character
arcs) that gives the Story Planner a book-level north star instead of only
re-deriving direction from the rolling summary chapter to chapter.

Three separate responsibilities, run at three different points in the pipeline:

  make_outline_manager_node() — runs early, right after Node 1 (Input Router).
    Cold start: one LLM call generates the initial outline from world rules/
    lore, any pre-seeded characters, and the user's opening directive.
    Continuation: just loads the existing outline (cheap DB read, no LLM call).

  apply_mechanical_outline_updates(...) — runs at the end of Node 10
    (Persistence), inside the same transaction. No LLM call: appends a beat
    when a new plotline is discovered, marks a beat completed when its linked
    plotline resolves, and appends a placeholder character arc for newly
    discovered characters. This is how "chapters add to the outline" cheaply,
    every chapter, without waiting for a full rewrite.

  maybe_revise_outline(...) — called once at the very end of run_chapter.
    Every OUTLINE_REVISION_INTERVAL chapters, one LLM call rewrites the whole
    outline (beat statuses, character current_stage, refined planned_ending)
    using everything that's happened since the last revision. Cheap on other
    chapters (just a chapter-number check, no DB or LLM work).

Model: same local Ollama server as other nodes.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Optional

import db
from llm_client import MODEL, chat
from llm_json import parse_json_response
from node_story_planner import full_history_text
from prompt_templates import get_template
from schema import (
    CharacterArcNote, ChapterGraphState, PlotlinePatch, StoryBeat, StoryOutline,
)

OUTLINE_REVISION_INTERVAL = 15  # chapters between full LLM outline rewrites (and act-summary boundaries)


def _call_llm(model: str, prompt: str, max_tokens: int = 1536, ollama_client=None) -> str:
    response = chat(
        prompt, model=model, max_tokens=max_tokens, timeout=600,
        response_format={"type": "json_object"}, client=ollama_client,
        label="Outline manager",
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Initial generation (cold start)
# ---------------------------------------------------------------------------

_INIT_TEMPLATE = """

=== OUTPUT FORMAT (respond with ONLY this JSON, no other text) ===
{
  "premise": "one or two sentence core premise / logline",
  "theme": "the thematic throughline this story explores",
  "planned_ending": "a rough sense of how this story should ultimately resolve (this will evolve, but give it direction)",
  "beats": [
    {"description": "a major story beat this novel should hit"}
  ],
  "character_arcs": [
    {"character_id": "exact-id-from-the-roster-above", "arc_summary": "where this character might go over the story"}
  ]
}"""


def _build_init_prompt(
    user_input: str, world_rules: list, world_lore: list, characters: list,
    template: Optional[str] = None,
) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["outline_init"]

    rules_text = "\n".join(f"  • [{r.rule_type}] {r.title}: {r.content}" for r in world_rules) or "  (none defined yet)"
    lore_text = "\n".join(f"  • [{l.category}] {l.title}: {l.content}" for l in world_lore) or "  (none yet)"
    chars_text = "\n".join(f"  • [{c.id}] {c.name}: {c.personality}" for c in characters) or "  (none pre-established — this is a blank slate)"

    return tpl.format(
        user_input=user_input, rules_text=rules_text, lore_text=lore_text, chars_text=chars_text,
    ) + _INIT_TEMPLATE


_INIT_DEFAULTS = {"premise": "", "theme": "", "planned_ending": "", "beats": [], "character_arcs": []}


def _parse_init_outline(raw_content: str, story_id: str) -> StoryOutline:
    data = parse_json_response(raw_content, error_label="Outline manager (init)")
    for key, default in _INIT_DEFAULTS.items():
        if key not in data:
            data[key] = default

    beats = [
        StoryBeat(description=b.get("description", "")) if isinstance(b, dict) else StoryBeat(description=str(b))
        for b in (data.get("beats") or [])
        if (isinstance(b, dict) and b.get("description")) or isinstance(b, str)
    ]
    arcs = [
        CharacterArcNote(character_id=a.get("character_id", ""), arc_summary=a.get("arc_summary", ""))
        for a in (data.get("character_arcs") or [])
        if isinstance(a, dict) and a.get("character_id")
    ]

    return StoryOutline(
        story_id=story_id,
        premise=data.get("premise", ""),
        theme=data.get("theme", ""),
        planned_ending=data.get("planned_ending", ""),
        beats=beats,
        character_arcs=arcs,
    )


def make_outline_manager_node(
    model: str = MODEL,
    ollama_client=None,
    db_path: Optional[Path] = None,
) -> Callable[[ChapterGraphState], dict]:
    """Runs right after Node 1. Loads the existing outline, or generates one
    on cold start (the only case that costs an LLM call)."""

    def node(state: ChapterGraphState) -> dict:
        existing = db.get_story_outline(state.story_id, db_path)
        if existing is not None:
            return {"story_outline": existing}

        characters = db.get_all_characters(state.story_id, db_path)
        world_rules = db.get_all_world_rules(db_path)
        world_lore = db.get_all_world_lore(db_path)

        template = get_template("outline_init", state.story_id, db_path)
        prompt = _build_init_prompt(state.user_input, world_rules, world_lore, characters, template=template)
        raw_content = _call_llm(model, prompt, ollama_client=ollama_client)
        outline = _parse_init_outline(raw_content, state.story_id)
        db.upsert_story_outline(outline, db_path)
        return {"story_outline": outline}

    return node


# ---------------------------------------------------------------------------
# Mechanical per-chapter updates (no LLM) — runs inside Node 10's transaction
# ---------------------------------------------------------------------------

def apply_mechanical_outline_updates(
    state: ChapterGraphState,
    story_id: str,
    db_path: Optional[Path],
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    outline = db.get_story_outline(story_id, db_path)
    if outline is None:
        outline = StoryOutline(story_id=story_id)

    changed = False

    for p in state.new_plotlines:
        outline.beats.append(StoryBeat(
            description=f"{p.name}: {p.progress_stage}".strip(": "),
            status="upcoming",
            related_plotline_id=p.id,
        ))
        changed = True

    resolved_plotline_ids = {
        patch.entity_id for patch in state.reconciled_patches
        if isinstance(patch, PlotlinePatch) and patch.status == "resolved" and patch.entity_id
    }
    if resolved_plotline_ids:
        for beat in outline.beats:
            if beat.related_plotline_id in resolved_plotline_ids and beat.status != "completed":
                beat.status = "completed"
                changed = True

    existing_arc_ids = {a.character_id for a in outline.character_arcs}
    for c in state.new_characters:
        if c.id in existing_arc_ids:
            continue
        outline.character_arcs.append(CharacterArcNote(
            character_id=c.id,
            arc_summary="",
            current_stage=f"Introduced in chapter {state.chapter_number}",
        ))
        changed = True

    if changed:
        db.upsert_story_outline(outline, db_path, conn=conn)


# ---------------------------------------------------------------------------
# Periodic full revision (LLM) — called once at the end of run_chapter
# ---------------------------------------------------------------------------

_REVISION_TEMPLATE = """

=== OUTPUT FORMAT (respond with ONLY this JSON, no other text — replace the ENTIRE outline) ===
{
  "premise": "premise (keep stable unless the story has clearly diverged)",
  "theme": "theme (keep stable unless the story has clearly diverged)",
  "planned_ending": "refine if the story's trajectory suggests a clearer resolution",
  "beats": [
    {"description": "...", "status": "upcoming | in_progress | completed"}
  ],
  "character_arcs": [
    {"character_id": "exact-id-from-the-roster-above", "arc_summary": "...", "current_stage": "where they are right now"}
  ]
}"""


def _build_revision_prompt(
    state: ChapterGraphState, outline: StoryOutline, characters: list,
    template: Optional[str] = None,
) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["outline_revision"]

    char_map = {c.id: c.name for c in characters}
    beats_text = "\n".join(f"  • [{b.status}] {b.description}" for b in outline.beats) or "  (none yet)"
    arcs_text = "\n".join(
        f"  • [{a.character_id}] {char_map.get(a.character_id, a.character_id[:8])}: "
        f"{a.arc_summary} (current stage: {a.current_stage})"
        for a in outline.character_arcs
    ) or "  (none yet)"
    roster_text = "\n".join(f"  • [{c.id}] {c.name}" for c in characters) or "  (none)"
    history = full_history_text(state.act_summaries, state.book_summary) or "  (no history recorded yet)"

    return tpl.format(
        premise=outline.premise,
        theme=outline.theme,
        planned_ending=outline.planned_ending,
        beats_text=beats_text,
        arcs_text=arcs_text,
        chapter_number=state.chapter_number,
        history=history,
        roster_text=roster_text,
    ) + _REVISION_TEMPLATE


_REVISION_DEFAULTS = {"premise": "", "theme": "", "planned_ending": "", "beats": [], "character_arcs": []}


def _parse_revision(raw_content: str, story_id: str, chapter_number: int, prior_version: int) -> StoryOutline:
    data = parse_json_response(raw_content, error_label="Outline manager (revision)")
    for key, default in _REVISION_DEFAULTS.items():
        if key not in data:
            data[key] = default

    beats = []
    for b in (data.get("beats") or []):
        if isinstance(b, dict) and b.get("description"):
            status = b.get("status", "upcoming")
            if status not in ("upcoming", "in_progress", "completed"):
                status = "upcoming"
            beats.append(StoryBeat(description=b["description"], status=status))

    arcs = [
        CharacterArcNote(
            character_id=a.get("character_id", ""),
            arc_summary=a.get("arc_summary", ""),
            current_stage=a.get("current_stage", ""),
        )
        for a in (data.get("character_arcs") or [])
        if isinstance(a, dict) and a.get("character_id")
    ]

    return StoryOutline(
        story_id=story_id,
        premise=data.get("premise", ""),
        theme=data.get("theme", ""),
        planned_ending=data.get("planned_ending", ""),
        beats=beats,
        character_arcs=arcs,
        last_revised_chapter=chapter_number,
        version=prior_version + 1,
    )


def maybe_revise_outline(
    state: ChapterGraphState,
    db_path: Optional[Path] = None,
    model: str = MODEL,
    ollama_client=None,
    print_fn=print,
) -> Optional[StoryOutline]:
    """Call once at the end of run_chapter. Only does anything (and only
    costs an LLM call) every OUTLINE_REVISION_INTERVAL chapters."""
    if state.chapter_number <= 0 or state.chapter_number % OUTLINE_REVISION_INTERVAL != 0:
        return None

    outline = db.get_story_outline(state.story_id, db_path)
    if outline is None:
        return None

    print_fn(f"  Outline due for revision (every {OUTLINE_REVISION_INTERVAL} chapters) — rewriting...")
    characters = db.get_all_characters(state.story_id, db_path)

    template = get_template("outline_revision", state.story_id, db_path)
    prompt = _build_revision_prompt(state, outline, characters, template=template)
    raw_content = _call_llm(model, prompt, ollama_client=ollama_client)
    revised = _parse_revision(raw_content, state.story_id, state.chapter_number, outline.version)
    db.upsert_story_outline(revised, db_path)
    print_fn(f"  Outline revised to v{revised.version} ({len(revised.beats)} beats, {len(revised.character_arcs)} arcs)")
    return revised
