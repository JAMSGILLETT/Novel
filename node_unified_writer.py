"""
Unified Writer — replaces Nodes 3 (Story Planner), 4 (Character Reasoner),
and 5 (Story Writer) with a single LLM call.

For reasoning models (qwen3, deepseek-r1, qwq) the model naturally spends its
<think> block planning the chapter and reasoning through character motivations
before writing prose — which is exactly what Nodes 3 and 4 were doing manually
across separate calls.

Output: populates both state.story_plan (needed by canon/craft checks) and
state.chapter_prose. The pipeline then continues from Node 6 as normal.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional

from llm_client import chat_text
from llm_json import extract_json_block
from node_story_planner import (
    _fmt_characters, _fmt_world_rules, _fmt_world_lore,
    _fmt_plotlines, _fmt_pov, _fmt_locations, _fmt_history, _fmt_roster,
    _fmt_outline, _parse_plan_json,
)
from schema import ChapterGraphState, StoryPlan


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_unified_prompt(state: ChapterGraphState) -> str:
    pack = state.context_pack
    if pack is None:
        raise ValueError("context_pack must be set before unified writer runs")

    mode_note = {
        "cold_start": (
            "COLD START — this is the very first chapter. Establish the world, "
            "introduce the POV character, and set the initial situation."
        ),
        "continuation": (
            "CONTINUATION — advance the story naturally from where it left off. "
            "Honor open plotlines and character arcs."
        ),
        "user_event_injection": (
            "USER EVENT INJECTION — the user has forced a plot event (see directive). "
            "Incorporate it coherently while still advancing the story."
        ),
    }.get(state.input_mode or "", "")

    target_words = 1000
    if state.story_plan and state.story_plan.target_word_count:
        target_words = state.story_plan.target_word_count

    dep_section = ""
    if pack.dependency_graph_hits:
        dep_section = "\n=== MANDATORY CONTEXT ===\n" + "\n".join(
            f"  • [{h.rule_id}] {h.reason}: {h.content}"
            for h in pack.dependency_graph_hits
        ) + "\n"

    history = _fmt_history(pack, state.book_summary)

    outline_block = _fmt_outline(state)
    outline_section = f"\n=== STORY OUTLINE & BEATS (make progress on these) ===\n{outline_block}\n" if outline_block.strip() != "(outline not yet populated)" else ""

    return f"""You are writing the next chapter of a novel.

=== MODE: {mode_note} ===
{outline_section}
=== WORLD RULES (never violate these) ===
{_fmt_world_rules(pack)}

=== RELEVANT WORLD LORE ===
{_fmt_world_lore(pack)}

=== ACTIVE PLOTLINES ===
{_fmt_plotlines(pack)}

=== CHARACTERS IN SCENE ===
{_fmt_characters(pack)}

=== POV STATE ===
{_fmt_pov(pack)}

=== NEARBY LOCATIONS ===
{_fmt_locations(pack)}

=== STORY HISTORY ===
{history}
{dep_section}
=== FULL CHARACTER ROSTER ===
{_fmt_roster(pack.character_roster)}

=== USER DIRECTIVE ===
"{state.user_input}"

=== YOUR TASK ===
Think through this chapter carefully: what happens, why each character acts the
way they do, how the plotlines advance, and what the emotional arc is.

Then produce your response in exactly this format — the PLAN section first,
then the PROSE section. Do not mix them.

%%PLAN%%
{{
  "scenes": ["scene 1 description", "scene 2 description", "scene 3 description"],
  "pacing_notes": "tone and rhythm guidance",
  "conflicts": ["active tension 1", "active tension 2"],
  "narrative_goals": ["what this chapter must accomplish"],
  "character_constraints": [],
  "required_callbacks": [],
  "requested_offscreen_character_ids": []
}}
%%PROSE%%
[Write the full chapter prose here. Third-person past tense. {target_words}–{int(target_words * 1.3)} words.
No title, no chapter heading, no commentary — just the prose.]"""


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_PLAN_RE  = re.compile(r"%%PLAN%%(.*?)%%PROSE%%", re.DOTALL)
_PROSE_RE = re.compile(r"%%PROSE%%(.*?)$", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_response(raw: str) -> tuple[StoryPlan, str]:
    """Extract (StoryPlan, prose) from the unified response."""
    cleaned = _strip_think(raw)

    plan_match  = _PLAN_RE.search(cleaned)
    prose_match = _PROSE_RE.search(cleaned)

    # ── Plan ──────────────────────────────────────────────────────────────
    plan: Optional[StoryPlan] = None
    if plan_match:
        plan_text = plan_match.group(1).strip()
        json_str = extract_json_block(plan_text) or plan_text
        try:
            plan = _parse_plan_json(json_str)
        except Exception:
            pass

    if plan is None:
        # Fallback: try to find any JSON object in the response
        json_str = extract_json_block(cleaned)
        if json_str:
            try:
                plan = _parse_plan_json(json_str)
            except Exception:
                pass

    if plan is None:
        plan = StoryPlan(
            scenes=["Chapter as written"],
            pacing_notes="",
            conflicts=[],
            narrative_goals=[],
            character_constraints=[],
            required_callbacks=[],
            requested_offscreen_character_ids=[],
        )

    # ── Prose ──────────────────────────────────────────────────────────────
    prose = ""
    if prose_match:
        prose = prose_match.group(1).strip()
    else:
        # Fallback: if no %%PROSE%% marker, treat everything after the plan JSON as prose
        if plan_match:
            prose = cleaned[plan_match.end():].strip()
        else:
            # Last resort: the whole response is the prose
            prose = cleaned

    return plan, prose


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_unified_writer_node(
    model: Optional[str] = None,
    ollama_client=None,
    db_path: Optional[Path] = None,
    print_fn=print,
) -> Callable[[ChapterGraphState], dict]:
    _p = print_fn

    def node(
        state: ChapterGraphState,
        violation_feedback: Optional[list[str]] = None,
    ) -> dict:
        if violation_feedback:
            # Canon/craft check asked for a rewrite — build a revision prompt
            from node_story_writer import build_writer_prompt
            import db
            style_sample = db.get_style_sample(state.story_id, db_path)
            from prompt_templates import get_template
            revision_template = get_template("story_writer_revision", state.story_id, db_path)
            prompt = build_writer_prompt(
                state, violation_feedback,
                style_sample=style_sample,
                revision_template=revision_template,
            )
            _p("  [unified writer] Revision pass...")
            prose = chat_text(
                prompt, model=model, max_tokens=4096, timeout=900,
                client=ollama_client, label="Unified writer (revision)",
            )
            return {"chapter_prose": prose}

        # ── First pass: full think+plan+write ─────────────────────────────
        from prompt_templates import get_template
        import db
        style_sample = db.get_style_sample(state.story_id, db_path) if db_path else None

        prompt = build_unified_prompt(state)
        if style_sample:
            prompt += f"\n\n=== AUTHOR STYLE SAMPLE ===\n{style_sample}"

        _p("  Thinking + planning + writing in one pass...")
        # 12 000 tokens gives reasoning models room to think deeply before prose.
        # The <think> block alone can run 2-4 k tokens on complex chapters;
        # capping at 6 k was cutting off the model mid-reasoning.
        raw = chat_text(
            prompt, model=model, max_tokens=12000, timeout=1800,
            client=ollama_client, label="Unified writer",
        )

        plan, prose = _parse_response(raw)

        # Continuation if prose is too short
        target = 1000
        if state.story_plan and state.story_plan.target_word_count:
            target = state.story_plan.target_word_count
        if len(prose.split()) < int(target * 0.7):
            _p(f"  [unified writer] Output short ({len(prose.split())} words) — requesting continuation...")
            continuation_prompt = (
                f"Continue the chapter from exactly where it left off. "
                f"Write at least 400 more words. "
                f"Third-person only. No title or commentary. "
                f"Begin directly after:\n\n{prose[-300:]}"
            )
            try:
                continuation = chat_text(
                    continuation_prompt, model=model, max_tokens=2048, timeout=900,
                    client=ollama_client, retries=1, label="Unified writer (continuation)",
                )
                if continuation:
                    prose = prose + "\n\n" + _strip_think(continuation)
            except Exception as e:
                _p(f"  [unified writer] Continuation failed: {e}")

        _p(f"  {len(prose.split())} words written")
        _p(f"  Plan: {len(plan.scenes)} scene(s), {len(plan.conflicts)} conflict(s)")

        return {
            "story_plan": plan,
            "chapter_prose": prose,
            "character_reasonings": [],  # not needed — model reasoned internally
        }

    return node
