"""
Node 5: Story Writer.

Reads the full chapter context and writes prose (length set by the Story
Plan's target_word_count) in close third-person limited from the POV
character's perspective.

Inputs (all on ChapterGraphState):
  context_pack        — world rules, plotlines, characters, POV, locations, history
  story_plan          — scenes, pacing, conflicts, narrative goals, constraints, target length
  character_reasonings — per-character intentions/dialogue/emotion (verbatim)
  chapter_prose       — present only on a canon-check retry (prose to revise)

On first call: writes fresh prose from the plan.
On retry (violation_feedback provided): revises existing prose to fix specific
  canon violations. The original prose + violations are both shown to the model.

If the user has saved a style sample (Prompts tab), it's included as a fixed
voice/rhythm reference in every prompt — deliberately not a rolling excerpt of
the story's own previous chapter, since that would let any drift in the
writer's output compound chapter over chapter.

Output: {"chapter_prose": str}

Model: same local Ollama server as other nodes.
Plain text response — no tool call, no JSON parsing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import db
from llm_client import MODEL, chat_text
from schema import (
    Character, CharacterReasoning, ChapterGraphState, ContextPack, StoryPlan,
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _fmt_world_rules(pack: ContextPack) -> str:
    if not pack.relevant_world_rules:
        return "  (none defined)"
    return "\n".join(
        f"  • [{r.rule_type.upper()}] {r.title}: {r.content}"
        for r in pack.relevant_world_rules
    )


def _fmt_history(pack: ContextPack) -> str:
    s = pack.last_chapter_summary
    if s is None:
        return "  This is the first chapter — no prior history."
    events = ("\n".join(f"    - {e}" for e in s.timeline_events)
              if s.timeline_events else "    (none recorded)")
    return (
        f"  Chapter {s.chapter_number} summary: {s.medium_summary}\n"
        f"  Key events:\n{events}"
    )


def _fmt_plotlines(pack: ContextPack) -> str:
    if not pack.active_plotlines:
        return "  (none active)"
    return "\n".join(
        f"  • {p.name} [tension {p.current_tension}/10]: {p.progress_stage}"
        for p in pack.active_plotlines
    )


def _fmt_pov(pack: ContextPack, pov_name: str) -> str:
    pov = pack.pov_state
    if pov is None:
        return "  (not yet established)"
    loc_name = "unknown"
    for l in pack.nearby_locations:
        if l.id == pov.location_id:
            loc_name = l.name
            break
    companions = ", ".join(pov.companions) if pov.companions else "none"
    injuries = ", ".join(pov.injuries) if pov.injuries else "none"
    inventory = ", ".join(pov.inventory) if pov.inventory else "none"
    knowledge = "; ".join(pov.knowledge) if pov.knowledge else "none"
    return (
        f"  POV character: {pov_name}\n"
        f"  Location: {loc_name}\n"
        f"  Companions present: {companions}\n"
        f"  Emotional state: {pov.emotional_state or 'unknown'}\n"
        f"  Injuries: {injuries}\n"
        f"  Inventory: {inventory}\n"
        f"  What {pov_name} knows: {knowledge}"
    )


def _fmt_locations(pack: ContextPack) -> str:
    if not pack.nearby_locations:
        return "  (none)"
    return "\n".join(
        f"  • {l.name}: {l.description}"
        + (f" [tone: {l.tone}]" if l.tone else "")
        + (f" [recent: {'; '.join(l.recent_events)}]" if l.recent_events else "")
        for l in pack.nearby_locations
    )


def _fmt_characters(pack: ContextPack) -> str:
    if not pack.active_characters:
        return "  (none in scene)"
    lines = []
    for c in pack.active_characters:
        rels = (", ".join(f"{k}: {v}" for k, v in c.relationships.items())
                if c.relationships else "none")
        lines.append(
            f"  • {c.name}: {c.personality}\n"
            f"    Emotional state: {c.emotional_state or 'unknown'}\n"
            f"    Relationships: {rels}"
        )
    return "\n".join(lines)


def _fmt_plan(plan: StoryPlan) -> str:
    scenes = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan.scenes))
    conflicts = "\n".join(f"  • {c}" for c in plan.conflicts) if plan.conflicts else "  (none)"
    goals = "\n".join(f"  • {g}" for g in plan.narrative_goals) if plan.narrative_goals else "  (none)"
    callbacks = "\n".join(f"  • {r}" for r in plan.required_callbacks) if plan.required_callbacks else "  (none)"
    return (
        f"Scenes (in order):\n{scenes}\n\n"
        f"Pacing: {plan.pacing_notes}\n\n"
        f"Conflicts to dramatize:\n{conflicts}\n\n"
        f"Narrative goals (must accomplish):\n{goals}\n\n"
        f"Story-level required beats (must appear in prose):\n{callbacks}"
    )


def _fmt_constraints(plan: StoryPlan, char_map: dict[str, str]) -> str:
    if not plan.character_constraints:
        return "  (none)"
    lines = []
    for cc in plan.character_constraints:
        name = char_map.get(cc.character_id, cc.character_id[:8])
        if cc.forbidden_actions:
            for fa in cc.forbidden_actions:
                lines.append(f"  • {name} MUST NOT: {fa}")
        if cc.required_callbacks:
            for rb in cc.required_callbacks:
                lines.append(f"  • {name} MUST: {rb}")
    return "\n".join(lines) if lines else "  (none)"


def _fmt_reasonings(reasonings: list[CharacterReasoning], char_map: dict[str, str]) -> str:
    if not reasonings:
        return "  (none)"
    lines = []
    for r in reasonings:
        name = char_map.get(r.character_id, r.character_id[:8])
        intentions = "; ".join(r.action_intentions) if r.action_intentions else "none"
        lines.append(
            f"  {name}:\n"
            f"    Plans to: {intentions}\n"
            f"    Dialogue intent: {r.dialogue_intent}\n"
            f"    Emotional state: {r.emotional_response}"
        )
    return "\n".join(lines)


def _word_count_band(target: int) -> tuple[int, int]:
    """Chapter length band around the plan's target_word_count (-20%/+20%)."""
    return int(target * 0.8), int(target * 1.2)


def _fmt_style_sample(style_sample: Optional[str]) -> str:
    if not style_sample:
        return ""
    return (
        "\n=== STYLE REFERENCE (match this voice, rhythm, and tone — do not copy its content) ===\n"
        f"{style_sample}\n"
    )


def build_writer_prompt(
    state: ChapterGraphState,
    violation_feedback: Optional[list[str]] = None,
    style_sample: Optional[str] = None,
    template: Optional[str] = None,
    revision_template: Optional[str] = None,
) -> str:
    from prompt_templates import DEFAULT_TEMPLATES

    pack = state.context_pack
    plan = state.story_plan
    if pack is None:
        raise ValueError("context_pack must be set before story writer runs")
    if plan is None:
        raise ValueError("story_plan must be set before story writer runs")

    # Resolve POV character name
    pov_name = "the protagonist"
    if pack.pov_state and pack.pov_state.pov_character_id:
        for c in pack.active_characters:
            if c.id == pack.pov_state.pov_character_id:
                pov_name = c.name
                break

    char_map = {c.id: c.name for c in pack.active_characters}
    min_words, max_words = _word_count_band(plan.target_word_count)
    style_section = _fmt_style_sample(style_sample)

    if violation_feedback:
        # Retry path: revise existing prose to fix specific violations
        tpl = revision_template if revision_template is not None else DEFAULT_TEMPLATES["story_writer_revision"]
        violations_text = "\n".join(f"  • {v}" for v in violation_feedback)
        return tpl.format(
            style_section=style_section,
            world_rules_block=_fmt_world_rules(pack),
            violations_block=violations_text,
            original_prose=state.chapter_prose,
            constraints_block=_fmt_constraints(plan, char_map),
            min_words=min_words,
            max_words=max_words,
        )

    # First-write path
    tpl = template if template is not None else DEFAULT_TEMPLATES["story_writer_first"]
    return tpl.format(
        pov_name=pov_name,
        min_words=min_words,
        max_words=max_words,
        target_word_count=plan.target_word_count,
        style_section=style_section,
        world_rules_block=_fmt_world_rules(pack),
        history_block=_fmt_history(pack),
        plotlines_block=_fmt_plotlines(pack),
        pov_block=_fmt_pov(pack, pov_name),
        locations_block=_fmt_locations(pack),
        characters_block=_fmt_characters(pack),
        plan_block=_fmt_plan(plan),
        constraints_block=_fmt_constraints(plan, char_map),
        reasonings_block=_fmt_reasonings(state.character_reasonings, char_map),
        user_input=state.user_input,
        chapter_number=state.chapter_number,
    )


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_story_writer_node(
    model: str = MODEL,
    ollama_client=None,
    db_path: Optional[Path] = None,
) -> Callable[[ChapterGraphState], dict]:

    def node(
        state: ChapterGraphState,
        violation_feedback: Optional[list[str]] = None,
    ) -> dict:
        from prompt_templates import get_template
        style_sample = db.get_style_sample(state.story_id, db_path)
        template = get_template("story_writer_first", state.story_id, db_path)
        revision_template = get_template("story_writer_revision", state.story_id, db_path)
        prompt = build_writer_prompt(
            state, violation_feedback, style_sample=style_sample,
            template=template, revision_template=revision_template,
        )

        prose = chat_text(
            prompt, model=model, max_tokens=4096,
            timeout=900,  # 15 min — 70B writing 1200 words takes a while
            client=ollama_client, label="Story writer",
        )

        target = state.story_plan.target_word_count if state.story_plan else 1000
        MIN_WORDS = int(target * 0.7)  # continuation-trigger floor — below this, ask for more
        if len(prose.split()) < MIN_WORDS:
            print(f"  [writer] Output too short ({len(prose.split())} words) — requesting continuation...")
            continuation_prompt = (
                f"Continue the chapter from exactly where it left off. "
                f"Write at least 400 more words. "
                f"Third-person only. No title or commentary. "
                f"Begin directly after this passage:\n\n{prose[-300:]}"
            )
            try:
                continuation = chat_text(
                    continuation_prompt, model=model, max_tokens=2048, timeout=900,
                    client=ollama_client, retries=1, label="Story writer (continuation)",
                )
                if continuation:
                    prose = prose + "\n\n" + continuation
            except Exception as e:
                print(f"  [writer] Continuation failed (using short version): {e}")

        return {"chapter_prose": prose}

    return node


story_writer_node = make_story_writer_node()
