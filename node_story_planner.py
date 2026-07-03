"""
Node 3: Story Planner ("Director AI").

Reads the ContextPack from Node 2 and produces a StoryPlan that constrains
everything downstream:

  scenes                          — ordered scene descriptions for this chapter
  pacing_notes                    — tone and rhythm guidance for the writer
  conflicts                       — active tensions to dramatize
  narrative_goals                 — what this chapter must accomplish story-wise
  character_constraints           — per-character forbidden_actions + required_callbacks
  required_callbacks              — story-level beats the writer must hit
  requested_offscreen_character_ids — roster IDs the planner wants pulled in from
                                    off-screen; Node 4 fetches their full profiles

Model: local Ollama server (OpenAI-compatible API on localhost:11434).
Default model: llama3.1:8b — override with NOVELGEN_MODEL env var.
Structured output via OpenAI-compatible tool calling — tool_choice is forced to
"story_plan" so the model cannot return freeform text.

Requires:
  pip install openai
  Ollama installed and running (https://ollama.com)
  Model pulled: ollama pull llama3.1:8b   (or whatever NOVELGEN_MODEL is set to)
"""
from __future__ import annotations

from typing import Callable, Optional

from llm_client import chat_json
from schema import (
    CharacterRosterEntry, ContextPack, ChapterGraphState, StoryPlan,
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _fmt_world_rules(pack: ContextPack) -> str:
    if not pack.relevant_world_rules:
        return "  (none defined yet)"
    return "\n".join(
        f"  • [{r.rule_type}] {r.title}: {r.content}"
        for r in pack.relevant_world_rules
    )


def _fmt_world_lore(pack: ContextPack) -> str:
    if not pack.relevant_world_lore:
        return "  (none retrieved)"
    return "\n".join(
        f"  • [{l.category}] {l.title}: {l.content}"
        for l in pack.relevant_world_lore
    )


def _fmt_plotlines(pack: ContextPack) -> str:
    if not pack.active_plotlines:
        return "  (none active)"
    lines = []
    for p in pack.active_plotlines:
        devs = ("; ".join(p.next_possible_developments)
                if p.next_possible_developments else "unspecified")
        lines.append(
            f"  • [{p.id}] {p.name} | stage: {p.progress_stage} | tension: {p.current_tension}/10\n"
            f"    Next possible: {devs}"
        )
    return "\n".join(lines)


def _fmt_characters(pack: ContextPack) -> str:
    if not pack.active_characters:
        return "  (none in scene)"
    lines = []
    for c in pack.active_characters:
        goals = "; ".join(c.goals) if c.goals else "none stated"
        objs = "; ".join(c.current_objectives) if c.current_objectives else "none"
        rels = (", ".join(f"{k}: {v}" for k, v in c.relationships.items())
                if c.relationships else "none")
        lines.append(
            f"  ### [{c.id}] {c.name}\n"
            f"    Personality: {c.personality}\n"
            f"    Goals: {goals}\n"
            f"    Current objectives: {objs}\n"
            f"    Emotional state: {c.emotional_state or 'unknown'}\n"
            f"    Relationships: {rels}"
        )
    return "\n".join(lines)


def _fmt_pov(pack: ContextPack) -> str:
    pov = pack.pov_state
    if pov is None:
        return "  (not yet established — cold start)"
    loc_name = "unknown"
    for l in pack.nearby_locations:
        if l.id == pov.location_id:
            loc_name = l.name
            break
    companions = ", ".join(pov.companions) if pov.companions else "none"
    injuries = ", ".join(pov.injuries) if pov.injuries else "none"
    inventory = ", ".join(pov.inventory) if pov.inventory else "none"
    return (
        f"  Location: {loc_name} (id: {pov.location_id})\n"
        f"  Companions: {companions}\n"
        f"  Emotional state: {pov.emotional_state or 'unknown'}\n"
        f"  Injuries: {injuries}\n"
        f"  Inventory: {inventory}"
    )


def _fmt_locations(pack: ContextPack) -> str:
    if not pack.nearby_locations:
        return "  (none)"
    return "\n".join(
        f"  • [{l.id}] {l.name}: {l.description}"
        + (f" [tone: {l.tone}]" if l.tone else "")
        for l in pack.nearby_locations
    )


def full_history_text(act_summaries: list[str], book_summary: Optional[str]) -> str:
    """Joins permanent per-act summaries with the current rolling summary since
    the last act boundary — the hierarchical replacement for a single
    ever-growing book_summary string. Early acts are never re-compressed, so
    they survive intact no matter how long the novel gets."""
    parts = []
    if act_summaries:
        parts.append("\n\n".join(act_summaries))
    if book_summary:
        parts.append(book_summary)
    return "\n\n".join(parts) if parts else ""


def _fmt_history(pack: ContextPack, book_summary: Optional[str] = None) -> str:
    lines = []
    if book_summary:
        lines.append(f"NOVEL SO FAR:\n{book_summary}")
    s = pack.last_chapter_summary
    if s:
        lines.append(f"LAST CHAPTER (Chapter {s.chapter_number}):\n{s.medium_summary}")
    return "\n\n".join(lines) if lines else "  This is the first chapter — no prior history."


def _fmt_stale_plotlines(pack: ContextPack) -> str:
    if not pack.stale_plotlines:
        return ""
    lines = "\n".join(f"  • [{p.id}] {p.name} — stage: {p.progress_stage}" for p in pack.stale_plotlines)
    return (
        "\n=== THREADS GOING QUIET (no movement in a while — consider addressing, not mandatory) ===\n"
        f"{lines}\n"
    )


def _fmt_roster(roster: list) -> str:
    if not roster:
        return "  (no characters exist yet)"
    return "\n".join(
        f"  • [{e.id}] {e.name}"
        + (" [DECEASED]" if not e.is_alive else "")
        + (f" — at location {e.current_location_id}" if e.current_location_id else "")
        for e in roster
    )


def _fmt_outline(state: ChapterGraphState) -> str:
    outline = state.story_outline
    if outline is None:
        return "  (no outline yet)"
    roster_names = {e.id: e.name for e in (state.context_pack.character_roster if state.context_pack else [])}

    lines = []
    if outline.premise:
        lines.append(f"Premise: {outline.premise}")
    if outline.theme:
        lines.append(f"Theme: {outline.theme}")
    if outline.planned_ending:
        lines.append(f"Planned ending: {outline.planned_ending}")

    open_beats = [b for b in outline.beats if b.status != "completed"]
    if open_beats:
        lines.append("Beats still ahead:")
        lines.extend(f"  • [{b.status}] {b.description}" for b in open_beats)

    if outline.character_arcs:
        lines.append("Character arcs:")
        for a in outline.character_arcs:
            name = roster_names.get(a.character_id, a.character_id[:8])
            stage = f" (currently: {a.current_stage})" if a.current_stage else ""
            lines.append(f"  • {name}: {a.arc_summary}{stage}")

    return "\n".join(lines) if lines else "  (outline not yet populated)"


def build_planner_prompt(state: ChapterGraphState, template: Optional[str] = None) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["story_planner"]

    pack = state.context_pack
    if pack is None:
        raise ValueError("context_pack must be set before story planner runs")

    mode_note = {
        "cold_start": (
            "COLD START — this is the very first chapter. Establish the world, "
            "introduce the POV character, and set the initial situation."
        ),
        "continuation": (
            "CONTINUATION — advance the story naturally from where it left off. "
            "Honor open plotlines and character arcs. If the USER DIRECTIVE below "
            "contains a specific instruction or plot event, treat it as mandatory "
            "and incorporate it coherently while still advancing the story."
        ),
    }.get(state.input_mode or "", "UNKNOWN MODE")

    dep_section = ""
    if pack.dependency_graph_hits:
        dep_section = "\n=== MANDATORY CONTEXT (canon rules fired) ===\n" + "\n".join(
            f"  • [{h.rule_id}] {h.reason}: {h.content}"
            for h in pack.dependency_graph_hits
        ) + "\n"

    return tpl.format(
        mode_note=mode_note,
        outline_block=_fmt_outline(state),
        world_rules_block=_fmt_world_rules(pack),
        world_lore_block=_fmt_world_lore(pack),
        plotlines_block=_fmt_plotlines(pack),
        characters_block=_fmt_characters(pack),
        pov_block=_fmt_pov(pack),
        locations_block=_fmt_locations(pack),
        history_block=_fmt_history(pack, full_history_text(state.act_summaries, state.book_summary)),
        dep_section=dep_section,
        stale_plotlines_block=_fmt_stale_plotlines(pack),
        user_input=state.user_input,
        roster_block=_fmt_roster(pack.character_roster),
        chapter_number=state.chapter_number,
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def make_story_planner_node(
    model: Optional[str] = None,
    ollama_client=None,
    db_path=None,
) -> Callable[[ChapterGraphState], dict]:
    """
    Returns a LangGraph node. Pass ollama_client in tests to inject a mock.
    In production the shared llm_client (pointed at the local Ollama server) is used.
    """

    def node(state: ChapterGraphState) -> dict:
        from prompt_templates import get_template
        template = get_template("story_planner", state.story_id, db_path)
        prompt = build_planner_prompt(state, template=template)

        # Append a JSON template to the prompt so the model knows exactly what
        # fields are required, then force JSON output via response_format.
        data = chat_json(
            prompt + _JSON_TEMPLATE, model=model, max_tokens=4096,
            client=ollama_client, label="Story planner",
        )
        return {"story_plan": _parse_plan_json(data)}

    return node


def _parse_plan_json(data: dict) -> StoryPlan:
    """Apply defaults for any missing fields, then validate.

    Schema stays strict — defaults live here in the parser, not in the model.
    """
    # Apply top-level defaults for any missing fields
    for key, default in _PLAN_DEFAULTS.items():
        if key not in data:
            data[key] = default

    # Apply per-constraint defaults
    constraints = data.get("character_constraints", [])
    if isinstance(constraints, list):
        for i, cc in enumerate(constraints):
            if isinstance(cc, dict):
                for key, default in _CONSTRAINT_DEFAULTS.items():
                    if key not in cc:
                        cc[key] = default

    return StoryPlan.model_validate(data)


# ---------------------------------------------------------------------------
# JSON output template — appended to the prompt so models know exactly
# what fields are required. response_format=json_object forces valid JSON;
# this template ensures the right keys are present.
# ---------------------------------------------------------------------------
_JSON_TEMPLATE = """

=== OUTPUT FORMAT (respond with ONLY this JSON, no other text) ===
{
  "scenes": ["scene description 1", "scene description 2", "scene description 3"],
  "pacing_notes": "tone and rhythm guidance",
  "conflicts": ["conflict 1", "conflict 2"],
  "narrative_goals": ["goal 1", "goal 2"],
  "character_constraints": [
    {
      "character_id": "exact-character-id-from-roster",
      "forbidden_actions": ["thing they cannot do"],
      "required_callbacks": ["beat they must address"]
    }
  ],
  "required_callbacks": ["story-level beat 1"],
  "target_word_count": 1000,
  "requested_offscreen_character_ids": []
}"""

# ---------------------------------------------------------------------------
# Defaults applied when a model omits a required field.
# Schema stays strict; parser is lenient.
# ---------------------------------------------------------------------------
_PLAN_DEFAULTS: dict = {
    "scenes": [],
    "pacing_notes": "",
    "conflicts": [],
    "narrative_goals": [],
    "character_constraints": [],
    "required_callbacks": [],
    "target_word_count": 1000,
    "requested_offscreen_character_ids": [],
}

_CONSTRAINT_DEFAULTS: dict = {
    "character_id": "",
    "forbidden_actions": [],
    "required_callbacks": [],
}


story_planner_node = make_story_planner_node()  # uses NOVELGEN_MODEL env var, falls back to DEFAULT_MODEL
