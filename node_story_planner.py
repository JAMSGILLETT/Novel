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

Model: Llama 3.3 70B Instruct via OpenRouter (free tier).
Structured output via OpenAI-compatible tool calling — tool_choice is forced to
"story_plan" so the model cannot return freeform text.

Requires:
  pip install openai
  OPENROUTER_API_KEY environment variable set.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

from schema import (
    CharacterRosterEntry, ContextPack, ChapterGraphState, StoryPlan,
)

MODEL = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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


def _fmt_history(pack: ContextPack) -> str:
    s = pack.last_chapter_summary
    if s is None:
        return "  This is the first chapter — no prior history."
    return f"  Chapter {s.chapter_number}: {s.medium_summary}"


def _fmt_roster(roster: list) -> str:
    if not roster:
        return "  (no characters exist yet)"
    return "\n".join(
        f"  • [{e.id}] {e.name}"
        + (" [DECEASED]" if not e.is_alive else "")
        + (f" — at location {e.current_location_id}" if e.current_location_id else "")
        for e in roster
    )


def build_planner_prompt(state: ChapterGraphState) -> str:
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
            "Honor open plotlines and character arcs."
        ),
        "user_event_injection": (
            "USER EVENT INJECTION — the user has specified a forced plot event (see directive). "
            "Incorporate it coherently while still advancing the story."
        ),
    }.get(state.input_mode or "", "UNKNOWN MODE")

    dep_section = ""
    if pack.dependency_graph_hits:
        dep_section = "\n=== MANDATORY CONTEXT (canon rules fired) ===\n" + "\n".join(
            f"  • [{h.rule_id}] {h.reason}: {h.content}"
            for h in pack.dependency_graph_hits
        ) + "\n"

    return f"""You are the Story Director for a collaborative novel. Your job is to PLAN the next chapter — not write it. The Story Writer executes your plan.

=== MODE: {mode_note} ===

=== WORLD RULES (ABSOLUTE — never violate, never have a character or scene violate these) ===
{_fmt_world_rules(pack)}

=== RELEVANT WORLD LORE ===
{_fmt_world_lore(pack)}

=== ACTIVE PLOTLINES (advance at least one) ===
{_fmt_plotlines(pack)}

=== CHARACTERS IN SCENE (full profiles) ===
{_fmt_characters(pack)}

=== POV STATE ===
{_fmt_pov(pack)}

=== NEARBY LOCATIONS ===
{_fmt_locations(pack)}

=== STORY HISTORY ===
{_fmt_history(pack)}
{dep_section}
=== USER DIRECTIVE ===
"{state.user_input}"

=== FULL CHARACTER ROSTER (all characters including off-screen) ===
{_fmt_roster(pack.character_roster)}
  To bring an off-screen character back, put their exact ID in requested_offscreen_character_ids.
  Do NOT include deceased characters unless their return is plot-justified.

=== YOUR TASK — plan chapter {state.chapter_number} ===
Produce a StoryPlan with:
  • scenes: 3–5 ordered scene descriptions forming a coherent chapter arc
  • pacing_notes: tone and rhythm guidance (e.g. "slow burn, end on a revelation")
  • conflicts: specific tensions to dramatize this chapter
  • narrative_goals: what this chapter must accomplish (theme, plot, character)
  • character_constraints: for each character who appears:
      - forbidden_actions: things they CANNOT do (consistency or world-rule constraints)
      - required_callbacks: specific beats or plotline IDs they MUST address
  • required_callbacks: story-level mandatory beats independent of any single character
  • requested_offscreen_character_ids: roster IDs you want brought in from off-screen (empty list if none)

Rules:
  - Never violate World Rules
  - Do not kill a character without a required_callback justifying it
  - Reference plotlines by their ID in required_callbacks so the canon checker can verify"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def _make_openrouter_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Get a free key at https://openrouter.ai"
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def make_story_planner_node(
    model: str = MODEL,
    openrouter_client=None,
) -> Callable[[ChapterGraphState], dict]:
    """
    Returns a LangGraph node. Pass openrouter_client in tests to inject a mock.
    In production the real OpenAI client pointed at OpenRouter is built lazily.
    """

    def _client():
        return openrouter_client if openrouter_client is not None else _make_openrouter_client()

    def node(state: ChapterGraphState) -> dict:
        prompt = build_planner_prompt(state)

        response = _client().chat.completions.create(
            model=model,
            max_tokens=2048,
            tools=[{
                "type": "function",
                "function": {
                    "name": "story_plan",
                    "description": "The structured plan for this chapter of the novel.",
                    "parameters": StoryPlan.model_json_schema(),
                },
            }],
            tool_choice={"type": "function", "function": {"name": "story_plan"}},
            messages=[{"role": "user", "content": prompt}],
        )

        tool_call = response.choices[0].message.tool_calls[0]
        plan = StoryPlan.model_validate_json(tool_call.function.arguments)
        return {"story_plan": plan}

    return node


story_planner_node = make_story_planner_node()
