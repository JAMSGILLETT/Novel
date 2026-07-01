"""
Node 4: Character Reasoner.

For each character who appears in this chapter, produces a CharacterReasoning:
  action_intentions       — what the character plans to do this chapter
  dialogue_intent         — what they want to communicate / what they're hiding
  emotional_response      — how they feel about the situation and events
  constraint_acknowledgement — how they reconcile their forbidden_actions and
                               required_callbacks with their own goals

Characters processed:
  1. active_characters from the ContextPack (already fetched by Node 2)
  2. requested_offscreen_character_ids from the StoryPlan (fetched here from DB;
     warn and skip if an ID is not found)

Reasoning order: the POV character goes first (their intent anchors the
scene), then everyone else in the order the context pack provided them.

Characters do NOT see each other's private reasoning — only action_intentions
(what a character plans to physically do), never dialogue_intent or
emotional_response. This lets later characters react to what's already in
motion in the scene without knowing anyone else's hidden thoughts or secrets,
so deception and private framing stay private, and the Story Writer still
has to discover how dialogue actually plays out rather than transcribing a
pre-agreed choreography. Each character otherwise still sees only:
  - Their own full Character profile
  - Their CharacterConstraint (forbidden_actions + required_callbacks) from the plan,
    or an empty constraint if the planner didn't constrain them
  - The plan's scenes, pacing_notes, and conflicts (so they reason in context)
  - The current POV state (location, companions, emotional context)

Model: same local Ollama server as Node 3.
Structured output via forced tool-call to CharacterReasoning schema.

Factory pattern: make_character_reasoner_node() injects dependencies so tests
can pass a fake client and db_path without touching production singletons.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import db
from llm_client import MODEL, chat_json
from node_story_planner import full_history_text
from schema import (
    Character, CharacterConstraint, CharacterReasoning,
    ChapterGraphState, ContextPack, StoryPlan,
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _fmt_prior_actions(prior_actions: list[tuple[str, list[str]]]) -> str:
    if not prior_actions:
        return ""
    lines = []
    for name, intentions in prior_actions:
        if intentions:
            lines.append(f"  • {name} plans to: {'; '.join(intentions)}")
    if not lines:
        return ""
    return (
        "\n=== WHAT'S ALREADY IN MOTION THIS SCENE ===\n"
        "(Other characters' planned actions only — you do NOT know their private thoughts, dialogue "
        "intentions, or feelings. React to what they're doing, not what they're thinking.)\n"
        + "\n".join(lines) + "\n"
    )


def _build_character_prompt(
    character: Character,
    constraint: CharacterConstraint,
    plan: StoryPlan,
    pack: ContextPack,
    history_text: Optional[str] = None,
    prior_actions: Optional[list[tuple[str, list[str]]]] = None,
    template: Optional[str] = None,
) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["character_reasoner"]

    loc_name = "unknown"
    if pack.pov_state:
        for l in pack.nearby_locations:
            if l.id == pack.pov_state.location_id:
                loc_name = l.name
                break

    goals = "; ".join(character.goals) if character.goals else "none stated"
    objs = "; ".join(character.current_objectives) if character.current_objectives else "none"
    knowledge = "; ".join(character.knowledge) if character.knowledge else "none"
    rels = (
        "\n".join(f"    {k}: {v}" for k, v in character.relationships.items())
        if character.relationships else "    none"
    )
    secrets = "; ".join(character.secrets) if character.secrets else "none"

    scenes_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan.scenes))
    conflicts_text = "\n".join(f"  • {c}" for c in plan.conflicts) if plan.conflicts else "  (none)"

    forbidden = (
        "\n".join(f"  • {a}" for a in constraint.forbidden_actions)
        if constraint.forbidden_actions else "  (none)"
    )
    required = (
        "\n".join(f"  • {r}" for r in constraint.required_callbacks)
        if constraint.required_callbacks else "  (none)"
    )

    pov_section = "  (not yet established)"
    if pack.pov_state:
        companions = ", ".join(pack.pov_state.companions) if pack.pov_state.companions else "none"
        pov_section = (
            f"  Location: {loc_name}\n"
            f"  Companions present: {companions}\n"
            f"  Overall emotional tone: {pack.pov_state.emotional_state or 'unknown'}"
        )

    history_section = ""
    if history_text:
        history_section = f"\n=== NOVEL SO FAR ===\n{history_text}\n"

    prior_section = _fmt_prior_actions(prior_actions or [])

    return tpl.format(
        history_section=history_section,
        character_name=character.name,
        personality=character.personality,
        goals=goals,
        objectives=objs,
        emotional_state=character.emotional_state or "unknown",
        knowledge=knowledge,
        secrets=secrets,
        relationships=rels,
        scenes_block=scenes_text,
        pacing_notes=plan.pacing_notes,
        conflicts_block=conflicts_text,
        forbidden_block=forbidden,
        required_block=required,
        pov_section=pov_section,
        prior_section=prior_section,
    )


# ---------------------------------------------------------------------------
# JSON output template + defaults (same philosophy as story planner)
# ---------------------------------------------------------------------------

_REASONING_TEMPLATE = """

=== OUTPUT FORMAT (respond with ONLY this JSON, no other text) ===
{
  "character_id": "leave blank — will be overwritten",
  "action_intentions": ["specific thing they plan to do 1", "specific thing 2"],
  "dialogue_intent": "what they want to communicate and what they are hiding",
  "emotional_response": "how they feel about the situation",
  "constraint_acknowledgement": ["one sentence per constraint on how they reconcile it"]
}"""

_REASONING_DEFAULTS: dict = {
    "character_id": "",
    "action_intentions": [],
    "dialogue_intent": "",
    "emotional_response": "",
    "constraint_acknowledgement": [],
}


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_character_reasoner_node(
    model: str = MODEL,
    ollama_client=None,
    db_path: Optional[Path] = None,
) -> Callable[[ChapterGraphState], dict]:

    def _reason_one(
        character: Character, constraint: CharacterConstraint,
        plan: StoryPlan, pack: ContextPack,
        history_text: Optional[str],
        prior_actions: list[tuple[str, list[str]]],
        story_id: str,
    ) -> CharacterReasoning:
        from prompt_templates import get_template
        template = get_template("character_reasoner", story_id, db_path)
        prompt = _build_character_prompt(
            character, constraint, plan, pack, history_text, prior_actions, template=template
        )

        data = chat_json(
            prompt + _REASONING_TEMPLATE, model=model, max_tokens=1024,
            client=ollama_client, label=f"Character reasoner ({character.name})",
        )
        for key, default in _REASONING_DEFAULTS.items():
            if key not in data:
                data[key] = default
        reasoning = CharacterReasoning.model_validate(data)
        # Ensure the character_id is set correctly regardless of what the model returned
        return reasoning.model_copy(update={"character_id": character.id})

    def node(state: ChapterGraphState) -> dict:
        pack = state.context_pack
        plan = state.story_plan
        if pack is None:
            raise ValueError("context_pack must be set before character reasoner runs")
        if plan is None:
            raise ValueError("story_plan must be set before character reasoner runs")

        # Index constraints by character_id for O(1) lookup
        constraint_map: dict[str, CharacterConstraint] = {
            cc.character_id: cc for cc in plan.character_constraints
        }
        empty_constraint = CharacterConstraint(
            character_id="", forbidden_actions=[], required_callbacks=[]
        )

        # Start with characters already in the context pack
        characters_to_reason: list[Character] = list(pack.active_characters)
        already_have: set[str] = {c.id for c in characters_to_reason}

        # Fetch any off-screen characters the planner requested
        for cid in plan.requested_offscreen_character_ids:
            if cid in already_have:
                continue
            character = db.get_character_by_id(cid, state.story_id, db_path)
            if character is None:
                print(f"  [character_reasoner] WARNING: requested off-screen character "
                      f"{cid!r} not found in DB — skipping")
                continue
            characters_to_reason.append(character)
            already_have.add(cid)

        # POV character reasons first — their intent anchors the scene, giving the
        # ordering a narrative reason rather than being arbitrary. Stable sort keeps
        # everyone else in the order the context pack provided them.
        pov_id = pack.pov_state.pov_character_id if pack.pov_state else None
        characters_to_reason.sort(key=lambda c: 0 if c.id == pov_id else 1)

        history_text = full_history_text(state.act_summaries, state.book_summary)

        reasonings: list[CharacterReasoning] = []
        prior_actions: list[tuple[str, list[str]]] = []
        for character in characters_to_reason:
            constraint = constraint_map.get(character.id, empty_constraint)
            print(f"  Reasoning for: {character.name}")
            reasoning = _reason_one(character, constraint, plan, pack, history_text, prior_actions, state.story_id)
            reasonings.append(reasoning)
            prior_actions.append((character.name, reasoning.action_intentions))

        return {"character_reasonings": reasonings}

    return node


character_reasoner_node = make_character_reasoner_node()
