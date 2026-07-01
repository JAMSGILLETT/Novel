"""
Node 6: Canon Check.

Reads the chapter prose against the context pack and story plan and flags
contradictions:
  - World rules violated in the prose
  - Forbidden actions (from character_constraints) that appear anyway
  - Required callbacks (from the plan) that were never addressed
  - Knowledge leaks — a character references something they couldn't know
  - Location inconsistencies — a character appears somewhere they can't be

This node is a pure single-pass check: `check(state) -> CanonCheckResult`.
It does not call the writer and does not loop. The revision retry loop
(re-check, decide whether to ask the writer to fix violations, or give up and
flag for review) lives in pipeline.py, which owns both this check and the
craft check the same way — so adding a second check never means duplicating
a retry-loop hack inside the check node itself.

Model: same local Ollama server as other nodes.
Structured output via response_format=json_object + JSON template + defaults
applied in the parser (same pattern as nodes 3 and 4).
"""
from __future__ import annotations

from typing import Callable, Optional

from llm_client import MODEL, chat_json
from schema import (
    CanonCheckResult, ChapterGraphState, ContextPack, StoryPlan,
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _fmt_world_rules(pack: ContextPack) -> str:
    if not pack.relevant_world_rules:
        return "  (none defined)"
    return "\n".join(
        f"  • [{r.rule_type}] {r.title}: {r.content}" for r in pack.relevant_world_rules
    )


def _fmt_constraints(plan: StoryPlan, char_map: dict[str, str]) -> str:
    lines = []
    for cc in plan.character_constraints:
        name = char_map.get(cc.character_id, cc.character_id or "unknown character")
        for fa in cc.forbidden_actions:
            lines.append(f"  • {name} must NOT: {fa}")
        for rb in cc.required_callbacks:
            lines.append(f"  • {name} must: {rb}")
    return "\n".join(lines) if lines else "  (none)"


def _fmt_required_callbacks(plan: StoryPlan) -> str:
    if not plan.required_callbacks:
        return "  (none)"
    return "\n".join(f"  • {rb}" for rb in plan.required_callbacks)


def _fmt_characters(pack: ContextPack) -> str:
    if not pack.active_characters:
        return "  (no characters in scene)"
    lines = []
    for c in pack.active_characters:
        knowledge = "; ".join(c.knowledge) if c.knowledge else "nothing notable on record"
        secrets = "; ".join(c.secrets) if c.secrets else "none"
        goals = "; ".join(c.goals) if c.goals else "none"
        rels = ", ".join(f"{k}: {v}" for k, v in c.relationships.items()) if c.relationships else "none"
        lines.append(
            f"  • {c.name}\n"
            f"    Personality: {c.personality}\n"
            f"    Goals: {goals}\n"
            f"    Relationships: {rels}\n"
            f"    What they know: {knowledge}\n"
            f"    Secrets (must NOT be revealed by others without justification): {secrets}"
        )
    return "\n".join(lines)


def _fmt_locations(pack: ContextPack) -> str:
    if not pack.nearby_locations:
        return "  (none established)"
    return "\n".join(f"  • {l.name}: {l.description}" for l in pack.nearby_locations)


_CHECK_TEMPLATE = """

=== OUTPUT FORMAT (respond with ONLY this JSON, no other text) ===
{
  "passed": true,
  "violations": []
}

If violations exist, add them to the list:
{
  "passed": false,
  "violations": [
    {
      "violation_type": "knowledge_leak | location_inconsistency | lore_violation | forbidden_action_violated",
      "description": "quote the specific offending passage and state the exact rule it breaks",
      "related_entity_id": null,
      "severity": "minor | major"
    }
  ]
}"""


def build_canon_check_prompt(state: ChapterGraphState, template: Optional[str] = None) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["canon_check"]

    pack = state.context_pack
    plan = state.story_plan
    if pack is None or plan is None or state.chapter_prose is None:
        raise ValueError("context_pack, story_plan, and chapter_prose must all be set before canon check runs")

    char_map = {c.id: c.name for c in pack.active_characters}

    return tpl.format(
        world_rules_block=_fmt_world_rules(pack),
        constraints_block=_fmt_constraints(plan, char_map),
        required_callbacks_block=_fmt_required_callbacks(plan),
        characters_block=_fmt_characters(pack),
        locations_block=_fmt_locations(pack),
        chapter_prose=state.chapter_prose,
    ) + _CHECK_TEMPLATE


# ---------------------------------------------------------------------------
# Defaults (schema stays strict; parser fills gaps)
# ---------------------------------------------------------------------------

_RESULT_DEFAULTS: dict = {"passed": True, "violations": []}
_VIOLATION_DEFAULTS: dict = {
    "violation_type": "lore_violation",
    "description": "",
    "related_entity_id": None,
    "severity": "major",
}


def _parse_check_result(data: dict) -> CanonCheckResult:
    for key, default in _RESULT_DEFAULTS.items():
        if key not in data:
            data[key] = default

    violations = data.get("violations", [])
    if isinstance(violations, list):
        for v in violations:
            if isinstance(v, dict):
                for key, default in _VIOLATION_DEFAULTS.items():
                    if key not in v:
                        v[key] = default

    return CanonCheckResult.model_validate(data)


# ---------------------------------------------------------------------------
# Node factory — a single pass, no writer, no loop. pipeline.py drives retries.
# ---------------------------------------------------------------------------

def make_canon_check_node(
    model: str = MODEL,
    ollama_client=None,
    db_path=None,
) -> Callable[[ChapterGraphState], CanonCheckResult]:

    def check(state: ChapterGraphState) -> CanonCheckResult:
        from prompt_templates import get_template
        template = get_template("canon_check", state.story_id, db_path)
        prompt = build_canon_check_prompt(state, template=template)
        data = chat_json(
            prompt, model=model, max_tokens=2048, client=ollama_client,
            label="Canon check", response_format_fallback=True,
        )
        return _parse_check_result(data)

    return check


canon_check = make_canon_check_node()
