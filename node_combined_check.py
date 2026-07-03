"""
Combined Canon + Craft Check — replaces the two separate check nodes with a
single LLM call that evaluates both continuity and writing quality at once.

Returns a (CanonCheckResult, CraftCheckResult) pair. pipeline.py maps this
back to the same state fields as before, so the revision loop and checkpoint
system are unchanged.

Risk vs. reward: saves one LLM call per check iteration. If quality degrades
noticeably (the model conflates the two tasks), you can always revert to the
separate nodes by changing the two lines in pipeline.py that build the check
nodes.
"""
from __future__ import annotations

from typing import Callable, Optional

from llm_client import chat_json, chat_structured
from schema import (
    CanonCheckResult, CraftCheckResult, CombinedCheckResult,
    ChapterGraphState, ContextPack, StoryPlan,
)

# Reuse the context formatters from the individual check modules
from node_canon_check import (
    _fmt_world_rules, _fmt_constraints, _fmt_required_callbacks,
    _fmt_characters, _fmt_locations, _parse_check_result as _parse_canon,
)
from node_craft_check import _fmt_plan, _parse_check_result as _parse_craft


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_OUTPUT_FORMAT = """

=== OUTPUT FORMAT (respond with ONLY this JSON, no other text) ===
{
  "canon_passed": true,
  "violations": [],
  "craft_passed": true,
  "issues": []
}

If violations exist (continuity problems):
  "violations": [
    {
      "violation_type": "knowledge_leak | location_inconsistency | lore_violation | forbidden_action_violated",
      "description": "quote the offending passage and state the exact rule it breaks",
      "related_entity_id": null,
      "severity": "minor | major"
    }
  ]

If craft issues exist (writing quality problems):
  "issues": [
    {
      "issue_type": "pacing | tension | show_dont_tell | dialogue | voice_consistency",
      "description": "quote or describe the specific passage and what's weak about it",
      "severity": "minor | major"
    }
  ]

Set canon_passed to false if any violations exist.
Set craft_passed to false if any issues exist."""


def build_combined_check_prompt(
    state: ChapterGraphState, template: Optional[str] = None, include_demos: bool = True,
    history: Optional[list] = None,
) -> str:
    from prompt_templates import DEFAULT_TEMPLATES

    pack = state.context_pack
    plan = state.story_plan
    if pack is None or plan is None or state.chapter_prose is None:
        raise ValueError("context_pack, story_plan, and chapter_prose must be set before combined check runs")

    char_map = {c.id: c.name for c in pack.active_characters}

    # DSPy-lite: attach relevance-selected few-shot demonstrations so the small
    # local model sees worked examples of this exact judgment before making it.
    demo_block = ""
    if include_demos:
        from check_demos import select_demos, format_demos
        demo_block = format_demos(select_demos(pack, history=history))

    return f"""You are reviewing a novel chapter for two things simultaneously:
1. CANON CONSISTENCY — does the prose violate world rules, character knowledge, or location constraints?
2. CRAFT QUALITY — does the prose have genuine writing problems (pacing, tension, show-don't-tell, dialogue, voice)?
{demo_block}
=== WORLD RULES (hard constraints) ===
{_fmt_world_rules(pack)}

=== CHARACTER CONSTRAINTS ===
{_fmt_constraints(plan, char_map)}

=== REQUIRED STORY BEATS ===
{_fmt_required_callbacks(plan)}

=== CHARACTERS IN SCENE (what they know, their secrets) ===
{_fmt_characters(pack)}

=== LOCATIONS ===
{_fmt_locations(pack)}

=== CHAPTER PLAN (pacing, conflicts, target length) ===
{_fmt_plan(plan)}

=== CHAPTER PROSE ===
{state.chapter_prose}
{_OUTPUT_FORMAT}"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_combined(data: dict) -> tuple[CanonCheckResult, CraftCheckResult]:
    canon_data = {
        "passed": data.get("canon_passed", True),
        "violations": data.get("violations", []),
    }
    craft_data = {
        "passed": data.get("craft_passed", True),
        "issues": data.get("issues", []),
    }
    return _parse_canon(canon_data), _parse_craft(craft_data)


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_combined_check_node(
    model: Optional[str] = None,
    ollama_client=None,
    db_path=None,
) -> Callable[[ChapterGraphState], tuple[CanonCheckResult, CraftCheckResult]]:

    # Mined clean-pass history is stable across a chapter's many check calls
    # (best-of-N × attempts), so fetch it once per node build.
    _history_cache: dict = {}

    def _history(story_id: str) -> list:
        if "h" not in _history_cache:
            try:
                import db
                _history_cache["h"] = db.get_recent_check_verdicts(story_id, limit=20, db_path=db_path)
            except Exception:
                _history_cache["h"] = []
        return _history_cache["h"]

    def check(state: ChapterGraphState) -> tuple[CanonCheckResult, CraftCheckResult]:
        prompt = build_combined_check_prompt(state, history=_history(state.story_id))
        try:
            result = chat_structured(
                prompt, CombinedCheckResult, model=model, max_tokens=2048,
                client=ollama_client, label="Combined check",
            )
            return result.split()
        except Exception as e:
            # Instructor exhausted its retries (or isn't available) — fall back
            # to the tolerant hand-rolled parser so a check is never skipped.
            # Warn loudly: a *persistent* fallback (e.g. instructor failing to
            # import) silently loses the validation guarantee, which is exactly
            # the bug chat_structured was added to fix.
            print(f"  [warn] Combined check: structured path failed, using JSON fallback: "
                  f"{type(e).__name__}: {str(e)[:150]}")
            data = chat_json(
                prompt, model=model, max_tokens=2048, client=ollama_client,
                label="Combined check", response_format_fallback=True,
            )
            return _parse_combined(data)

    return check
