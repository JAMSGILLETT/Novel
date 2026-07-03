"""
Craft Check.

Reads the chapter prose and flags genuine craft/engagement problems — not
continuity (that's Canon Check's job), but whether the chapter is actually
well written:
  - pacing            — chapter drags, or rushes past what should land
  - tension           — no real conflict, stakes, or forward pull in the scene
  - show_dont_tell    — key emotional beats are summarized/told rather than shown
  - dialogue          — dialogue reads as stilted, expository, or interchangeable
  - voice_consistency — narration drifts from the established POV voice/tense

This node is a pure single-pass check: `check(state) -> CraftCheckResult`.
Same shape as node_canon_check.py — no writer call, no loop. pipeline.py
drives the revision retry for both checks the same way.

Model: same local Ollama server as other nodes.
Structured output via response_format=json_object + JSON template + defaults
applied in the parser.
"""
from __future__ import annotations

from typing import Callable, Optional

from llm_client import chat_json
from schema import CraftCheckResult, ChapterGraphState, ContextPack, StoryPlan


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _fmt_plan(plan: StoryPlan) -> str:
    return (
        f"  Pacing notes: {plan.pacing_notes}\n"
        f"  Target length: ~{plan.target_word_count} words\n"
        f"  Conflicts meant to be dramatized: {'; '.join(plan.conflicts) if plan.conflicts else '(none)'}"
    )


_CHECK_TEMPLATE = """

=== OUTPUT FORMAT (respond with ONLY this JSON, no other text) ===
{
  "passed": true,
  "issues": []
}

If issues exist, add them to the list:
{
  "passed": false,
  "issues": [
    {
      "issue_type": "pacing | tension | show_dont_tell | dialogue | voice_consistency",
      "description": "quote or describe the specific passage and what's weak about it",
      "severity": "minor | major"
    }
  ]
}"""


def build_craft_check_prompt(state: ChapterGraphState, template: Optional[str] = None) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["craft_check"]

    plan = state.story_plan
    if plan is None or state.chapter_prose is None:
        raise ValueError("story_plan and chapter_prose must both be set before craft check runs")

    return tpl.format(
        plan_block=_fmt_plan(plan),
        chapter_prose=state.chapter_prose,
    ) + _CHECK_TEMPLATE


# ---------------------------------------------------------------------------
# Defaults (schema stays strict; parser fills gaps)
# ---------------------------------------------------------------------------

_RESULT_DEFAULTS: dict = {"passed": True, "issues": []}
_ISSUE_DEFAULTS: dict = {
    "issue_type": "pacing",
    "description": "",
    "severity": "major",
}


def _parse_check_result(data: dict) -> CraftCheckResult:
    for key, default in _RESULT_DEFAULTS.items():
        if key not in data:
            data[key] = default

    issues = data.get("issues", [])
    if isinstance(issues, list):
        for i in issues:
            if isinstance(i, dict):
                for key, default in _ISSUE_DEFAULTS.items():
                    if key not in i:
                        i[key] = default

    return CraftCheckResult.model_validate(data)


# ---------------------------------------------------------------------------
# Node factory — a single pass, no writer, no loop. pipeline.py drives retries.
# ---------------------------------------------------------------------------

def make_craft_check_node(
    model: Optional[str] = None,
    ollama_client=None,
    db_path=None,
) -> Callable[[ChapterGraphState], CraftCheckResult]:

    def check(state: ChapterGraphState) -> CraftCheckResult:
        from prompt_templates import get_template
        template = get_template("craft_check", state.story_id, db_path)
        prompt = build_craft_check_prompt(state, template=template)
        data = chat_json(
            prompt, model=model, max_tokens=2048, client=ollama_client,
            label="Craft check", response_format_fallback=True,
        )
        return _parse_check_result(data)

    return check


craft_check = make_craft_check_node()
