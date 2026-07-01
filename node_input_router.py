"""
Node 1: Input router.

Responsibility: classify the incoming request into one of three modes
before any retrieval or generation happens:

  - cold_start: no chapters exist yet for this story (first invocation)
  - continuation: chapters exist, and the user input is "just write the
    next chapter" (no explicit injected event)
  - user_event_injection: chapters exist, and the user input contains an
    explicit instruction that should be treated as a forced plot event
    (e.g. "have a stranger attack Alice at the market")

This node does NOT do any heavy reasoning. It does one DB read (cold
start check) and one cheap classification of the user input. It does not
touch the LLM for the cold_start vs continuation distinction — that's a
pure DB fact, not a judgment call. It DOES use a cheap LLM call to
distinguish continuation vs user_event_injection, because that's a
judgment call about intent that a keyword check would get wrong often
enough to matter (e.g. "Alice should be more cautious next chapter" is
arguably a soft injection, "what happens next" is not).

Model: same local Ollama server as other nodes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from llm_client import MODEL, chat_json
from prompt_templates import get_template
from schema import ChapterGraphState
from db import has_any_chapters, get_latest_chapter_number, init_db

_OUTPUT_FORMAT = """

=== OUTPUT FORMAT (respond with ONLY this JSON, no other text) ===
{
  "mode": "continuation | user_event_injection",
  "reasoning": "one short sentence explaining the classification"
}"""


class InputClassification(BaseModel):
    """Structured output for the continuation-vs-injection judgment call."""
    mode: Literal["continuation", "user_event_injection"]
    reasoning: str = Field(description="One short sentence explaining the classification")


def build_classification_prompt(user_input: str, template: Optional[str] = None) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["input_router_classification"]
    return tpl.format(user_input=user_input) + _OUTPUT_FORMAT


def make_input_router_node(
    model: str = MODEL,
    ollama_client=None,
    db_path: Optional[Path] = None,
) -> Callable[[ChapterGraphState], dict]:
    """Returns the Node 1 function. Pass ollama_client in tests to inject a fake."""

    def classify_continuation_or_injection(user_input: str, story_id: str) -> InputClassification:
        template = get_template("input_router_classification", story_id, db_path)
        prompt = build_classification_prompt(user_input, template=template)
        data = chat_json(
            prompt, model=model, max_tokens=256, timeout=120,
            client=ollama_client, label="Input router",
        )
        data.setdefault("mode", "continuation")
        data.setdefault("reasoning", "")
        return InputClassification.model_validate(data)

    def node(state: ChapterGraphState) -> dict:
        """LangGraph node function. Takes the current state, returns a partial
        state update (dict of fields to merge), per LangGraph convention."""
        init_db()  # idempotent; ensures table exists. Cheap no-op after first call.

        story_has_chapters = has_any_chapters(state.story_id)

        if not story_has_chapters:
            return {
                "input_mode": "cold_start",
                "chapter_number": 1,
            }

        latest = get_latest_chapter_number(state.story_id)
        next_chapter_number = (latest or 0) + 1

        classification = classify_continuation_or_injection(state.user_input, state.story_id)

        return {
            "input_mode": classification.mode,
            "chapter_number": next_chapter_number,
        }

    return node


input_router_node = make_input_router_node()  # uses NOVELGEN_MODEL env var, falls back to DEFAULT_MODEL
