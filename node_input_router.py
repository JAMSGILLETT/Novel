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
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

from schema import ChapterGraphState
from db import has_any_chapters, get_latest_chapter_number, init_db


class InputClassification(BaseModel):
    """Structured output for the continuation-vs-injection judgment call."""
    mode: Literal["continuation", "user_event_injection"]
    reasoning: str = Field(description="One short sentence explaining the classification")


# --- LLM call stub --------------------------------------------------------
# This is the ONLY place in this node that would call an LLM, and only
# when chapters already exist (cold start never reaches this branch).
# Left as an explicit stub function (not hidden inline) so it's obvious
# where the real API call needs to be wired in, and so this node can be
# unit tested without hitting the network at all.

def classify_continuation_or_injection(user_input: str) -> InputClassification:
    """
    STUB — not a real implementation.

    Real version: cheap/fast model call (same tier as canon checker),
    structured output via InputClassification, prompt roughly:

      "Given this user input to a chapter-generation system, classify
       whether it is (a) a generic request to continue the story, or
       (b) an explicit instruction that should be treated as a forced
       plot event this chapter. Examples of (b): introducing a new
       character, forcing an action/encounter, changing a character's
       state directly. Examples of (a): 'continue', 'write the next
       chapter', 'what happens next'."

    For now: simple heuristic so the node is testable end-to-end without
    an API key. This MUST be swapped for a real LLM call before this
    node is considered done — flagging explicitly rather than disguising
    it as final logic.
    """
    generic_phrases = {"continue", "next chapter", "what happens next", "go on", "proceed"}
    normalized = user_input.strip().lower()
    if normalized in generic_phrases or len(normalized) < 12:
        return InputClassification(
            mode="continuation",
            reasoning="Input matches a generic continuation phrase or is too short to encode an event.",
        )
    return InputClassification(
        mode="user_event_injection",
        reasoning="Input contains specific, actionable content beyond a generic continue request.",
    )


# --- The actual node -------------------------------------------------------

def input_router_node(state: ChapterGraphState) -> dict:
    """
    LangGraph node function. Takes the current state, returns a partial
    state update (dict of fields to merge), per LangGraph convention.
    """
    init_db()  # idempotent; ensures table exists. Cheap no-op after first call.

    story_has_chapters = has_any_chapters(state.story_id)

    if not story_has_chapters:
        return {
            "input_mode": "cold_start",
            "chapter_number": 1,
        }

    latest = get_latest_chapter_number(state.story_id)
    next_chapter_number = (latest or 0) + 1

    classification = classify_continuation_or_injection(state.user_input)

    return {
        "input_mode": classification.mode,
        "chapter_number": next_chapter_number,
    }
