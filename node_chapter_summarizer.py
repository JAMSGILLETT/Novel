"""
Node 7: Chapter Summarizer.

Reads the final approved chapter prose and produces a structured summary
that is stored in SQLite and used by future context builders as
last_chapter_summary.

Outputs three levels of summary so downstream nodes can choose granularity:
  short_summary   — one sentence, used in character roster / quick checks
  medium_summary  — 2–4 sentences, used by the context builder as chapter history
  timeline_events — bullet list of concrete, datable facts (who did what, where)

Model: same local Ollama server as other nodes.
Structured output via tool-calling with the same json_repair fallback chain.
"""
from __future__ import annotations

import json as _json
from typing import Callable, Optional

from llm_client import chat
from llm_json import parse_json_response
from schema import ChapterGraphState, ChapterSummary

def _output_format(include_book_summary: bool) -> str:
    book_field = (
        ',\n  "updated_book_summary": "Updated 150-250 word running novel summary incorporating this chapter."'
        if include_book_summary else ""
    )
    return f"""

=== OUTPUT FORMAT (respond with ONLY this JSON) ===
{{
  "short_summary": "One sentence summary here.",
  "medium_summary": "Two to four sentence summary here.",
  "timeline_events": [
    "Kael confronted Mira at the docks about the ledger.",
    "Mira denied knowing where it was but her expression betrayed her."
  ]{book_field}
}}"""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_summarizer_prompt(
    state: ChapterGraphState,
    template: Optional[str] = None,
    current_book_summary: Optional[str] = None,
) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["chapter_summarizer"]
    prompt = tpl.format(
        chapter_number=state.chapter_number,
        chapter_prose=state.chapter_prose,
    )
    if current_book_summary:
        prompt += f"\n\n=== RUNNING NOVEL SUMMARY (update this to include the new chapter) ===\n{current_book_summary}"
    return prompt + _output_format(include_book_summary=bool(current_book_summary))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "short_summary": "",
    "medium_summary": "",
    "timeline_events": [],
}


def _parse_summary(raw_content: str, chapter_number: int) -> tuple[ChapterSummary, Optional[str]]:
    data = parse_json_response(raw_content, error_label="Chapter summarizer")

    for key, default in _DEFAULTS.items():
        if key not in data:
            data[key] = default

    events = data.get("timeline_events", [])
    if isinstance(events, str):
        try:
            events = _json.loads(events)
        except Exception:
            events = [e.strip("- •").strip() for e in events.splitlines() if e.strip()]
        data["timeline_events"] = events

    updated_book_summary: Optional[str] = data.pop("updated_book_summary", None) or None

    data["chapter_number"] = chapter_number
    return ChapterSummary.model_validate(data), updated_book_summary


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_chapter_summarizer_node(
    model: Optional[str] = None,
    ollama_client=None,
    db_path=None,
    print_fn=print,
) -> Callable[[ChapterGraphState], dict]:
    _p = print_fn

    def _tool_schema(with_book_summary: bool) -> list:
        props = {
            "short_summary": {"type": "string"},
            "medium_summary": {"type": "string"},
            "timeline_events": {"type": "array", "items": {"type": "string"}},
        }
        required = ["short_summary", "medium_summary", "timeline_events"]
        if with_book_summary:
            props["updated_book_summary"] = {"type": "string", "description": "Updated running novel summary incorporating this chapter. 150-250 words."}
            required.append("updated_book_summary")
        return [{
            "type": "function",
            "function": {
                "name": "chapter_summary",
                "description": "Structured summary of a novel chapter.",
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }]

    def _raw_from(response) -> str:
        msg = response.choices[0].message
        if msg.tool_calls:
            return msg.tool_calls[0].function.arguments
        return msg.content or ""

    def node(state: ChapterGraphState) -> dict:
        if not state.chapter_prose:
            raise ValueError("chapter_prose must be set before chapter summarizer runs")

        from prompt_templates import get_template
        template = get_template("chapter_summarizer", state.story_id, db_path)
        current_book_summary = state.book_summary
        prompt = build_summarizer_prompt(state, template=template, current_book_summary=current_book_summary)
        tools = _tool_schema(with_book_summary=bool(current_book_summary))

        response = chat(
            prompt, model=model, max_tokens=1500, timeout=300,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "chapter_summary"}},
            client=ollama_client, label="Chapter summarizer",
        )
        raw = _raw_from(response)

        # If we got nothing, retry without tool_choice (model ignored it)
        if not raw.strip():
            _p("  [summarizer] Empty response — retrying without tool_choice...")
            retry = chat(
                prompt, model=model, max_tokens=1500, timeout=300,
                client=ollama_client, label="Chapter summarizer",
            )
            raw = _raw_from(retry)

        summary, updated_book_summary = _parse_summary(raw, state.chapter_number)
        _p(f"  Chapter {state.chapter_number} summarized")
        if updated_book_summary:
            _p(f"  Book summary updated inline ({len(updated_book_summary.split())} words)")

        result: dict = {"chapter_summary": summary}
        if updated_book_summary:
            result["book_summary"] = updated_book_summary
        return result

    return node


chapter_summarizer_node = make_chapter_summarizer_node()
