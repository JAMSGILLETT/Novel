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
import time
from typing import Callable, Optional

from llm_json import parse_json_response
from node_story_planner import OLLAMA_BASE_URL, MODEL
from schema import ChapterGraphState, ChapterSummary

_OUTPUT_FORMAT = """

=== OUTPUT FORMAT (respond with ONLY this JSON) ===
{
  "short_summary": "One sentence summary here.",
  "medium_summary": "Two to four sentence summary here.",
  "timeline_events": [
    "Kael confronted Mira at the docks about the ledger.",
    "Mira denied knowing where it was but her expression betrayed her."
  ]
}"""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_summarizer_prompt(state: ChapterGraphState, template: Optional[str] = None) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["chapter_summarizer"]
    return tpl.format(
        chapter_number=state.chapter_number,
        chapter_prose=state.chapter_prose,
    ) + _OUTPUT_FORMAT


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "short_summary": "",
    "medium_summary": "",
    "timeline_events": [],
}


def _parse_summary(raw_content: str, chapter_number: int) -> ChapterSummary:
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

    data["chapter_number"] = chapter_number
    return ChapterSummary.model_validate(data)


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_chapter_summarizer_node(
    model: str = MODEL,
    ollama_client=None,
    db_path=None,
) -> Callable[[ChapterGraphState], dict]:

    def _client():
        if ollama_client is not None:
            return ollama_client
        from openai import OpenAI
        return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    def node(state: ChapterGraphState) -> dict:
        if not state.chapter_prose:
            raise ValueError("chapter_prose must be set before chapter summarizer runs")

        from prompt_templates import get_template
        template = get_template("chapter_summarizer", state.story_id, db_path)
        prompt = build_summarizer_prompt(state, template=template)

        for attempt in range(3):
            try:
                response = _client().chat.completions.create(
                    model=model,
                    max_tokens=1024,
                    timeout=300,
                    tools=[{
                        "type": "function",
                        "function": {
                            "name": "chapter_summary",
                            "description": "Structured summary of a novel chapter.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "short_summary": {"type": "string"},
                                    "medium_summary": {"type": "string"},
                                    "timeline_events": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["short_summary", "medium_summary", "timeline_events"],
                            },
                        },
                    }],
                    tool_choice={"type": "function", "function": {"name": "chapter_summary"}},
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt * 3
                print(f"  Ollama error — retrying in {wait}s (attempt {attempt + 1}/3): {e}")
                time.sleep(wait)

        msg = response.choices[0].message
        if msg.tool_calls:
            raw = msg.tool_calls[0].function.arguments
        else:
            raw = msg.content or ""

        # If we got nothing, retry without tool_choice (model ignored it)
        if not raw.strip():
            print("  [summarizer] Empty response — retrying without tool_choice...")
            retry = _client().chat.completions.create(
                model=model,
                max_tokens=1024,
                timeout=300,
                messages=[{"role": "user", "content": prompt}],
            )
            retry_msg = retry.choices[0].message
            if retry_msg.tool_calls:
                raw = retry_msg.tool_calls[0].function.arguments
            else:
                raw = retry_msg.content or ""

        summary = _parse_summary(raw, state.chapter_number)
        return {"chapter_summary": summary}

    return node


chapter_summarizer_node = make_chapter_summarizer_node()
