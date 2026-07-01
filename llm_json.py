"""
Shared JSON-parsing helpers for local-LLM structured output.

Every node that asks Ollama for JSON deals with the same three problems:
  - reasoning models wrap their real answer in <think>...</think>
  - small models sometimes wrap the JSON in a markdown fence, or emit trailing
    prose, or return a tool-call envelope ({"name": ..., "parameters": {...}})
    even when tool calling wasn't requested
  - small models produce almost-valid JSON (trailing commas, single quotes)

This used to be duplicated across node_story_planner, node_canon_check,
node_character_reasoner, node_chapter_summarizer, and node_memory_extractor.
It now lives in one place so a fix here fixes every node.
"""
from __future__ import annotations

import json
from typing import Optional, Union


def strip_think(raw: str) -> str:
    """Remove a leading <think>...</think> block emitted by reasoning models."""
    if "<think>" not in raw:
        return raw
    parts = raw.split("</think>", 1)
    if len(parts) > 1:
        content = parts[1].strip()
        if content:
            return content
    # No closing tag found, or nothing after it — fall back to whatever was
    # inside the think block itself rather than returning nothing.
    inner = raw.split("<think>", 1)[1]
    return inner.split("</think>")[0].strip()


def extract_json_block(text: str) -> Optional[str]:
    """Return the first JSON object found in text, or None.

    Tracks string boundaries so braces inside string values don't confuse
    the depth counter. Returns the full text from the first { to the matching }
    — even if the JSON is malformed (repair_json handles that later).
    """
    import re

    # Fenced block: ```json ... ``` or ``` ... ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)

    # Bare: find outermost { ... }, skipping brace characters inside strings
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not in_string:
            in_string = True
            continue
        if ch == '"' and in_string:
            in_string = False
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]

    # No matching close brace — return everything from { so repair_json can patch it
    return text[start:] if depth > 0 else None


def unwrap_tool_envelope(data: Union[dict, list]) -> Union[dict, list]:
    """Some models wrap their answer in a tool-call-shaped envelope
    ({"name": ..., "parameters"/"arguments"/"input": {...}}) even when no
    tool calling was requested. Unwrap it if present."""
    if isinstance(data, dict) and "name" in data:
        for key in ("parameters", "arguments", "input"):
            if key in data and isinstance(data[key], (dict, list)):
                return data[key]
    return data


def loads_repaired(json_str: str) -> Union[dict, list]:
    """json.loads with a json_repair fallback for near-valid LLM output."""
    try:
        from json_repair import repair_json
        return json.loads(repair_json(json_str))
    except ImportError:
        return json.loads(json_str)


def parse_json_response(raw_content: str, error_label: str = "response") -> Union[dict, list]:
    """Full pipeline: strip <think>, extract the JSON object, repair+parse,
    unwrap a tool-call envelope. Raises ValueError with context on failure."""
    content = strip_think(raw_content)
    json_str = extract_json_block(content)
    if not json_str:
        raise ValueError(f"{error_label}: no parseable JSON in response.\nContent:\n{raw_content[:600]}")
    data = loads_repaired(json_str)
    return unwrap_tool_envelope(data)
