"""
Shared Ollama access layer (imported as `llm_client`).

Every node used to carry its own copy of three things:
  - the base URL / model constants,
  - a `_client()` closure that built a fresh OpenAI client per call,
  - a `for attempt in range(3)` retry loop with exponential backoff.

That duplication now lives here once, so a fix to retry behaviour, timeouts,
or model configuration happens in a single place. `llm_json.py` already did
this for JSON *parsing*; this module does it for the *call*.

Design notes:
  - One process-wide client is built lazily and reused (get_client()). Tests
    still inject their own fake via the `client=` argument.
  - `keep_alive` is sent so Ollama keeps the (large) model resident between the
    ~8 calls that make up one chapter instead of unloading/reloading each time.
  - `chat()` returns the raw response (needed for tool-calling paths);
    `chat_text()` and `chat_json()` cover the common return shapes.
"""
from __future__ import annotations

import os
import time
from typing import Any, List, Optional, Union

DEFAULT_MODEL = "qwen2.5:14b-instruct-q4_K_M"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Override the model at runtime without touching code:  set NOVELGEN_MODEL=...
MODEL = os.environ.get("NOVELGEN_MODEL", DEFAULT_MODEL)

# How long Ollama should keep the model loaded after a request. Keeping it warm
# across a chapter's calls avoids repeated multi-second weight reloads.
# Override with NOVELGEN_KEEP_ALIVE (e.g. "0" to unload immediately, "-1" forever).
KEEP_ALIVE = os.environ.get("NOVELGEN_KEEP_ALIVE", "30m")

_shared_client = None


def get_client(client: Optional[Any] = None):
    """Return the injected client (tests) or a lazily-built shared singleton."""
    if client is not None:
        return client
    global _shared_client
    if _shared_client is None:
        from openai import OpenAI
        _shared_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    return _shared_client


def chat(
    prompt: Union[str, List[dict]],
    *,
    model: str = MODEL,
    max_tokens: int = 2048,
    timeout: int = 600,
    response_format: Optional[dict] = None,
    tools: Optional[list] = None,
    tool_choice: Optional[dict] = None,
    client: Optional[Any] = None,
    retries: int = 3,
    label: str = "Ollama",
    print_fn=print,
):
    """One chat completion with exponential-backoff retry.

    `prompt` may be a plain string (wrapped as a single user message) or a
    ready-made messages list. Returns the raw response object.
    """
    c = get_client(client)
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "messages": messages,
        # Ollama reads keep_alive from the request body; the OpenAI SDK forwards
        # unknown fields via extra_body. Harmless if the server ignores it.
        "extra_body": {"keep_alive": KEEP_ALIVE},
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    for attempt in range(retries):
        try:
            return c.chat.completions.create(**kwargs)
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 3
            print_fn(f"  {label} error — retrying in {wait}s (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(wait)


def chat_text(prompt: Union[str, List[dict]], **kwargs) -> str:
    """chat() convenience for plain-text prose responses (writer, summaries)."""
    response = chat(prompt, **kwargs)
    return (response.choices[0].message.content or "").strip()


def chat_json(
    prompt: Union[str, List[dict]],
    *,
    label: str = "response",
    model: str = MODEL,
    max_tokens: int = 2048,
    timeout: int = 600,
    client: Optional[Any] = None,
    retries: int = 3,
    use_response_format: bool = True,
    response_format_fallback: bool = False,
    print_fn=print,
):
    """chat() + JSON parsing in one call.

    use_response_format          — request response_format=json_object (most nodes).
    response_format_fallback     — if the json_object request errors (some models
                                   reject it), retry once as a plain call before
                                   giving up. Mirrors the old canon/craft fallback.
    Returns the parsed dict/list (via llm_json.parse_json_response).
    """
    from llm_json import parse_json_response

    rf = {"type": "json_object"} if use_response_format else None

    if rf is not None and response_format_fallback:
        try:
            response = chat(
                prompt, model=model, max_tokens=max_tokens, timeout=timeout,
                response_format=rf, client=client, retries=1, label=label, print_fn=print_fn,
            )
        except Exception:
            response = chat(
                prompt, model=model, max_tokens=max_tokens, timeout=timeout,
                client=client, retries=retries, label=label, print_fn=print_fn,
            )
    else:
        response = chat(
            prompt, model=model, max_tokens=max_tokens, timeout=timeout,
            response_format=rf, client=client, retries=retries, label=label, print_fn=print_fn,
        )

    raw = response.choices[0].message.content or ""
    return parse_json_response(raw, error_label=label)
