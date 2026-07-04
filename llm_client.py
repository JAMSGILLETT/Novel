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

# The active model, resolved at call time (not import time) so the GUI Settings
# tab can change it live. Seeded from NOVELGEN_MODEL, then get_model()/set_model()
# own it. `MODEL` remains as a back-compat alias of the initial value.
_current_model = os.environ.get("NOVELGEN_MODEL", DEFAULT_MODEL)
MODEL = _current_model


def get_model() -> str:
    """The model every node uses unless one is passed explicitly."""
    return _current_model


def set_model(name: str) -> None:
    """Switch the active model for all subsequent LLM calls. Empty → default."""
    global _current_model
    _current_model = (name or "").strip() or DEFAULT_MODEL


def list_installed_models(client: Optional[Any] = None) -> List[str]:
    """Model ids Ollama currently has pulled (via the OpenAI-compatible
    /models endpoint). Raises if Ollama isn't reachable."""
    resp = get_client(client).models.list()
    return sorted(m.id for m in resp.data)


# How long Ollama should keep the model loaded after a request. Keeping it warm
# across a chapter's calls avoids repeated multi-second weight reloads.
# Override with NOVELGEN_KEEP_ALIVE (e.g. "0" to unload immediately, "-1" forever).
KEEP_ALIVE = os.environ.get("NOVELGEN_KEEP_ALIVE", "30m")

_shared_client = None

# Optional live-token sink for "typewriter" output. When the GUI registers one,
# prose calls made with stream=True emit each token to it as it arrives; when it
# is None (default), those calls behave exactly like normal blocking calls.
_stream_sink: Optional[Any] = None


def set_stream_sink(sink: Optional[Any]) -> None:
    """Register (or clear, with None) where streamed tokens go. The sink is a
    callable taking one string; it must be cheap and thread-safe (the GUI's
    just enqueues onto its output queue)."""
    global _stream_sink
    _stream_sink = sink


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
    model: Optional[str] = None,
    max_tokens: int = 2048,
    timeout: int = 600,
    response_format: Optional[dict] = None,
    tools: Optional[list] = None,
    tool_choice: Optional[dict] = None,
    temperature: Optional[float] = None,
    client: Optional[Any] = None,
    retries: int = 3,
    label: str = "Ollama",
    print_fn=print,
):
    """One chat completion with exponential-backoff retry.

    `prompt` may be a plain string (wrapped as a single user message) or a
    ready-made messages list. `model=None` uses the active model (get_model()),
    so the GUI can switch models live. Returns the raw response object.
    """
    c = get_client(client)
    model = model or get_model()
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
    if temperature is not None:
        kwargs["temperature"] = temperature
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


_instructor_client = None


def get_instructor_client(client: Optional[Any] = None):
    """OpenAI client patched by Instructor for validated structured output.

    Uses JSON mode (not tool-calling) because Ollama's tool support is uneven
    across small models, whereas json_object is broadly honored. The underlying
    client is the same shared singleton chat() uses, so keep_alive/base_url are
    unchanged. Tests can still inject a fake via `client`."""
    import instructor

    if client is not None:
        return instructor.from_openai(client, mode=instructor.Mode.JSON)
    global _instructor_client
    if _instructor_client is None:
        _instructor_client = instructor.from_openai(get_client(), mode=instructor.Mode.JSON)
    return _instructor_client


def chat_structured(
    prompt: Union[str, List[dict]],
    response_model,
    *,
    model: Optional[str] = None,
    max_tokens: int = 2048,
    timeout: int = 600,
    client: Optional[Any] = None,
    max_retries: int = 3,
    label: str = "Structured",
    print_fn=print,
):
    """chat() for a validated Pydantic model. Returns an instance of
    `response_model`, re-asking the model up to `max_retries` times if its
    output doesn't validate (truncated JSON, wrong types, missing fields).

    Unlike chat_json(), an incomplete response can't slip through as a default
    'passed' object — validation failure surfaces as an exception the caller
    (or Instructor's own retry) handles, instead of a silent fallback."""
    ic = get_instructor_client(client)
    model = model or get_model()
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt

    return ic.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
        messages=messages,
        response_model=response_model,
        max_retries=max_retries,
        extra_body={"keep_alive": KEEP_ALIVE},
    )


def chat_text(prompt: Union[str, List[dict]], *, stream: bool = False, **kwargs) -> str:
    """chat() convenience for plain-text prose responses (writer, summaries).

    Pass stream=True on prose calls you'd like shown token-by-token. It only
    actually streams when a sink is registered (set_stream_sink); otherwise it
    falls back to a normal blocking call. Returns the full accumulated text
    either way, so callers are unaffected.
    """
    if stream and _stream_sink is not None:
        return _chat_text_streaming(prompt, sink=_stream_sink, **kwargs)
    response = chat(prompt, **kwargs)
    return (response.choices[0].message.content or "").strip()


def _chat_text_streaming(
    prompt: Union[str, List[dict]], *, sink, model: Optional[str] = None,
    max_tokens: int = 2048, timeout: int = 600, temperature: Optional[float] = None,
    client: Optional[Any] = None, retries: int = 3, label: str = "Ollama", print_fn=print,
) -> str:
    """Blocking call with stream=True: forward each token to `sink` as it lands
    and return the full accumulated text. Same retry/keep-alive shape as chat()."""
    c = get_client(client)
    model = model or get_model()
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    kwargs: dict = {
        "model": model, "max_tokens": max_tokens, "timeout": timeout,
        "messages": messages, "extra_body": {"keep_alive": KEEP_ALIVE}, "stream": True,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    for attempt in range(retries):
        try:
            parts: List[str] = []
            for chunk in c.chat.completions.create(**kwargs):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    parts.append(delta)
                    try:
                        sink(delta)
                    except Exception:
                        pass  # never let a display hiccup break generation
            return "".join(parts).strip()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 3
            print_fn(f"  {label} error — retrying in {wait}s (attempt {attempt + 1}/{retries}): {e}")
            try:
                sink("\n  [stream interrupted — retrying]\n")
            except Exception:
                pass
            time.sleep(wait)


def chat_json(
    prompt: Union[str, List[dict]],
    *,
    label: str = "response",
    model: Optional[str] = None,
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
