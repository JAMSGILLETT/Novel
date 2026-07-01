"""
Shared fixtures for the NovelGen test suite.

Every node factory in this codebase already accepts injected dependencies
(ollama_client, db_path, chroma_client) for exactly this purpose — these
fixtures just standardize how tests build those fakes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import db as db_module


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """A fresh, initialized SQLite DB for one test."""
    path = tmp_path / "test_story.db"
    db_module.init_db(path)
    return path


@pytest.fixture
def ephemeral_chroma():
    """In-memory Chroma client — no disk writes."""
    import vector_store as vs
    return vs.get_ephemeral_client()


class FakeOllamaClient:
    """Minimal stand-in for the OpenAI client pointed at Ollama.

    Construct with either a fixed `content` string, or a `responses` list
    consumed in order (one per call) for tests that need several distinct
    LLM calls to return different things.
    """

    def __init__(self, content: str | None = None, responses: list[str] | None = None):
        self._responses = list(responses) if responses is not None else None
        self._content = content
        self.calls: list[dict] = []
        self.chat = self  # so `.chat.completions.create(...)` resolves
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._responses is not None:
            content = self._responses.pop(0) if self._responses else "{}"
        else:
            content = self._content if self._content is not None else "{}"
        return _FakeResponse(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = None


@pytest.fixture
def fake_ollama_json():
    """Factory: fake_ollama_json({"passed": True}) -> FakeOllamaClient returning that JSON."""
    def _make(payload, responses=None):
        if responses is not None:
            return FakeOllamaClient(responses=[json.dumps(r) for r in responses])
        return FakeOllamaClient(content=json.dumps(payload))
    return _make
