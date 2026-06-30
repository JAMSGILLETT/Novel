"""
Tests for Node 3: Story Planner.

No real API calls — FakeOpenRouterClient mimics the openai SDK's response
shape (choices[0].message.tool_calls[0].function.arguments).

Run: python test_story_planner.py
"""
from __future__ import annotations

import json

from schema import (
    Character, CharacterConstraint, CharacterRosterEntry, ChapterSummary,
    ContextPack, ChapterGraphState, Location, Plotline,
    POVState, StoryPlan, WorldLore, WorldRule,
)
from node_story_planner import build_planner_prompt, make_story_planner_node, MODEL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pack() -> ContextPack:
    rule = WorldRule(rule_type="magic_system", title="Bloodbinding",
                     content="Magic requires a blood price")
    lore = WorldLore(category="history", title="The Founding War",
                     content="City built on a battlefield")
    alice = Character(name="Alice", personality="Cautious",
                      goals=["Find Marcus"], emotional_state="anxious")
    bob = Character(name="Bob", personality="Reckless",
                    goals=["Leave town"], emotional_state="restless")
    mentor = Character(name="The Mentor", personality="Wise")
    plot = Plotline(name="The Search", progress_stage="rising_action",
                    current_tension=7, involved_character_ids=[alice.id],
                    next_possible_developments=["Alice finds a lead"],
                    status="active")
    loc = Location(name="Riverside Market", description="Crowded trade hub", tone="tense")
    pov = POVState(location_id=loc.id, companions=[bob.id], emotional_state="anxious")
    summary = ChapterSummary(chapter_number=1, short_summary="Ch1 short",
                              medium_summary="Alice arrived at the market and met Bob")
    return ContextPack(
        pov_state=pov,
        character_roster=[
            CharacterRosterEntry(id=alice.id, name=alice.name),
            CharacterRosterEntry(id=bob.id, name=bob.name),
            CharacterRosterEntry(id=mentor.id, name=mentor.name),
        ],
        active_characters=[alice, bob],
        active_plotlines=[plot],
        nearby_locations=[loc],
        relevant_world_rules=[rule],
        relevant_world_lore=[lore],
        last_chapter_summary=summary,
    )


def _make_state(user_input: str = "continue",
                input_mode: str = "continuation") -> ChapterGraphState:
    state = ChapterGraphState(
        story_id="story-001",
        chapter_number=2,
        input_mode=input_mode,
        user_input=user_input,
    )
    state.context_pack = _make_pack()
    return state


# ---------------------------------------------------------------------------
# Fake OpenRouter client (mimics openai SDK response shape)
# ---------------------------------------------------------------------------

FAKE_PLAN = {
    "scenes": [
        "Alice presses the merchant for information about Marcus",
        "Bob tries to pull Alice away, revealing he knows more than he lets on",
        "Alice discovers a torn piece of cloth that belonged to Marcus",
    ],
    "pacing_notes": "Slow build with rising tension; end on a cliffhanger revelation",
    "conflicts": [
        "Alice's urgency vs Bob's reluctance to stay",
        "The merchant's fear of someone watching him",
    ],
    "narrative_goals": [
        "Advance the search for Marcus by one concrete lead",
        "Deepen tension between Alice and Bob",
    ],
    "character_constraints": [
        {
            "character_id": "placeholder-id",
            "forbidden_actions": ["Reveal she knows about Bob's debt"],
            "required_callbacks": ["The Search plotline must progress"],
        }
    ],
    "required_callbacks": ["A new clue about Marcus must surface"],
    "requested_offscreen_character_ids": [],
}


class _FakeFunction:
    arguments = json.dumps(FAKE_PLAN)


class _FakeToolCall:
    function = _FakeFunction()


class _FakeMessage:
    tool_calls = [_FakeToolCall()]


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


class FakeOpenRouterClient:
    """Records the last call kwargs so tests can inspect what was sent."""

    def __init__(self):
        self.last_call: dict = {}
        self.chat = _FakeChat(self)


class _FakeCompletions:
    def __init__(self, recorder):
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder.last_call = kwargs
        return _FakeResponse()


class _FakeChat:
    def __init__(self, recorder):
        self.completions = _FakeCompletions(recorder)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_prompt_contains_key_sections():
    state = _make_state()
    prompt = build_planner_prompt(state)

    assert "Bloodbinding" in prompt, "World rule missing"
    assert "The Search" in prompt, "Active plotline missing"
    assert "Alice" in prompt, "Character missing"
    assert "The Mentor" in prompt, "Off-screen character missing from roster"
    assert "requested_offscreen_character_ids" in prompt
    assert "chapter 2" in prompt.lower()
    print("test_prompt_contains_key_sections OK")


def test_prompt_cold_start_mode():
    state = _make_state(user_input="Begin the story", input_mode="cold_start")
    state.context_pack.last_chapter_summary = None
    state.context_pack.pov_state = None
    prompt = build_planner_prompt(state)

    assert "COLD START" in prompt
    assert "first chapter" in prompt.lower()
    print("test_prompt_cold_start_mode OK")


def test_prompt_user_event_injection():
    state = _make_state(
        user_input="Have a stranger attack Alice at the market",
        input_mode="user_event_injection",
    )
    prompt = build_planner_prompt(state)

    assert "USER EVENT INJECTION" in prompt
    assert "Have a stranger attack Alice" in prompt
    print("test_prompt_user_event_injection OK")


def test_node_produces_valid_story_plan():
    client = FakeOpenRouterClient()
    node = make_story_planner_node(openrouter_client=client)
    state = _make_state(user_input="Push Alice harder for answers")

    result = node(state)
    plan = result["story_plan"]

    assert isinstance(plan, StoryPlan)
    assert len(plan.scenes) == 3
    assert "Slow build" in plan.pacing_notes
    assert len(plan.conflicts) == 2
    assert len(plan.narrative_goals) == 2
    assert len(plan.character_constraints) == 1
    assert plan.requested_offscreen_character_ids == []
    print("test_node_produces_valid_story_plan OK — scenes:", plan.scenes[0][:50])


def test_node_sends_correct_openai_tool_schema():
    """Node uses OpenAI function-calling format and forces the story_plan tool."""
    client = FakeOpenRouterClient()
    node = make_story_planner_node(openrouter_client=client)
    node(_make_state())

    call = client.last_call
    assert call["model"] == MODEL
    assert call["tool_choice"] == {"type": "function", "function": {"name": "story_plan"}}
    assert call["tools"][0]["type"] == "function"
    assert call["tools"][0]["function"]["name"] == "story_plan"
    schema_str = str(call["tools"][0]["function"]["parameters"])
    assert "scenes" in schema_str
    assert "character_constraints" in schema_str
    assert "requested_offscreen_character_ids" in schema_str
    print("test_node_sends_correct_openai_tool_schema OK")


def test_requested_offscreen_characters_round_trip():
    mentor_id = "mentor-123"
    plan = StoryPlan(
        scenes=["The Mentor appears at the docks"],
        pacing_notes="Surprise reunion",
        conflicts=["Alice distrusts the Mentor"],
        narrative_goals=["New lead via Mentor"],
        character_constraints=[],
        required_callbacks=[],
        requested_offscreen_character_ids=[mentor_id],
    )
    restored = StoryPlan.model_validate_json(plan.model_dump_json())
    assert restored.requested_offscreen_character_ids == [mentor_id]
    print("test_requested_offscreen_characters_round_trip OK —", restored.requested_offscreen_character_ids)


if __name__ == "__main__":
    test_prompt_contains_key_sections()
    test_prompt_cold_start_mode()
    test_prompt_user_event_injection()
    test_node_produces_valid_story_plan()
    test_node_sends_correct_openai_tool_schema()
    test_requested_offscreen_characters_round_trip()
    print("\nAll story planner tests passed.")
