"""
Quick smoke test for schema.py — no LangGraph, no LLM calls.
Just proves the Pydantic models validate, serialize, and that the
discriminated union resolves correctly.

Run: python test_schema.py
"""
from schema import (
    Character, Plotline, Location, WorldRule, WorldLore, POVState, ChapterSummary,
    CharacterRosterEntry, RetrievalConfig, ContextPack, DependencyGraphHit,
    StoryPlan, CharacterConstraint, CharacterReasoning,
    CanonViolation, CanonCheckResult,
    CharacterPatch, PlotlinePatch, LocationPatch,
    ChapterGraphState, MAX_CANON_CHECK_RETRIES,
)
from pydantic import TypeAdapter


def test_basic_models():
    alice = Character(name="Alice", personality="Cautious, loyal", goals=["Find her brother"])
    bob = Character(name="Bob", personality="Reckless")
    plot = Plotline(name="The Search for Marcus", progress_stage="rising_action", current_tension=6,
                     involved_character_ids=[alice.id])
    loc = Location(name="Riverside Market", description="A crowded trade hub")
    rule = WorldRule(rule_type="magic_system", title="Bloodbinding", content="Magic requires a blood price")
    lore = WorldLore(category="history", title="The Founding War", content="The city was built on a battlefield")
    pov = POVState(location_id=loc.id, companions=[bob.id], goals=["Survive the week"])
    summary = ChapterSummary(chapter_number=1, short_summary="Alice arrives at the market",
                              medium_summary="Alice arrives looking for clues about Marcus, meets Bob")
    print("Basic models OK:", alice.name, plot.name, loc.name, rule.title, lore.title, pov.location_id, summary.chapter_number)
    return alice, bob, plot, loc, rule, lore, pov, summary


def test_context_pack(alice, bob, plot, loc, rule, lore, pov, summary):
    pack = ContextPack(
        pov_state=pov,
        character_roster=[
            CharacterRosterEntry(id=alice.id, name=alice.name),
            CharacterRosterEntry(id=bob.id, name=bob.name),
        ],
        active_characters=[alice, bob],
        active_plotlines=[plot],
        nearby_locations=[loc],
        relevant_world_rules=[rule],
        relevant_world_lore=[lore],
        last_chapter_summary=summary,
        dependency_graph_hits=[
            DependencyGraphHit(rule_id="rel-001", reason="Alice+Bob secret history", content="Bob owes Alice a debt")
        ],
        vector_search_scores={alice.id: 0.91, loc.id: 0.77},
    )
    print("ContextPack OK, characters:", [c.name for c in pack.active_characters])
    return pack


def test_plan_and_reasoning(alice, bob, plot):
    plan = StoryPlan(
        scenes=["Alice questions a merchant about Marcus"],
        pacing_notes="Slow build, withhold the reveal",
        conflicts=["Bob wants to leave town before Alice is ready"],
        narrative_goals=["Establish stakes for the search"],
        character_constraints=[
            CharacterConstraint(character_id=alice.id, forbidden_actions=["reveal she knows Bob's debt"],
                                 required_callbacks=[plot.id]),
        ],
        required_callbacks=[plot.id],
    )
    reasoning = CharacterReasoning(
        character_id=alice.id,
        action_intentions=["press the merchant for details"],
        dialogue_intent="guarded, probing",
        emotional_response="anxious but composed",
        constraint_acknowledgement=["did not reveal Bob's debt"],
    )
    print("Plan + reasoning OK:", plan.scenes[0], "|", reasoning.dialogue_intent)
    return plan, reasoning


def test_canon_check():
    result_fail = CanonCheckResult(
        passed=False,
        violations=[CanonViolation(violation_type="forbidden_action_violated",
                                    description="Alice revealed Bob's debt despite constraint",
                                    severity="major")],
    )
    result_pass = CanonCheckResult(passed=True)
    print("CanonCheckResult OK:", result_fail.passed, result_pass.passed)
    return result_fail


def test_discriminated_union(alice, plot, loc):
    patches = [
        CharacterPatch(entity_id=alice.id, emotional_state="resolved", knowledge_added=["Marcus was seen at the docks"]),
        PlotlinePatch(entity_id=plot.id, current_tension=7, progress_stage="climax_approaching"),
        LocationPatch(entity_id=loc.id, recent_events_added=["A scuffle broke out near the stalls"]),
    ]
    from schema import MemoryPatch
    adapter = TypeAdapter(list[MemoryPatch])
    dumped = [p.model_dump() for p in patches]
    restored = adapter.validate_python(dumped)
    types = [type(p).__name__ for p in restored]
    assert types == ["CharacterPatch", "PlotlinePatch", "LocationPatch"], types
    print("Discriminated union OK:", types)
    return patches


def test_graph_state(alice, bob, plot, loc, rule, lore, pov, summary, pack, plan, reasoning, patches):
    state = ChapterGraphState(
        story_id="story-001",
        chapter_number=2,
        input_mode="continuation",
        user_input="Have Alice push harder for answers this chapter.",
        context_pack=pack,
        story_plan=plan,
        character_reasonings=[reasoning],
        memory_patches=patches,
    )
    assert state.canon_check_attempts == 0
    assert state.flagged_for_review is False
    assert len(state.character_reasonings) == 1
    assert len(state.memory_patches) == 3
    print("ChapterGraphState OK. MAX_CANON_CHECK_RETRIES =", MAX_CANON_CHECK_RETRIES)

    state.canon_check_attempts += 1
    while state.canon_check_attempts <= MAX_CANON_CHECK_RETRIES:
        state.canon_check_attempts += 1
    state.flagged_for_review = True
    state.flagged_violations = [CanonViolation(violation_type="lore_violation",
                                                description="Simulated unresolved violation after retries")]
    print("Retry/flag simulation OK. attempts:", state.canon_check_attempts,
          "flagged:", state.flagged_for_review)

    raw = state.model_dump_json()
    restored = ChapterGraphState.model_validate_json(raw)
    assert restored.story_id == state.story_id
    assert len(restored.memory_patches) == 3
    print("Full state JSON round-trip OK. Bytes:", len(raw))


if __name__ == "__main__":
    alice, bob, plot, loc, rule, lore, pov, summary = test_basic_models()
    pack = test_context_pack(alice, bob, plot, loc, rule, lore, pov, summary)
    plan, reasoning = test_plan_and_reasoning(alice, bob, plot)
    test_canon_check()
    patches = test_discriminated_union(alice, plot, loc)
    test_graph_state(alice, bob, plot, loc, rule, lore, pov, summary, pack, plan, reasoning, patches)
    print("\nAll smoke tests passed.")
