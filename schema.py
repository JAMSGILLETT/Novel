from __future__ import annotations
from typing import Optional, Literal, Union, Annotated, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import operator
import uuid


def gen_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Persistent domain models (mirror future SQLite tables)
# ---------------------------------------------------------------------------

class Character(BaseModel):
    id: str = Field(default_factory=gen_id)
    name: str
    personality: str
    goals: list[str] = Field(default_factory=list)
    relationships: dict[str, str] = Field(default_factory=dict)
    knowledge: list[str] = Field(default_factory=list)
    emotional_state: str = ""
    reputation: dict[str, str] = Field(default_factory=dict)
    current_location_id: Optional[str] = None
    current_objectives: list[str] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)
    is_alive: bool = True
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Plotline(BaseModel):
    id: str = Field(default_factory=gen_id)
    name: str
    status: Literal["dormant", "active", "resolved", "abandoned"] = "active"
    progress_stage: str
    current_tension: int = Field(ge=0, le=10)
    involved_character_ids: list[str] = Field(default_factory=list)
    next_possible_developments: list[str] = Field(default_factory=list)
    last_touched_chapter: int = 0  # chapter this plotline was last advanced by a patch; drives the stale-plotline audit
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Location(BaseModel):
    id: str = Field(default_factory=gen_id)
    name: str
    description: str
    political_control: Optional[str] = None
    npcs_present: list[str] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)
    recent_events: list[str] = Field(default_factory=list)
    tone: str = ""
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class WorldRule(BaseModel):
    """Hard fixed constraints that can never be violated: magic systems, physics, etc."""
    id: str = Field(default_factory=gen_id)
    rule_type: Literal["magic_system", "physics", "social_rule", "hard_constraint"]
    title: str
    content: str


class WorldLore(BaseModel):
    """Discoverable world knowledge: history, politics, culture, canon facts."""
    id: str = Field(default_factory=gen_id)
    category: Literal["history", "politics", "geography", "culture", "canon_fact"]
    title: str
    content: str


class POVState(BaseModel):
    location_id: str
    pov_character_id: Optional[str] = None  # always pulled mandatorily by context builder
    companions: list[str] = Field(default_factory=list)
    inventory: list[str] = Field(default_factory=list)
    emotional_state: str = ""
    injuries: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    knowledge: list[str] = Field(default_factory=list)


class ChapterSummary(BaseModel):
    chapter_number: int
    short_summary: str
    medium_summary: str
    timeline_events: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval config (per-entity-type vector thresholds, loaded alongside story config)
# ---------------------------------------------------------------------------

class RetrievalConfig(BaseModel):
    vector_hit_threshold: dict[
        Literal["character", "plotline", "location", "world_rule", "world_lore", "chapter_summary"], float
    ] = Field(default_factory=lambda: {
        "character": 0.75,
        "plotline": 0.75,
        "location": 0.70,
        "world_rule": 0.85,
        "world_lore": 0.75,
        "chapter_summary": 0.70,
    })
    max_results_per_type: dict[
        Literal["character", "plotline", "location", "world_rule", "world_lore", "chapter_summary"], int
    ] = Field(default_factory=lambda: {
        "character": 6,
        "plotline": 4,
        "location": 3,
        "world_rule": 4,
        "world_lore": 5,
        "chapter_summary": 1,
    })


# ---------------------------------------------------------------------------
# Retrieval / context pack
# ---------------------------------------------------------------------------

class DependencyGraphHit(BaseModel):
    rule_id: str
    reason: str
    content: str


class CharacterRosterEntry(BaseModel):
    """Lightweight record of every character the story has ever introduced.
    Always included in ContextPack so the Story Planner knows who exists,
    even if their full Character record wasn't pulled by retrieval."""
    id: str
    name: str
    is_alive: bool = True
    current_location_id: Optional[str] = None


class ContextPack(BaseModel):
    pov_state: Optional[POVState]  # None on cold start before POV is established
    character_roster: list[CharacterRosterEntry]  # all characters, always present
    active_characters: list[Character]      # full details, retrieved subset
    active_plotlines: list[Plotline]        # all active plotlines, always present
    nearby_locations: list[Location]
    relevant_world_rules: list[WorldRule]   # all world rules, always present
    relevant_world_lore: list[WorldLore]
    last_chapter_summary: Optional[ChapterSummary]  # always present if exists
    dependency_graph_hits: list[DependencyGraphHit] = Field(default_factory=list)
    vector_search_scores: dict[str, float] = Field(default_factory=dict)
    stale_plotlines: list[Plotline] = Field(default_factory=list)  # active but untouched for STALE_PLOTLINE_THRESHOLD+ chapters


# ---------------------------------------------------------------------------
# Planner / character reasoner / writer / canon check outputs
# ---------------------------------------------------------------------------

class CharacterConstraint(BaseModel):
    character_id: str = ""  # 8B models often omit this; reasoner overwrites it anyway
    forbidden_actions: list[str] = Field(default_factory=list)
    required_callbacks: list[str] = Field(default_factory=list)


def _coerce_str_list(v: Any) -> Any:
    """Small models sometimes return list fields as:
      - a single-key dict  {"items": [...]}  → extract the list
      - a JSON string      "['a', 'b']"      → parse it
    Both cases are normalised to a real Python list before Pydantic validates."""
    if isinstance(v, str):
        # Try strict JSON first, then json_repair for single-quoted / malformed strings
        import json as _json
        for candidate in [v, v.strip()]:
            try:
                parsed = _json.loads(candidate)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        try:
            from json_repair import repair_json
            parsed = _json.loads(repair_json(v))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        # Give up — return as-is and let Pydantic raise a clear error
        return v
    if isinstance(v, dict):
        lists = [val for val in v.values() if isinstance(val, list)]
        if lists:
            return lists[0]
        return list(v.values())
    return v


class StoryPlan(BaseModel):
    scenes: list[str]
    pacing_notes: str
    conflicts: list[str]
    narrative_goals: list[str]
    character_constraints: list[CharacterConstraint]
    required_callbacks: list[str] = Field(default_factory=list)
    target_word_count: int = Field(
        default=1000, ge=400, le=3000,
        description="Chapter length the writer should aim for, chosen based on pacing (short/punchy for tense "
                     "or climactic chapters, longer for slow-burn/world-building chapters).",
    )
    requested_offscreen_character_ids: list[str] = Field(
        default_factory=list,
        description=(
            "IDs from the character roster that are not currently in active_characters "
            "but should be brought into this chapter. Node 4 will fetch their full profiles."
        ),
    )

    @field_validator(
        "scenes", "conflicts", "narrative_goals", "required_callbacks",
        "requested_offscreen_character_ids", "character_constraints",
        mode="before",
    )
    @classmethod
    def _coerce_list_fields(cls, v: Any) -> Any:
        return _coerce_str_list(v)


class CharacterReasoning(BaseModel):
    character_id: str
    action_intentions: list[str]
    dialogue_intent: str
    emotional_response: str
    constraint_acknowledgement: list[str]


class CanonViolation(BaseModel):
    violation_type: Literal[
        "knowledge_leak", "location_inconsistency", "lore_violation", "forbidden_action_violated"
    ]
    description: str
    related_entity_id: Optional[str] = None
    severity: Literal["minor", "major"] = "major"


class CanonCheckResult(BaseModel):
    passed: bool
    violations: list[CanonViolation] = Field(default_factory=list)


class CraftIssue(BaseModel):
    issue_type: Literal["pacing", "tension", "show_dont_tell", "dialogue", "voice_consistency"]
    description: str
    severity: Literal["minor", "major"] = "major"


class CraftCheckResult(BaseModel):
    passed: bool
    issues: list[CraftIssue] = Field(default_factory=list)


class CombinedCheckResult(BaseModel):
    """Single-call canon + craft verdict. Used by node_combined_check when it
    asks the model (via Instructor) for a validated result in one shot — the
    model can't return a truncated JSON that silently reads as 'passed', because
    Instructor re-asks until the whole object validates against this schema."""
    canon_passed: bool
    violations: list[CanonViolation] = Field(default_factory=list)
    craft_passed: bool
    issues: list[CraftIssue] = Field(default_factory=list)

    def split(self) -> "tuple[CanonCheckResult, CraftCheckResult]":
        return (
            CanonCheckResult(passed=self.canon_passed, violations=self.violations),
            CraftCheckResult(passed=self.craft_passed, issues=self.issues),
        )


# ---------------------------------------------------------------------------
# Story outline: premise/theme/beats/character arcs, revisited periodically
# ---------------------------------------------------------------------------

class StoryBeat(BaseModel):
    id: str = Field(default_factory=gen_id)
    description: str
    status: Literal["upcoming", "in_progress", "completed"] = "upcoming"
    related_plotline_id: Optional[str] = None


class CharacterArcNote(BaseModel):
    character_id: str
    arc_summary: str
    current_stage: str = ""


class StoryOutline(BaseModel):
    story_id: str
    premise: str = ""
    theme: str = ""
    planned_ending: str = ""
    beats: list[StoryBeat] = Field(default_factory=list)
    character_arcs: list[CharacterArcNote] = Field(default_factory=list)
    last_revised_chapter: int = 0
    version: int = 1
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ActSummary(BaseModel):
    act_number: int
    chapter_start: int
    chapter_end: int
    summary: str
    key_events: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Memory patches: discriminated union on entity_type
# ---------------------------------------------------------------------------

class CharacterPatch(BaseModel):
    entity_type: Literal["character"] = "character"
    entity_id: Optional[str] = None
    name: Optional[str] = None
    personality: Optional[str] = None
    goals: Optional[list[str]] = None
    relationships: Optional[dict[str, str]] = None
    knowledge_added: Optional[list[str]] = None
    emotional_state: Optional[str] = None
    reputation: Optional[dict[str, str]] = None
    current_location_id: Optional[str] = None
    current_objectives: Optional[list[str]] = None
    secrets_added: Optional[list[str]] = None
    is_alive: Optional[bool] = None  # explicit, since life/death matters for reconciliation
    source: str = "memory_extractor"


class PlotlinePatch(BaseModel):
    entity_type: Literal["plotline"] = "plotline"
    entity_id: Optional[str] = None
    name: Optional[str] = None
    status: Optional[Literal["dormant", "active", "resolved", "abandoned"]] = None
    progress_stage: Optional[str] = None
    current_tension: Optional[int] = None
    involved_character_ids_added: Optional[list[str]] = None
    next_possible_developments: Optional[list[str]] = None
    implies_character_death: Optional[list[str]] = None  # character_ids, for reconciliation cross-check
    source: str = "memory_extractor"


class LocationPatch(BaseModel):
    entity_type: Literal["location"] = "location"
    entity_id: Optional[str] = None
    description: Optional[str] = None
    political_control: Optional[str] = None
    npcs_present: Optional[list[str]] = None
    secrets_added: Optional[list[str]] = None
    recent_events_added: Optional[list[str]] = None
    tone: Optional[str] = None
    source: str = "memory_extractor"


class WorldRulePatch(BaseModel):
    entity_type: Literal["world_rule"] = "world_rule"
    entity_id: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    source: str = "memory_extractor"


class WorldLorePatch(BaseModel):
    entity_type: Literal["world_lore"] = "world_lore"
    entity_id: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    source: str = "memory_extractor"


class POVPatch(BaseModel):
    entity_type: Literal["pov_state"] = "pov_state"
    pov_character_id: Optional[str] = None
    location_id: Optional[str] = None
    companions: Optional[list[str]] = None
    inventory_added: Optional[list[str]] = None
    inventory_removed: Optional[list[str]] = None
    emotional_state: Optional[str] = None
    injuries_added: Optional[list[str]] = None
    goals: Optional[list[str]] = None
    knowledge_added: Optional[list[str]] = None
    source: str = "memory_extractor"


MemoryPatch = Annotated[
    Union[CharacterPatch, PlotlinePatch, LocationPatch, WorldRulePatch, WorldLorePatch, POVPatch],
    Field(discriminator="entity_type"),
]


class WorldEntity(BaseModel):
    """Flexible world-building entity for organisations, cultures, species, etc."""
    id: str = Field(default_factory=gen_id)
    category: str  # e.g. "organization", "culture", "species"
    name: str
    description: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CanonRule(BaseModel):
    """A manually authored dependency-graph rule stored in SQLite."""
    rule_id: str
    story_id: str
    trigger_entity_type: str
    trigger_entity_id: str
    inject_entity_type: Literal["character", "plotline", "location", "world_rule", "world_lore"]
    inject_entity_id: str
    reason: str


class ReconciliationConflict(BaseModel):
    description: str
    conflicting_patches: list[MemoryPatch]
    resolution: Optional[str] = None
    resolved_by_rule: Optional[str] = None


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

MAX_CANON_CHECK_RETRIES = 2  # original attempt + this many revisions = 3 total
MAX_CRAFT_CHECK_RETRIES = 1  # craft is subjective — cap revisions tighter than canon


class ChapterGraphState(BaseModel):
    story_id: str
    chapter_number: int
    input_mode: Optional[Literal["cold_start", "continuation"]] = None
    user_input: str

    retrieval_config: RetrievalConfig = Field(default_factory=RetrievalConfig)

    context_pack: Optional[ContextPack] = None
    story_plan: Optional[StoryPlan] = None

    character_reasonings: Annotated[list[CharacterReasoning], operator.add] = Field(default_factory=list)

    chapter_prose: Optional[str] = None

    canon_check_result: Optional[CanonCheckResult] = None
    canon_check_attempts: int = 0
    flagged_for_review: bool = False
    flagged_violations: list[CanonViolation] = Field(default_factory=list)

    craft_check_result: Optional[CraftCheckResult] = None
    craft_check_attempts: int = 0
    flagged_for_craft_review: bool = False
    craft_issues: list[CraftIssue] = Field(default_factory=list)

    chapter_summary: Optional[ChapterSummary] = None

    memory_patches: Annotated[list[MemoryPatch], operator.add] = Field(default_factory=list)

    reconciliation_conflicts: list[ReconciliationConflict] = Field(default_factory=list)
    reconciled_patches: list[MemoryPatch] = Field(default_factory=list)

    # Rolling summary since the last act boundary, and permanent per-act summaries before it
    book_summary: Optional[str] = None
    act_summaries: list[str] = Field(default_factory=list)

    # Story-level outline: premise/theme/beats/character arcs, revisited periodically
    story_outline: Optional[StoryOutline] = None

    # New entities discovered in prose that don't yet exist in the DB
    new_characters: list[Character] = Field(default_factory=list)
    new_locations: list[Location] = Field(default_factory=list)
    new_plotlines: list[Plotline] = Field(default_factory=list)
    new_world_rules: list[WorldRule] = Field(default_factory=list)
    new_world_lore: list[WorldLore] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
