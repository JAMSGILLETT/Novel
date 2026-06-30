from __future__ import annotations
from typing import Optional, Literal, Union, Annotated
from pydantic import BaseModel, Field
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
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Plotline(BaseModel):
    id: str = Field(default_factory=gen_id)
    name: str
    status: Literal["dormant", "active", "resolved", "abandoned"] = "active"
    progress_stage: str
    current_tension: int = Field(ge=0, le=10)
    involved_character_ids: list[str] = Field(default_factory=list)
    next_possible_developments: list[str] = Field(default_factory=list)
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


class LoreEntry(BaseModel):
    id: str = Field(default_factory=gen_id)
    category: Literal["magic_system", "politics", "physics", "social_rules", "canon_fact"]
    title: str
    content: str
    is_fixed_constraint: bool = True


class POVState(BaseModel):
    location_id: str
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
        Literal["character", "plotline", "location", "lore", "chapter_summary"], float
    ] = Field(default_factory=lambda: {
        "character": 0.75,
        "plotline": 0.75,
        "location": 0.70,
        "lore": 0.80,
        "chapter_summary": 0.70,
    })
    max_results_per_type: dict[
        Literal["character", "plotline", "location", "lore", "chapter_summary"], int
    ] = Field(default_factory=lambda: {
        "character": 6,
        "plotline": 4,
        "location": 3,
        "lore": 5,
        "chapter_summary": 1,
    })


# ---------------------------------------------------------------------------
# Retrieval / context pack
# ---------------------------------------------------------------------------

class DependencyGraphHit(BaseModel):
    rule_id: str
    reason: str
    content: str


class ContextPack(BaseModel):
    pov_state: POVState
    active_characters: list[Character]
    active_plotlines: list[Plotline]
    nearby_locations: list[Location]
    relevant_lore: list[LoreEntry]
    last_chapter_summary: Optional[ChapterSummary]
    dependency_graph_hits: list[DependencyGraphHit] = Field(default_factory=list)
    vector_search_scores: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Planner / character reasoner / writer / canon check outputs
# ---------------------------------------------------------------------------

class CharacterConstraint(BaseModel):
    character_id: str
    forbidden_actions: list[str]
    required_callbacks: list[str]


class StoryPlan(BaseModel):
    scenes: list[str]
    pacing_notes: str
    conflicts: list[str]
    narrative_goals: list[str]
    character_constraints: list[CharacterConstraint]
    required_callbacks: list[str]


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


class LorePatch(BaseModel):
    entity_type: Literal["lore"] = "lore"
    entity_id: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    source: str = "memory_extractor"


class POVPatch(BaseModel):
    entity_type: Literal["pov_state"] = "pov_state"
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
    Union[CharacterPatch, PlotlinePatch, LocationPatch, LorePatch, POVPatch],
    Field(discriminator="entity_type"),
]


class ReconciliationConflict(BaseModel):
    description: str
    conflicting_patches: list[MemoryPatch]
    resolution: Optional[str] = None
    resolved_by_rule: Optional[str] = None


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

MAX_CANON_CHECK_RETRIES = 2  # original attempt + this many revisions = 3 total


class ChapterGraphState(BaseModel):
    story_id: str
    chapter_number: int
    input_mode: Optional[Literal["cold_start", "continuation", "user_event_injection"]] = None
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

    chapter_summary: Optional[ChapterSummary] = None

    memory_patches: Annotated[list[MemoryPatch], operator.add] = Field(default_factory=list)

    reconciliation_conflicts: list[ReconciliationConflict] = Field(default_factory=list)
    reconciled_patches: list[MemoryPatch] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
