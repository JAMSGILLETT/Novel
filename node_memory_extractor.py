"""
Node 8: Memory Extractor.

Reads the final approved prose and chapter summary, then emits a list of
MemoryPatch objects — structured diffs describing what changed this chapter.

One LLM call per entity type (character, plotline, location, pov_state).
World rule and world lore patches are excluded from LLM extraction — those are
authored manually and should not be auto-mutated by prose.

Each call asks the model: "given what the prose shows, what fields changed?"
Only non-None fields in the returned patch mean "this changed". Fields left
as null/None mean "no change — leave the DB value alone."

The patches are accumulated on state.memory_patches (operator.add reduced list)
and passed to Node 9 (Reconciliation) which resolves conflicts before Node 10
writes anything to the DB.
"""
from __future__ import annotations

from typing import Callable, Optional

from llm_client import chat_text
from llm_json import extract_json_block, parse_json_response
from schema import (
    Character, ChapterGraphState, ChapterSummary, ContextPack,
    CharacterPatch, PlotlinePatch, LocationPatch, POVPatch,
    Plotline, Location, POVState, WorldRule, WorldLore,
)


# ---------------------------------------------------------------------------
# Shared JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict | list:
    return parse_json_response(raw, error_label="Memory extractor")


def _unwrap_envelope(data: dict | list) -> dict | list:
    if isinstance(data, dict) and "name" in data:
        for key in ("parameters", "arguments", "input"):
            if key in data and isinstance(data[key], (dict, list)):
                return data[key]
    return data


def _call_llm(model: str, ollama_client, prompt: str, max_tokens: int = 1024) -> str:
    return chat_text(
        prompt, model=model, max_tokens=max_tokens, timeout=300,
        client=ollama_client, label="Memory extractor",
    )


# ---------------------------------------------------------------------------
# Per-entity-type prompt builders and parsers
# ---------------------------------------------------------------------------

def _character_prompt(c: Character, prose: str, summary: ChapterSummary, template: Optional[str] = None) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["memory_character"]
    body = tpl.format(
        character_name=c.name,
        character_id=c.id,
        personality=c.personality,
        emotional_state=c.emotional_state or "unknown",
        goals="; ".join(c.goals) if c.goals else "none",
        knowledge="; ".join(c.knowledge) if c.knowledge else "none",
        objectives="; ".join(c.current_objectives) if c.current_objectives else "none",
        chapter_summary=summary.medium_summary,
        chapter_prose=prose,
    )
    return body + f"""

OUTPUT: JSON object with ONLY fields that changed. Use null for unchanged fields.
{{
  "entity_type": "character",
  "entity_id": "{c.id}",
  "emotional_state": "new state if it changed, else null",
  "knowledge_added": ["new thing they learned"] or null,
  "current_objectives": ["updated objective list"] or null,
  "goals": ["updated goal list if goals shifted"] or null,
  "is_alive": true or false or null,
  "current_location_id": "location_id if they moved, else null",
  "secrets_added": ["new secret revealed about them"] or null
}}
Respond with ONLY the JSON. No explanation."""


def _plotline_prompt(p: Plotline, prose: str, summary: ChapterSummary, template: Optional[str] = None) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["memory_plotline"]
    devs = "; ".join(p.next_possible_developments) if p.next_possible_developments else "none"
    body = tpl.format(
        plotline_name=p.name,
        plotline_id=p.id,
        status=p.status,
        progress_stage=p.progress_stage,
        current_tension=p.current_tension,
        next_developments=devs,
        chapter_summary=summary.medium_summary,
        chapter_prose=prose,
    )
    return body + f"""

OUTPUT: JSON object with ONLY fields that changed. Use null for unchanged fields.
{{
  "entity_type": "plotline",
  "entity_id": "{p.id}",
  "status": "dormant|active|resolved|abandoned or null",
  "progress_stage": "new stage description or null",
  "current_tension": 0-10 integer or null,
  "next_possible_developments": ["updated list"] or null,
  "involved_character_ids_added": ["char_id"] or null
}}
Respond with ONLY the JSON. No explanation."""


def _location_prompt(loc: Location, prose: str, summary: ChapterSummary, template: Optional[str] = None) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["memory_location"]
    body = tpl.format(
        location_name=loc.name,
        location_id=loc.id,
        description=loc.description,
        tone=loc.tone or "none",
        recent_events="; ".join(loc.recent_events) if loc.recent_events else "none",
        chapter_summary=summary.medium_summary,
        chapter_prose=prose,
    )
    return body + f"""

OUTPUT: JSON object with ONLY fields that changed. Use null for unchanged fields.
{{
  "entity_type": "location",
  "entity_id": "{loc.id}",
  "tone": "new tone if it shifted, else null",
  "recent_events_added": ["new event that happened here"] or null,
  "secrets_added": ["new secret uncovered about this place"] or null,
  "political_control": "new controller if changed, else null"
}}
Respond with ONLY the JSON. No explanation."""


def _pov_prompt(
    pov: POVState, pack: ContextPack, prose: str, summary: ChapterSummary, template: Optional[str] = None,
) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["memory_pov"]

    companions = ", ".join(pov.companions) if pov.companions else "none"
    inventory = ", ".join(pov.inventory) if pov.inventory else "none"
    injuries = ", ".join(pov.injuries) if pov.injuries else "none"
    knowledge = "; ".join(pov.knowledge) if pov.knowledge else "none"

    # Resolve location name for context
    loc_name = pov.location_id
    for l in pack.nearby_locations:
        if l.id == pov.location_id:
            loc_name = f"{l.name} ({l.id})"
            break

    # List available location ids for the model to reference
    loc_options = "\n".join(
        f"  {l.id} = {l.name}" for l in pack.nearby_locations
    ) or "  (none known)"

    body = tpl.format(
        location_name=loc_name,
        companions=companions,
        inventory=inventory,
        emotional_state=pov.emotional_state or "unknown",
        injuries=injuries,
        knowledge=knowledge,
        location_options=loc_options,
        chapter_summary=summary.medium_summary,
        chapter_prose=prose,
    )
    return body + """

OUTPUT: JSON object with ONLY fields that changed. Use null for unchanged fields.
{
  "entity_type": "pov_state",
  "pov_character_id": "character_id if perspective switched, else null",
  "location_id": "new_location_id if POV character moved, else null",
  "companions": ["updated companion id list"] or null,
  "emotional_state": "new emotional state or null",
  "inventory_added": ["items gained"] or null,
  "inventory_removed": ["items lost or used"] or null,
  "injuries_added": ["new injuries"] or null,
  "knowledge_added": ["new things the POV character learned"] or null,
  "goals": ["updated goals if they shifted"] or null
}
Respond with ONLY the JSON. No explanation."""


# ---------------------------------------------------------------------------
# Batch patch prompts — 1 LLM call for all entities of a given type
# ---------------------------------------------------------------------------

def _batch_character_prompt(characters: list, prose: str, summary: ChapterSummary) -> str:
    blocks = []
    for c in characters:
        blocks.append(
            f"- ID: {c.id}\n"
            f"  Name: {c.name}\n"
            f"  Personality: {c.personality}\n"
            f"  Emotional state: {c.emotional_state or 'unknown'}\n"
            f"  Goals: {'; '.join(c.goals) if c.goals else 'none'}\n"
            f"  Objectives: {'; '.join(c.current_objectives) if c.current_objectives else 'none'}\n"
            f"  Knowledge: {'; '.join(c.knowledge) if c.knowledge else 'none'}"
        )
    char_list = "\n".join(blocks)
    ids = ", ".join(f'"{c.id}"' for c in characters)
    return f"""You are tracking character state changes in a novel chapter.

CHAPTER SUMMARY: {summary.medium_summary}

CHAPTER PROSE (excerpt):
{prose[:3000]}

CHARACTERS TO ANALYZE:
{char_list}

For EACH character, output a patch with ONLY fields that changed. Use null for unchanged fields.
Return a JSON array with exactly one object per character, in the same order listed above:

[
  {{
    "entity_type": "character",
    "entity_id": {ids.split(",")[0].strip()},
    "emotional_state": "new state if changed, else null",
    "knowledge_added": ["new thing learned"] or null,
    "current_objectives": ["updated list"] or null,
    "goals": ["updated goals"] or null,
    "is_alive": true or false or null,
    "current_location_id": "location_id if moved, else null",
    "secrets_added": ["new secret"] or null
  }}
]

Respond with ONLY the JSON array. No explanation."""


def _batch_plotline_prompt(plotlines: list, prose: str, summary: ChapterSummary) -> str:
    blocks = []
    for p in plotlines:
        devs = "; ".join(p.next_possible_developments) if p.next_possible_developments else "none"
        blocks.append(
            f"- ID: {p.id}\n"
            f"  Name: {p.name}\n"
            f"  Status: {p.status}\n"
            f"  Stage: {p.progress_stage}\n"
            f"  Tension: {p.current_tension}/10\n"
            f"  Next developments: {devs}"
        )
    plot_list = "\n".join(blocks)
    return f"""You are tracking plotline state changes in a novel chapter.

CHAPTER SUMMARY: {summary.medium_summary}

CHAPTER PROSE (excerpt):
{prose[:3000]}

PLOTLINES TO ANALYZE:
{plot_list}

For EACH plotline, output a patch with ONLY fields that changed. Use null for unchanged fields.
Return a JSON array with exactly one object per plotline, in the same order listed above:

[
  {{
    "entity_type": "plotline",
    "entity_id": "plotline_id_here",
    "status": "dormant|active|resolved|abandoned or null",
    "progress_stage": "new stage description or null",
    "current_tension": 0-10 integer or null,
    "next_possible_developments": ["updated list"] or null,
    "involved_character_ids_added": ["char_id"] or null
  }}
]

Respond with ONLY the JSON array. No explanation."""


def _batch_location_prompt(locations: list, prose: str, summary: ChapterSummary) -> str:
    blocks = []
    for loc in locations:
        blocks.append(
            f"- ID: {loc.id}\n"
            f"  Name: {loc.name}\n"
            f"  Description: {loc.description}\n"
            f"  Tone: {loc.tone or 'none'}\n"
            f"  Recent events: {'; '.join(loc.recent_events) if loc.recent_events else 'none'}"
        )
    loc_list = "\n".join(blocks)
    return f"""You are tracking location state changes in a novel chapter.

CHAPTER SUMMARY: {summary.medium_summary}

CHAPTER PROSE (excerpt):
{prose[:3000]}

LOCATIONS TO ANALYZE:
{loc_list}

For EACH location, output a patch with ONLY fields that changed. Use null for unchanged fields.
Return a JSON array with exactly one object per location, in the same order listed above:

[
  {{
    "entity_type": "location",
    "entity_id": "location_id_here",
    "tone": "new tone if shifted, else null",
    "recent_events_added": ["new event that happened here"] or null,
    "secrets_added": ["new secret uncovered"] or null,
    "political_control": "new controller if changed, else null"
  }}
]

Respond with ONLY the JSON array. No explanation."""


def _parse_batch_patches(raw: str, entities: list, patch_class, id_attr: str = "id") -> list:
    """Parse a JSON array of patches; zip with entities to ensure correct entity_id."""
    if not raw.strip():
        return []
    # Strip think blocks
    if "<think>" in raw:
        parts = raw.split("</think>", 1)
        raw = parts[1].strip() if len(parts) > 1 else raw
    try:
        data = _parse_json(raw)
    except Exception:
        return []
    if not isinstance(data, list):
        # Might be wrapped in a key
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
        if not isinstance(data, list):
            return []

    patches = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        entity = entities[i] if i < len(entities) else None
        item = _unwrap_envelope(item)
        if entity is not None:
            item["entity_id"] = getattr(entity, id_attr)
        try:
            patch = patch_class.model_validate(item)
            patches.append(patch)
        except Exception:
            pass
    return patches


# ---------------------------------------------------------------------------
# New entity discovery — prompts and parsers
# ---------------------------------------------------------------------------

def _new_entities_prompt(
    prose: str,
    existing_char_names: list[str],
    existing_loc_names: list[str],
    existing_plot_names: list[str],
    existing_rule_titles: list[str],
    existing_lore_titles: list[str],
) -> str:
    """Single combined prompt for all 5 new-entity discovery types."""
    def _roster(names):
        return ", ".join(names) if names else "(none yet)"

    return f"""You are reading a chapter of a novel. Identify any NEW story elements introduced for the first time.

CHAPTER PROSE:
{prose[:3000]}

EXISTING (DO NOT re-list these):
  Characters: {_roster(existing_char_names)}
  Locations: {_roster(existing_loc_names)}
  Plotlines: {_roster(existing_plot_names)}
  World rules: {_roster(existing_rule_titles)}
  World lore: {_roster(existing_lore_titles)}

Only include things that genuinely appear in this chapter and are NOT already listed above.
If nothing new exists for a category, return an empty array for it.

Respond with ONLY this JSON object:
{{
  "new_characters": [
    {{"name": "Name", "personality": "brief", "goals": [], "emotional_state": "", "relationships": {{}}, "knowledge": [], "secrets": []}}
  ],
  "new_locations": [
    {{"name": "Name", "description": "brief", "tone": "", "political_control": null, "secrets": [], "recent_events": []}}
  ],
  "new_plotlines": [
    {{"name": "Name", "status": "active", "progress_stage": "brief", "current_tension": 5, "next_possible_developments": []}}
  ],
  "new_world_rules": [
    {{"rule_type": "hard_constraint", "title": "Title", "content": "full description"}}
  ],
  "new_world_lore": [
    {{"category": "canon_fact", "title": "Title", "content": "full description"}}
  ]
}}"""


def _new_characters_prompt(prose: str, existing_names: list[str], template: Optional[str] = None) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["memory_new_characters"]
    roster = ", ".join(existing_names) if existing_names else "(none yet)"
    body = tpl.format(existing_names=roster, chapter_prose=prose)
    return body + """
Each entry:
{
  "name": "Character Name",
  "personality": "brief personality description from how they appear in the prose",
  "goals": ["inferred goal from prose"],
  "emotional_state": "their state in this chapter",
  "relationships": {"other character name": "relationship type"},
  "knowledge": ["what they demonstrably know"],
  "secrets": []
}
Return ONLY the JSON array."""


def _new_locations_prompt(prose: str, existing_names: list[str], template: Optional[str] = None) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["memory_new_locations"]
    roster = ", ".join(existing_names) if existing_names else "(none yet)"
    body = tpl.format(existing_names=roster, chapter_prose=prose)
    return body + """
Each entry:
{
  "name": "Location Name",
  "description": "description based on how it appears in the prose",
  "tone": "atmosphere or tone if apparent",
  "political_control": null,
  "secrets": [],
  "recent_events": []
}
Return ONLY the JSON array."""


def _new_plotlines_prompt(prose: str, existing_names: list[str], template: Optional[str] = None) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["memory_new_plotlines"]
    roster = ", ".join(existing_names) if existing_names else "(none yet)"
    body = tpl.format(existing_names=roster, chapter_prose=prose)
    return body + """
Each entry:
{
  "name": "Plotline Name",
  "status": "active",
  "progress_stage": "brief description of where this thread stands",
  "current_tension": 5,
  "next_possible_developments": ["what could happen next"]
}
Return ONLY the JSON array."""


def _new_world_rules_prompt(prose: str, existing_titles: list[str], template: Optional[str] = None) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["memory_new_world_rules"]
    roster = ", ".join(existing_titles) if existing_titles else "(none yet)"
    body = tpl.format(existing_names=roster, chapter_prose=prose)
    return body + """
Each entry:
{
  "rule_type": "magic_system | physics | social_rule | hard_constraint",
  "title": "Short Rule Title",
  "content": "Full description of the rule and what it forbids or requires."
}
Return ONLY the JSON array."""


def _new_world_lore_prompt(prose: str, existing_titles: list[str], template: Optional[str] = None) -> str:
    from prompt_templates import DEFAULT_TEMPLATES
    tpl = template if template is not None else DEFAULT_TEMPLATES["memory_new_world_lore"]
    roster = ", ".join(existing_titles) if existing_titles else "(none yet)"
    body = tpl.format(existing_names=roster, chapter_prose=prose)
    return body + """
Each entry:
{
  "category": "history | politics | geography | culture | canon_fact",
  "title": "Short Lore Title",
  "content": "Full description of this lore entry."
}
Return ONLY the JSON array."""


def _parse_new_entities(raw: str) -> list[dict]:
    """Parse a JSON array response, returning [] on any failure."""
    if not raw.strip():
        return []
    if "<think>" in raw:
        parts = raw.split("</think>", 1)
        raw = parts[1].strip() if len(parts) > 1 else raw

    # Find array (not just object)
    start = raw.find("[")
    if start == -1:
        return []
    # Try to find matching close bracket
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(raw[start:], start):
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
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                json_str = raw[start:i + 1]
                break
    else:
        json_str = raw[start:]

    try:
        from json_repair import repair_json
        import json as _j
        result = _j.loads(repair_json(json_str))
    except Exception:
        try:
            import json as _j
            result = _j.loads(json_str)
        except Exception:
            return []

    return result if isinstance(result, list) else []


# ---------------------------------------------------------------------------
# Per-type patch parsers
# ---------------------------------------------------------------------------

def _to_patch(data: dict | list, patch_class):
    if isinstance(data, list):
        # Model returned a list — take first dict element
        data = next((x for x in data if isinstance(x, dict)), {})
    data = _unwrap_envelope(data)
    try:
        return patch_class.model_validate(data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_memory_extractor_node(
    model: Optional[str] = None,
    ollama_client=None,
    db_path=None,
    print_fn=print,
) -> Callable[[ChapterGraphState], dict]:
    _p = print_fn

    def node(state: ChapterGraphState) -> dict:
        if not state.chapter_prose:
            raise ValueError("chapter_prose must be set before memory extractor runs")
        if not state.chapter_summary:
            raise ValueError("chapter_summary must be set before memory extractor runs")

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from prompt_templates import get_template
        import db as _db_module

        story_id = state.story_id
        tpl_character = get_template("memory_character", story_id, db_path)
        tpl_plotline = get_template("memory_plotline", story_id, db_path)
        tpl_location = get_template("memory_location", story_id, db_path)
        tpl_pov = get_template("memory_pov", story_id, db_path)

        pack = state.context_pack
        prose = state.chapter_prose
        summary = state.chapter_summary

        def _has_changes(patch, exclude=("entity_type", "entity_id", "source")):
            return any(v is not None for f, v in patch.model_dump().items() if f not in exclude)

        def _count_changes(patch, exclude=("entity_type", "entity_id", "source")):
            return sum(1 for f, v in patch.model_dump().items() if f not in exclude and v is not None)

        # ── Build batch tasks (1 call per entity type) ─────────────────────────
        active_chars = pack.active_characters
        active_plots = pack.active_plotlines
        active_locs  = pack.nearby_locations

        batch_tasks: list[tuple[str, str, list]] = []  # (kind, prompt, entities)
        if active_chars:
            batch_tasks.append(("char", _batch_character_prompt(active_chars, prose, summary), active_chars))
        if active_plots:
            batch_tasks.append(("plot", _batch_plotline_prompt(active_plots, prose, summary), active_plots))
        if active_locs:
            batch_tasks.append(("loc", _batch_location_prompt(active_locs, prose, summary), active_locs))
        if pack.pov_state:
            batch_tasks.append(("pov", _pov_prompt(pack.pov_state, pack, prose, summary, template=tpl_pov), []))

        # Discovery call (all 5 entity types in one LLM call)
        existing_char_names  = [e.name for e in pack.character_roster]
        existing_loc_names   = [l.name for l in _db_module.get_all_locations(state.story_id, db_path)]
        existing_plot_names  = [p.name for p in _db_module.get_all_plotlines(state.story_id, db_path)]
        existing_rule_titles = [r.title for r in _db_module.get_all_world_rules(db_path)]
        existing_lore_titles = [l.title for l in _db_module.get_all_world_lore(db_path)]
        discovery_prompt = _new_entities_prompt(
            prose, existing_char_names, existing_loc_names,
            existing_plot_names, existing_rule_titles, existing_lore_titles,
        )

        # ── Run all LLM calls in parallel ─────────────────────────────────────
        total_calls = len(batch_tasks) + 1  # +1 for discovery
        _p(f"  Running {total_calls} LLM calls in parallel (batch mode)...")

        def _run_batch(kind, prompt, entities):
            raw = _call_llm(model, ollama_client, prompt, max_tokens=2048)
            return kind, raw, entities

        def _run_discovery(prompt):
            return _call_llm(model, ollama_client, prompt, max_tokens=2048)

        patches = []
        discovery_raw = None

        with ThreadPoolExecutor(max_workers=min(total_calls, 6)) as ex:
            fut_batches = {
                ex.submit(_run_batch, kind, prompt, entities): kind
                for kind, prompt, entities in batch_tasks
            }
            fut_discovery = ex.submit(_run_discovery, discovery_prompt)

            for fut in as_completed(fut_batches):
                kind = fut_batches[fut]
                try:
                    _, raw, entities = fut.result()
                    if kind == "char":
                        batch = _parse_batch_patches(raw, entities, CharacterPatch)
                        for patch in batch:
                            excl = ("entity_type", "entity_id", "source")
                            if _has_changes(patch, exclude=excl):
                                patches.append(patch)
                                _p(f"  char '{patch.entity_id}': {_count_changes(patch, exclude=excl)} field(s) changed")
                            else:
                                _p(f"  char '{patch.entity_id}': no changes")
                    elif kind == "plot":
                        batch = _parse_batch_patches(raw, entities, PlotlinePatch)
                        for patch in batch:
                            excl = ("entity_type", "entity_id", "source")
                            if _has_changes(patch, exclude=excl):
                                patches.append(patch)
                                _p(f"  plot '{patch.entity_id}': {_count_changes(patch, exclude=excl)} field(s) changed")
                            else:
                                _p(f"  plot '{patch.entity_id}': no changes")
                    elif kind == "loc":
                        batch = _parse_batch_patches(raw, entities, LocationPatch)
                        for patch in batch:
                            excl = ("entity_type", "entity_id", "source")
                            if _has_changes(patch, exclude=excl):
                                patches.append(patch)
                                _p(f"  loc '{patch.entity_id}': {_count_changes(patch, exclude=excl)} field(s) changed")
                            else:
                                _p(f"  loc '{patch.entity_id}': no changes")
                    elif kind == "pov":
                        data = _unwrap_envelope(_parse_json(raw))
                        patch = _to_patch(data, POVPatch)
                        if patch:
                            excl = ("entity_type", "source")
                            if _has_changes(patch, exclude=excl):
                                patches.append(patch)
                                _p(f"  POV state: {_count_changes(patch, exclude=excl)} field(s) changed")
                            else:
                                _p("  POV state: no changes")
                        else:
                            _p("  POV state: no patch returned")
                except Exception as e:
                    _p(f"  [warn] {kind} batch failed: {e}")

            try:
                discovery_raw = fut_discovery.result()
            except Exception as e:
                _p(f"  [warn] Discovery call failed: {e}")

        _p(f"  Total patches: {len(patches)}")

        # ── Parse discovery results ────────────────────────────────────────────
        new_characters: list[Character] = []
        new_locations: list[Location] = []
        new_plotlines: list[Plotline] = []
        new_world_rules: list[WorldRule] = []
        new_world_lore: list[WorldLore] = []

        if discovery_raw:
            _p(f"  Parsing new entity discoveries...")
            try:
                disc_data = _parse_json(discovery_raw)
                if isinstance(disc_data, dict):
                    for item in disc_data.get("new_characters", []):
                        try:
                            c = Character(
                                name=item.get("name", "Unknown"),
                                personality=item.get("personality", ""),
                                goals=item.get("goals") or [],
                                emotional_state=item.get("emotional_state", ""),
                                relationships=item.get("relationships") or {},
                                knowledge=item.get("knowledge") or [],
                                secrets=item.get("secrets") or [],
                            )
                            new_characters.append(c)
                            _p(f"    + New character: {c.name}")
                        except Exception as e:
                            _p(f"    [warn] Could not build character: {e}")

                    for item in disc_data.get("new_locations", []):
                        try:
                            loc = Location(
                                name=item.get("name", "Unknown"),
                                description=item.get("description", ""),
                                tone=item.get("tone") or "",
                                political_control=item.get("political_control"),
                                secrets=item.get("secrets") or [],
                                recent_events=item.get("recent_events") or [],
                            )
                            new_locations.append(loc)
                            _p(f"    + New location: {loc.name}")
                        except Exception as e:
                            _p(f"    [warn] Could not build location: {e}")

                    for item in disc_data.get("new_plotlines", []):
                        try:
                            pl = Plotline(
                                name=item.get("name", "Unknown"),
                                status=item.get("status", "active"),
                                progress_stage=item.get("progress_stage", ""),
                                current_tension=int(item.get("current_tension", 5)),
                                next_possible_developments=item.get("next_possible_developments") or [],
                            )
                            new_plotlines.append(pl)
                            _p(f"    + New plotline: {pl.name}")
                        except Exception as e:
                            _p(f"    [warn] Could not build plotline: {e}")

                    for item in disc_data.get("new_world_rules", []):
                        try:
                            r = WorldRule(
                                rule_type=item.get("rule_type", "hard_constraint"),
                                title=item.get("title", "Unknown Rule"),
                                content=item.get("content", ""),
                            )
                            new_world_rules.append(r)
                            _p(f"    + New world rule: {r.title}")
                        except Exception as e:
                            _p(f"    [warn] Could not build world rule: {e}")

                    for item in disc_data.get("new_world_lore", []):
                        try:
                            l = WorldLore(
                                category=item.get("category", "canon_fact"),
                                title=item.get("title", "Unknown Lore"),
                                content=item.get("content", ""),
                            )
                            new_world_lore.append(l)
                            _p(f"    + New lore: {l.title}")
                        except Exception as e:
                            _p(f"    [warn] Could not build world lore: {e}")
            except Exception as e:
                _p(f"  [warn] Discovery parse failed: {e}")

        total_new = len(new_characters) + len(new_locations) + len(new_plotlines) + \
                    len(new_world_rules) + len(new_world_lore)
        _p(f"  New entities discovered: {total_new}")

        return {
            "memory_patches": patches,
            "new_characters": new_characters,
            "new_locations": new_locations,
            "new_plotlines": new_plotlines,
            "new_world_rules": new_world_rules,
            "new_world_lore": new_world_lore,
        }

    return node


memory_extractor_node = make_memory_extractor_node()  # db_path=None → uses db.DB_PATH default
