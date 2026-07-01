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

from llm_client import MODEL, chat_text
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
# New entity discovery — prompts and parsers
# ---------------------------------------------------------------------------

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
    model: str = MODEL,
    ollama_client=None,
    db_path=None,
) -> Callable[[ChapterGraphState], dict]:

    def node(state: ChapterGraphState) -> dict:
        if not state.chapter_prose:
            raise ValueError("chapter_prose must be set before memory extractor runs")
        if not state.chapter_summary:
            raise ValueError("chapter_summary must be set before memory extractor runs")

        from prompt_templates import get_template
        story_id = state.story_id
        tpl_character = get_template("memory_character", story_id, db_path)
        tpl_plotline = get_template("memory_plotline", story_id, db_path)
        tpl_location = get_template("memory_location", story_id, db_path)
        tpl_pov = get_template("memory_pov", story_id, db_path)
        tpl_new_characters = get_template("memory_new_characters", story_id, db_path)
        tpl_new_locations = get_template("memory_new_locations", story_id, db_path)
        tpl_new_plotlines = get_template("memory_new_plotlines", story_id, db_path)
        tpl_new_world_rules = get_template("memory_new_world_rules", story_id, db_path)
        tpl_new_world_lore = get_template("memory_new_world_lore", story_id, db_path)

        pack = state.context_pack
        prose = state.chapter_prose
        summary = state.chapter_summary
        patches = []

        def _has_changes(patch, exclude=("entity_type", "entity_id", "source")):
            return any(v is not None for f, v in patch.model_dump().items() if f not in exclude)

        def _count_changes(patch, exclude=("entity_type", "entity_id", "source")):
            return sum(1 for f, v in patch.model_dump().items() if f not in exclude and v is not None)

        # --- Characters ---
        for c in pack.active_characters:
            print(f"  Extracting memory: {c.name}...")
            try:
                raw = _call_llm(model, ollama_client, _character_prompt(c, prose, summary, template=tpl_character))
                data = _unwrap_envelope(_parse_json(raw))
                patch = _to_patch(data, CharacterPatch)
                if patch:
                    patch = patch.model_copy(update={"entity_id": c.id})  # always force correct id
                    if _has_changes(patch):
                        patches.append(patch)
                        print(f"    → {_count_changes(patch)} field(s) changed")
                    else:
                        print(f"    → no changes")
            except Exception as e:
                print(f"    [warn] Failed to extract character patch for {c.name}: {e}")

        # --- Plotlines ---
        for p in pack.active_plotlines:
            print(f"  Extracting memory: plotline '{p.name}'...")
            try:
                raw = _call_llm(model, ollama_client, _plotline_prompt(p, prose, summary, template=tpl_plotline))
                data = _unwrap_envelope(_parse_json(raw))
                patch = _to_patch(data, PlotlinePatch)
                if patch:
                    patch = patch.model_copy(update={"entity_id": p.id})
                    if _has_changes(patch):
                        patches.append(patch)
                        print(f"    → {_count_changes(patch)} field(s) changed")
                    else:
                        print(f"    → no changes")
            except Exception as e:
                print(f"    [warn] Failed to extract plotline patch for {p.name}: {e}")

        # --- Locations ---
        for loc in pack.nearby_locations:
            print(f"  Extracting memory: location '{loc.name}'...")
            try:
                raw = _call_llm(model, ollama_client, _location_prompt(loc, prose, summary, template=tpl_location))
                data = _unwrap_envelope(_parse_json(raw))
                patch = _to_patch(data, LocationPatch)
                if patch:
                    patch = patch.model_copy(update={"entity_id": loc.id})
                    if _has_changes(patch):
                        patches.append(patch)
                        print(f"    → {_count_changes(patch)} field(s) changed")
                    else:
                        print(f"    → no changes")
            except Exception as e:
                print(f"    [warn] Failed to extract location patch for {loc.name}: {e}")

        # --- POV State ---
        if pack.pov_state:
            print(f"  Extracting memory: POV state...")
            try:
                raw = _call_llm(model, ollama_client, _pov_prompt(pack.pov_state, pack, prose, summary, template=tpl_pov))
                data = _unwrap_envelope(_parse_json(raw))
                patch = _to_patch(data, POVPatch)
                if patch and _has_changes(patch, exclude=("entity_type", "source")):
                    patches.append(patch)
                    print(f"    → {_count_changes(patch, exclude=('entity_type','source'))} field(s) changed")
                else:
                    print(f"    → no changes")
            except Exception as e:
                print(f"    [warn] Failed to extract POV patch: {e}")

        print(f"  Total patches: {len(patches)}")

        # --- New entity discovery pass ---
        print(f"  Scanning for new entities...")
        new_characters: list[Character] = []
        new_locations: list[Location] = []
        new_plotlines: list[Plotline] = []
        new_world_rules: list[WorldRule] = []
        new_world_lore: list[WorldLore] = []

        import db as _db_module
        _db = db_path
        # Use full DB contents so the exclusion list is complete regardless of
        # what the context builder retrieved this chapter.
        existing_char_names  = [e.name for e in pack.character_roster]  # roster is always full
        existing_loc_names   = [l.name for l in _db_module.get_all_locations(state.story_id, _db)]
        existing_plot_names  = [p.name for p in _db_module.get_all_plotlines(state.story_id, _db)]
        existing_rule_titles = [r.title for r in _db_module.get_all_world_rules(_db)]
        existing_lore_titles = [l.title for l in _db_module.get_all_world_lore(_db)]

        try:
            raw = _call_llm(model, ollama_client, _new_characters_prompt(prose, existing_char_names, template=tpl_new_characters))
            for item in _parse_new_entities(raw):
                if not isinstance(item, dict) or not item.get("name"):
                    continue
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
                    print(f"    + New character: {c.name}")
                except Exception as e:
                    print(f"    [warn] Could not build character from discovery: {e}")
        except Exception as e:
            print(f"    [warn] New character scan failed: {e}")

        try:
            raw = _call_llm(model, ollama_client, _new_locations_prompt(prose, existing_loc_names, template=tpl_new_locations))
            for item in _parse_new_entities(raw):
                if not isinstance(item, dict) or not item.get("name"):
                    continue
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
                    print(f"    + New location: {loc.name}")
                except Exception as e:
                    print(f"    [warn] Could not build location from discovery: {e}")
        except Exception as e:
            print(f"    [warn] New location scan failed: {e}")

        try:
            raw = _call_llm(model, ollama_client, _new_plotlines_prompt(prose, existing_plot_names, template=tpl_new_plotlines))
            for item in _parse_new_entities(raw):
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                try:
                    p = Plotline(
                        name=item.get("name", "Unknown"),
                        status=item.get("status", "active"),
                        progress_stage=item.get("progress_stage", ""),
                        current_tension=int(item.get("current_tension", 5)),
                        next_possible_developments=item.get("next_possible_developments") or [],
                    )
                    new_plotlines.append(p)
                    print(f"    + New plotline: {p.name}")
                except Exception as e:
                    print(f"    [warn] Could not build plotline from discovery: {e}")
        except Exception as e:
            print(f"    [warn] New plotline scan failed: {e}")

        try:
            raw = _call_llm(model, ollama_client, _new_world_rules_prompt(prose, existing_rule_titles, template=tpl_new_world_rules))
            for item in _parse_new_entities(raw):
                if not isinstance(item, dict) or not item.get("title"):
                    continue
                try:
                    r = WorldRule(
                        rule_type=item.get("rule_type", "hard_constraint"),
                        title=item.get("title", "Unknown Rule"),
                        content=item.get("content", ""),
                    )
                    new_world_rules.append(r)
                    print(f"    + New world rule: {r.title}")
                except Exception as e:
                    print(f"    [warn] Could not build world rule from discovery: {e}")
        except Exception as e:
            print(f"    [warn] New world rule scan failed: {e}")

        try:
            raw = _call_llm(model, ollama_client, _new_world_lore_prompt(prose, existing_lore_titles, template=tpl_new_world_lore))
            for item in _parse_new_entities(raw):
                if not isinstance(item, dict) or not item.get("title"):
                    continue
                try:
                    l = WorldLore(
                        category=item.get("category", "canon_fact"),
                        title=item.get("title", "Unknown Lore"),
                        content=item.get("content", ""),
                    )
                    new_world_lore.append(l)
                    print(f"    + New lore: {l.title}")
                except Exception as e:
                    print(f"    [warn] Could not build world lore from discovery: {e}")
        except Exception as e:
            print(f"    [warn] New world lore scan failed: {e}")

        total_new = len(new_characters) + len(new_locations) + len(new_plotlines) + \
                    len(new_world_rules) + len(new_world_lore)
        print(f"  New entities discovered: {total_new}")

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
