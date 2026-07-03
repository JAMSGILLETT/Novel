"""
Node 9: Reconciliation.

Pure logic — no LLM. Takes the raw memory_patches list from Node 8 and
resolves conflicts before Node 10 writes anything to the DB.

Conflicts detected:
  1. is_alive disagreement — two CharacterPatches for the same entity_id
     disagree on whether the character is alive.
  2. location disagreement — two patches place the same character in different
     locations in the same chapter.
  3. plotline status disagreement — two PlotlinePatches disagree on status
     (e.g. one says "resolved", another says "active").

Resolution strategy (simple, deterministic — no second LLM call):
  - For is_alive: "dead wins" — if any patch says is_alive=False, the character
    is dead. A kill is harder to walk back than a false resurrection.
  - For location: last patch wins (patches are ordered by extraction order,
    which follows active_characters order).
  - For plotline status: most-advanced status wins
    (resolved > abandoned > active > dormant).
  - All other field conflicts: last writer wins (merge by field, later patches
    overwrite earlier ones for the same field).

Conflicted patches that were auto-resolved are recorded in
reconciliation_conflicts with a resolution note. Unresolvable conflicts
(shouldn't happen with current rules, but guarded against) are flagged.

Output:
  reconciled_patches — clean list, one patch per entity, ready for Node 10
  reconciliation_conflicts — audit log of what was resolved and how
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from schema import (
    CharacterPatch, PlotlinePatch, LocationPatch, POVPatch,
    ReconciliationConflict, ChapterGraphState,
)

_STATUS_RANK = {"dormant": 0, "active": 1, "abandoned": 2, "resolved": 3}


def _merge_character_patches(patches: list[CharacterPatch]) -> tuple[CharacterPatch, list[ReconciliationConflict]]:
    conflicts = []
    merged = patches[0].model_copy()

    for later in patches[1:]:
        dump = later.model_dump()
        for field, value in dump.items():
            if field in ("entity_type", "entity_id", "source") or value is None:
                continue

            current = getattr(merged, field, None)

            if field == "is_alive" and current is not None and current != value:
                # Dead wins
                resolved_value = False
                conflicts.append(ReconciliationConflict(
                    description=f"Character {merged.entity_id}: is_alive conflict ({current} vs {value}) — dead wins",
                    conflicting_patches=[merged.model_copy(), later.model_copy()],
                    resolution="False (dead wins)",
                    resolved_by_rule="dead_wins",
                ))
                merged = merged.model_copy(update={"is_alive": resolved_value})

            elif field == "current_location_id" and current is not None and current != value:
                conflicts.append(ReconciliationConflict(
                    description=f"Character {merged.entity_id}: location conflict ({current} vs {value}) — last writer wins",
                    conflicting_patches=[merged.model_copy(), later.model_copy()],
                    resolution=f"{value} (last writer wins)",
                    resolved_by_rule="last_writer_wins",
                ))
                merged = merged.model_copy(update={"current_location_id": value})

            elif field in ("knowledge_added", "secrets_added", "current_objectives", "goals"):
                # Additive fields — union both lists
                existing = getattr(merged, field) or []
                combined = list(dict.fromkeys(existing + (value or [])))
                merged = merged.model_copy(update={field: combined})

            else:
                merged = merged.model_copy(update={field: value})

    return merged, conflicts


def _merge_plotline_patches(patches: list[PlotlinePatch]) -> tuple[PlotlinePatch, list[ReconciliationConflict]]:
    conflicts = []
    merged = patches[0].model_copy()

    for later in patches[1:]:
        dump = later.model_dump()
        for field, value in dump.items():
            if field in ("entity_type", "entity_id", "source") or value is None:
                continue

            current = getattr(merged, field, None)

            if field == "status" and current is not None and current != value:
                winner = max(current, value, key=lambda s: _STATUS_RANK.get(s, 0))
                conflicts.append(ReconciliationConflict(
                    description=f"Plotline {merged.entity_id}: status conflict ({current} vs {value}) — most-advanced wins",
                    conflicting_patches=[merged.model_copy(), later.model_copy()],
                    resolution=f"{winner} (most-advanced wins)",
                    resolved_by_rule="most_advanced_status",
                ))
                merged = merged.model_copy(update={"status": winner})

            elif field in ("next_possible_developments", "involved_character_ids_added", "implies_character_death"):
                existing = getattr(merged, field) or []
                combined = list(dict.fromkeys(existing + (value or [])))
                merged = merged.model_copy(update={field: combined})

            else:
                merged = merged.model_copy(update={field: value})

    return merged, conflicts


def _merge_location_patches(patches: list[LocationPatch]) -> tuple[LocationPatch, list[ReconciliationConflict]]:
    merged = patches[0].model_copy()
    for later in patches[1:]:
        for field, value in later.model_dump().items():
            if field in ("entity_type", "entity_id", "source") or value is None:
                continue
            if field in ("recent_events_added", "secrets_added", "npcs_present"):
                existing = getattr(merged, field) or []
                combined = list(dict.fromkeys(existing + (value or [])))
                merged = merged.model_copy(update={field: combined})
            else:
                merged = merged.model_copy(update={field: value})
    return merged, []


def _merge_pov_patches(patches: list[POVPatch]) -> tuple[POVPatch, list[ReconciliationConflict]]:
    merged = patches[0].model_copy()
    for later in patches[1:]:
        for field, value in later.model_dump().items():
            if field in ("entity_type", "source") or value is None:
                continue
            if field in ("inventory_added", "inventory_removed", "injuries_added", "knowledge_added"):
                existing = getattr(merged, field) or []
                combined = list(dict.fromkeys(existing + (value or [])))
                merged = merged.model_copy(update={field: combined})
            else:
                merged = merged.model_copy(update={field: value})
    return merged, []


def reconcile(state: ChapterGraphState) -> dict:
    patches = state.memory_patches
    all_conflicts: list[ReconciliationConflict] = []
    reconciled = []

    # Group patches by (entity_type, entity_id)
    char_groups: dict[str, list[CharacterPatch]] = defaultdict(list)
    plot_groups: dict[str, list[PlotlinePatch]] = defaultdict(list)
    loc_groups: dict[str, list[LocationPatch]] = defaultdict(list)
    pov_patches: list[POVPatch] = []

    for p in patches:
        if isinstance(p, CharacterPatch):
            char_groups[p.entity_id or ""].append(p)
        elif isinstance(p, PlotlinePatch):
            plot_groups[p.entity_id or ""].append(p)
        elif isinstance(p, LocationPatch):
            loc_groups[p.entity_id or ""].append(p)
        elif isinstance(p, POVPatch):
            pov_patches.append(p)

    for entity_id, group in char_groups.items():
        if len(group) == 1:
            reconciled.append(group[0])
        else:
            merged, conflicts = _merge_character_patches(group)
            reconciled.append(merged)
            all_conflicts.extend(conflicts)

    for entity_id, group in plot_groups.items():
        if len(group) == 1:
            reconciled.append(group[0])
        else:
            merged, conflicts = _merge_plotline_patches(group)
            reconciled.append(merged)
            all_conflicts.extend(conflicts)

    for entity_id, group in loc_groups.items():
        if len(group) == 1:
            reconciled.append(group[0])
        else:
            merged, conflicts = _merge_location_patches(group)
            reconciled.append(merged)
            all_conflicts.extend(conflicts)

    if pov_patches:
        if len(pov_patches) == 1:
            reconciled.append(pov_patches[0])
        else:
            merged, conflicts = _merge_pov_patches(pov_patches)
            reconciled.append(merged)
            all_conflicts.extend(conflicts)

    return {
        "reconciled_patches": reconciled,
        "reconciliation_conflicts": all_conflicts,
    }


def make_reconciliation_node(print_fn=print):
    _p = print_fn

    def node(state: ChapterGraphState) -> dict:
        result = reconcile(state)
        n_conflicts = len(result["reconciliation_conflicts"])
        n_patches = len(result["reconciled_patches"])
        _p(f"  {n_patches} reconciled patch(es), {n_conflicts} conflict(s) resolved")
        for c in result["reconciliation_conflicts"]:
            _p(f"    [resolved] {c.description}")
        return result
    return node


reconciliation_node = make_reconciliation_node()
