"""
DSPy-lite few-shot demonstrations for the combined canon+craft Evaluator.

The lesson we took from DSPy: don't hand-tune a prompt string in isolation —
declare the task as a signature (typed input -> typed output) and attach
*demonstrations* so a small local model sees worked examples of the exact
judgment it must make. On qwen2.5:14b, in-context examples are the single
biggest lever on structured-judgment reliability.

We don't (yet) have a persisted history of real verdicts to mine demos from, so
this is a curated seed bank — the DSPy "bootstrap from a few gold examples"
starting point. Selection is relevance-based: we include the canon example whose
rule category matches a world rule actually in the chapter, one craft example,
and ALWAYS a clean-pass example so the model is calibrated that passing is a
valid answer (guards against over-flagging). Later this bank can be augmented
with optimized demonstrations mined from accumulated real check outcomes — the
Option-3 upgrade path — without changing callers.

SIGNATURE (what the Evaluator maps):
    INPUT   world_rules, character_constraints, chapter_prose
    OUTPUT  CombinedCheckResult(canon_passed, violations, craft_passed, issues)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Set

from schema import CombinedCheckResult, CanonViolation, CraftIssue, ContextPack


@dataclass
class Demo:
    """One worked example: the inputs a reviewer saw and the correct verdict."""
    world_rules: str
    prose: str
    verdict: CombinedCheckResult
    # Which world-rule categories this demo is relevant to (empty = always eligible).
    rule_tags: Set[str] = field(default_factory=set)
    # Coarse role so selection can guarantee coverage.
    role: str = "canon"  # "canon" | "craft" | "clean"


# ---------------------------------------------------------------------------
# Seed bank
# ---------------------------------------------------------------------------

DEMO_BANK: List[Demo] = [
    # Canon — hard magic-system violation.
    Demo(
        role="canon",
        rule_tags={"magic_system"},
        world_rules="[MAGIC_SYSTEM] No magic: Magic is impossible in this world; no character can cast spells.",
        prose=("Cornered, Aldric thrust out his palm and a bolt of white fire tore "
               "across the room, scattering the guards like leaves."),
        verdict=CombinedCheckResult(
            canon_passed=False,
            violations=[CanonViolation(
                violation_type="lore_violation",
                description="\"a bolt of white fire tore across the room\" — Aldric casts magic, but the world rule states magic is impossible.",
                severity="major")],
            craft_passed=True, issues=[]),
    ),
    # Canon — knowledge leak (character acts on info they can't have).
    Demo(
        role="canon",
        rule_tags={"social_rule", "hard_constraint"},
        world_rules="[HARD_CONSTRAINT] Sealed letter: Only the queen has read the treaty; no one else knows its contents.",
        prose=("\"The treaty cedes the northern ports,\" the stablehand muttered to "
               "himself as he brushed down the horse, \"and she thinks no one knows.\""),
        verdict=CombinedCheckResult(
            canon_passed=False,
            violations=[CanonViolation(
                violation_type="knowledge_leak",
                description="\"The treaty cedes the northern ports\" — the stablehand states the treaty's contents, which only the queen knows.",
                severity="major")],
            craft_passed=True, issues=[]),
    ),
    # Craft — telling instead of showing.
    Demo(
        role="craft",
        world_rules="(no special rules relevant to this example)",
        prose=("Mara was extremely sad and also very brave. It was the worst day of "
               "her life. She felt terrible about everything that had happened."),
        verdict=CombinedCheckResult(
            canon_passed=True, violations=[],
            craft_passed=False,
            issues=[CraftIssue(
                issue_type="show_dont_tell",
                description="\"Mara was extremely sad and also very brave\" — emotions are stated outright rather than dramatized through action, sensation, or dialogue.",
                severity="major")]),
    ),
    # Clean pass — nothing wrong. Calibrates the model that PASS is valid.
    Demo(
        role="clean",
        world_rules="[MAGIC_SYSTEM] No magic: Magic is impossible in this world.",
        prose=("Rain needled the courtyard. Sela pressed her back to the cold stone "
               "and counted the guards' footsteps — three, then a pause, then three "
               "more — before she slipped through the gap toward the gate."),
        verdict=CombinedCheckResult(
            canon_passed=True, violations=[], craft_passed=True, issues=[]),
    ),
]


# ---------------------------------------------------------------------------
# Selection + formatting
# ---------------------------------------------------------------------------

_MINED_PROSE_CHARS = 400  # cap a mined demo's prose so the prompt doesn't bloat


def demo_from_verdict(v: dict) -> Optional[Demo]:
    """Build a mined demo from a stored verdict dict (db.get_recent_check_verdicts).

    We only trust CLEAN PASSES as mined demos: a passing chapter is a real,
    in-voice example of prose that should NOT be flagged, which sharpens
    calibration against this story's own style. We deliberately do NOT mine
    flagged verdicts — those are the model's own unresolved judgments and may be
    false positives, so reusing them as demos would reinforce mistakes. (Curated
    violation examples stay in the seed bank; the Option-3 optimizer is where
    mined violation demos would belong, filtered by a metric.)"""
    if not v.get("passed"):
        return None
    prose = (v.get("prose") or "").strip()
    if not prose:
        return None
    if len(prose) > _MINED_PROSE_CHARS:
        prose = prose[:_MINED_PROSE_CHARS].rsplit(" ", 1)[0] + " …"
    return Demo(
        role="clean",
        world_rules=v.get("world_rules") or "(no special rules)",
        prose=prose,
        verdict=CombinedCheckResult(canon_passed=True, violations=[],
                                    craft_passed=True, issues=[]),
    )


def select_demos(
    pack: Optional[ContextPack], k: int = 3, history: Optional[List[dict]] = None,
) -> List[Demo]:
    """Pick up to k demos relevant to this chapter. Always includes one clean-pass
    demo (anti-over-flagging) and one craft demo; fills the rest with canon demos
    whose rule category matches a world rule present in the chapter.

    history: recent stored verdicts (db.get_recent_check_verdicts). When present,
    a real clean-pass from this story is preferred over the seed clean-pass demo,
    so calibration uses the story's own voice."""
    present: Set[str] = set()
    if pack is not None:
        present = {r.rule_type for r in pack.relevant_world_rules}

    # Prefer a real clean pass mined from history; fall back to the seed demo.
    mined_clean = next(
        (d for d in (demo_from_verdict(v) for v in (history or [])) if d is not None),
        None,
    )
    clean = mined_clean or next((d for d in DEMO_BANK if d.role == "clean"), None)
    craft = next((d for d in DEMO_BANK if d.role == "craft"), None)

    # Canon demos ranked: those matching a present rule category first.
    canon = [d for d in DEMO_BANK if d.role == "canon"]
    canon.sort(key=lambda d: 0 if (d.rule_tags & present) else 1)

    chosen: List[Demo] = []
    if clean:
        chosen.append(clean)
    if canon:
        chosen.append(canon[0])
    if craft:
        chosen.append(craft)
    # If room remains, add the next-most-relevant canon demo.
    for d in canon[1:]:
        if len(chosen) >= k:
            break
        chosen.append(d)

    return chosen[:k]


def _verdict_json(v: CombinedCheckResult) -> str:
    return json.dumps({
        "canon_passed": v.canon_passed,
        "violations": [
            {"violation_type": x.violation_type, "description": x.description,
             "related_entity_id": x.related_entity_id, "severity": x.severity}
            for x in v.violations
        ],
        "craft_passed": v.craft_passed,
        "issues": [
            {"issue_type": x.issue_type, "description": x.description, "severity": x.severity}
            for x in v.issues
        ],
    }, indent=2)


def format_demos(demos: List[Demo]) -> str:
    """Render selected demos as a few-shot block for the check prompt."""
    if not demos:
        return ""
    blocks = []
    for i, d in enumerate(demos, 1):
        blocks.append(
            f"--- EXAMPLE {i} ---\n"
            f"WORLD RULES:\n  {d.world_rules}\n"
            f"CHAPTER PROSE:\n  {d.prose}\n"
            f"CORRECT VERDICT:\n{_verdict_json(d.verdict)}"
        )
    return (
        "\n=== WORKED EXAMPLES (judge the real chapter the same way — flag real "
        "problems, but PASS when there are none) ===\n"
        + "\n\n".join(blocks) + "\n"
    )
