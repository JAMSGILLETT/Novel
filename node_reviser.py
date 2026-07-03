"""
Reviser — the Actor half of a Reflexion/Re³-style Actor↔Evaluator loop.

The Evaluator is the combined canon+craft check (canon is *grounded* against the
world-rules DB; craft is subjective). This module is the Actor: given the
Evaluator's structured verdict, it produces a revised chapter and drives the
loop until the prose passes or a retry cap is hit.

Three properties, each borrowed from the papers we validated against:

  • LOCAL EDIT (Re³ "Edit"): the reviser is told to change only the passages
    that caused each flagged problem and leave the rest of the prose verbatim,
    instead of regenerating the whole chapter. Cheaper, and it stops a fix for
    one problem from silently introducing a new one elsewhere.

  • BEST-OF-N (Re³ "Rewrite"): each revision samples N candidate edits at a
    higher temperature, re-scores each with the Evaluator, and keeps the
    lowest-scoring (fewest/least-severe problems). A candidate that scores 0
    ends the search early.

  • MEMORY (Reflexion episodic buffer): every attempt's remaining problems are
    accumulated and shown to the reviser on the next attempt, so it doesn't
    repeat a fix that already failed or reintroduce a problem an earlier
    attempt had cleared.

pipeline.py calls make_reviser_loop(...) and gets back a callable with the same
(state, verdict, attempts, flagged) return shape the old _run_revision_loop had,
so the surrounding stage/checkpoint code is unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from schema import CanonCheckResult, CraftCheckResult, ChapterGraphState

# How many candidate edits to sample per revision, and the sampling temperature.
# Best-of-N is expensive on local inference (N generations + N evaluations per
# attempt), so N defaults low; override with NOVELGEN_BEST_OF_N.
BEST_OF_N = int(os.environ.get("NOVELGEN_BEST_OF_N", "2"))
EDIT_TEMPERATURE = float(os.environ.get("NOVELGEN_EDIT_TEMPERATURE", "0.8"))

_MAJOR_WEIGHT = 10
_MINOR_WEIGHT = 1
# Canon is a *grounded* hard gate: any canon violation adds a dominating penalty
# so a candidate with a continuity violation can never win best-of-N over one
# that's canon-clean, no matter how much better its craft is. Craft only breaks
# ties among canon-clean candidates. Larger than any realistic craft total.
_CANON_GATE = 100_000

Evaluator = Callable[[ChapterGraphState], Tuple[CanonCheckResult, CraftCheckResult]]


# ---------------------------------------------------------------------------
# Verdict: unifies the canon + craft result pair for the loop and pipeline.
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    canon: CanonCheckResult
    craft: CraftCheckResult

    @property
    def passed(self) -> bool:
        return self.canon.passed and self.craft.passed

    @property
    def violations(self) -> list:
        return list(self.canon.violations)

    @property
    def issues(self) -> list:
        return list(self.craft.issues)

    def score(self) -> int:
        """Lower is better; 0 means fully clean. Majors dominate minors so
        best-of-N never trades a major issue for several minor ones. Canon is a
        hard gate: any canon violation adds _CANON_GATE, so a canon-dirty
        candidate always outscores every canon-clean one and can never win."""
        def _w(p) -> int:
            return _MAJOR_WEIGHT if getattr(p, "severity", "major") == "major" else _MINOR_WEIGHT

        canon = sum(_w(v) for v in self.violations)
        craft = sum(_w(i) for i in self.issues)
        if canon:
            canon += _CANON_GATE
        return canon + craft

    def problem_lines(self) -> List[str]:
        lines = []
        for v in self.violations:
            kind = getattr(v, "violation_type", "?")
            lines.append(f"[{getattr(v, 'severity', '?')}] canon/{kind}: {getattr(v, 'description', '')}")
        for i in self.issues:
            kind = getattr(i, "issue_type", "?")
            lines.append(f"[{getattr(i, 'severity', '?')}] craft/{kind}: {getattr(i, 'description', '')}")
        return lines


# ---------------------------------------------------------------------------
# Local-edit prompt
# ---------------------------------------------------------------------------

def _fmt_memory(memory: List[dict]) -> str:
    if not memory:
        return ""
    blocks = []
    for m in memory:
        probs = "\n".join(f"      - {p}" for p in m["problems"])
        blocks.append(f"  Attempt {m['attempt']} still had these problems:\n{probs}")
    return (
        "\n=== PRIOR ATTEMPTS (learn from these — do not repeat these fixes, and "
        "do not reintroduce a problem an earlier attempt already cleared) ===\n"
        + "\n".join(blocks) + "\n"
    )


def build_edit_prompt(state: ChapterGraphState, verdict: Verdict, memory: List[dict]) -> str:
    from node_story_writer import _fmt_world_rules, _fmt_constraints

    pack = state.context_pack
    plan = state.story_plan
    char_map = {c.id: c.name for c in pack.active_characters} if pack else {}

    problems = "\n".join(f"  • {line}" for line in verdict.problem_lines())
    memory_block = _fmt_memory(memory)

    return f"""You are revising a novel chapter to fix specific problems a reviewer found.

Make TARGETED edits: change only the passages responsible for each problem
below, and leave the rest of the prose exactly as it is. Do NOT rewrite the
whole chapter, drop scenes, or change the length. Preserve the voice and every
part that wasn't flagged.

=== WORLD RULES (canon fixes must satisfy these — they are hard constraints) ===
{_fmt_world_rules(pack) if pack else '  (none)'}

=== CHARACTER CONSTRAINTS ===
{_fmt_constraints(plan, char_map) if plan else '  (none)'}

=== PROBLEMS TO FIX ===
{problems}
{memory_block}
=== CURRENT CHAPTER PROSE ===
{state.chapter_prose}

=== YOUR TASK ===
Output ONLY the full revised chapter prose — every word of it, with the flagged
passages fixed and everything else unchanged. No commentary, no headings, no JSON."""


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    score: int
    prose: str
    verdict: Verdict


def make_reviser_loop(
    evaluator: Evaluator,
    model: Optional[str] = None,
    ollama_client=None,
    best_of_n: int = BEST_OF_N,
    print_fn=print,
) -> Callable[..., Tuple[ChapterGraphState, Verdict, int, bool]]:
    """Build the Actor↔Evaluator revision loop.

    evaluator(state) -> (CanonCheckResult, CraftCheckResult) is the grounded
    combined check. Returns a callable(state, max_retries, first_result=None)
    -> (final_state, final_verdict, attempts, flagged)."""
    p = print_fn

    def _generate_edit(state: ChapterGraphState, verdict: Verdict,
                       memory: List[dict], temperature: float) -> str:
        from llm_client import chat_text
        from llm_json import strip_think
        prompt = build_edit_prompt(state, verdict, memory)
        raw = chat_text(
            prompt, model=model, max_tokens=4096, timeout=900,
            temperature=temperature, client=ollama_client, label="Reviser (local edit)",
        )
        return strip_think(raw).strip()

    def loop(
        state: ChapterGraphState,
        max_retries: int,
        first_result: Optional[Tuple[CanonCheckResult, CraftCheckResult]] = None,
    ) -> Tuple[ChapterGraphState, Verdict, int, bool]:
        current = state
        memory: List[dict] = []

        # Attempt 1's verdict: reuse the upstream check if it was already run on
        # this exact prose, otherwise evaluate now.
        if first_result is not None:
            verdict = Verdict(*first_result)
        else:
            verdict = Verdict(*evaluator(current))
        attempts = 1

        while True:
            score = verdict.score()
            p(f"  Reviser attempt {attempts}/{max_retries + 1} — evaluator score {score} "
              f"({len(verdict.violations)} canon, {len(verdict.issues)} craft)")

            if verdict.passed:
                p("  Evaluator PASSED — no problems remain")
                return current, verdict, attempts, False

            for line in verdict.problem_lines():
                p(f"    {line}")

            memory.append({"attempt": attempts, "problems": verdict.problem_lines()})

            if attempts > max_retries:
                p(f"  Retry cap reached ({max_retries}) — flagging for review, publishing best version")
                return current, verdict, attempts, True

            # ── Best-of-N local edit ────────────────────────────────────────
            p(f"  Sampling {best_of_n} candidate edit(s)...")
            best: Optional[_Candidate] = None
            for i in range(best_of_n):
                temp = EDIT_TEMPERATURE + 0.05 * i  # nudge diversity across samples
                try:
                    edited = _generate_edit(current, verdict, memory, temp)
                except Exception as e:
                    p(f"    candidate {i + 1} generation failed: {e}")
                    continue
                if not edited:
                    continue
                cand_verdict = Verdict(*evaluator(current.model_copy(update={"chapter_prose": edited})))
                cand_score = cand_verdict.score()
                p(f"    candidate {i + 1}: score {cand_score}")
                if best is None or cand_score < best.score:
                    best = _Candidate(cand_score, edited, cand_verdict)
                if cand_score == 0:
                    p("    clean candidate found — stopping best-of-N early")
                    break

            if best is None:
                # Every candidate failed to generate — keep current prose, flag.
                p("  No usable candidate produced — flagging and publishing as-is")
                return current, verdict, attempts, True

            current = current.model_copy(update={"chapter_prose": best.prose})
            verdict = best.verdict  # already evaluated; no re-check next iteration
            attempts += 1

    return loop
