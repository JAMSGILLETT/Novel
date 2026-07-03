"""
Self-Refine — a cheap, craft-only self-improvement pass on the writer's draft,
run BEFORE the grounded canon+craft Evaluator and best-of-N reviser.

Based on Self-Refine (Madaan et al., 2023): one model critiques and improves its
own output in a short loop with no external evaluator. The paper's known weak
spot is exactly self-verification without grounding — so we apply it ONLY to
craft (subjective, no ground truth) and leave canon to the external grounded
gate downstream. Canon facts are shown only as guardrails ("don't introduce
anything that breaks these"), never as something this pass judges or fixes.

Why it's worth a pass: it adds cheap same-model calls with NO external check per
iteration, and a stronger draft means the downstream reviser triggers fewer
best-of-N rounds — each of which is N generations + N *grounded* evaluations,
the most expensive step in the pipeline. Front-load cheap self-editing to avoid
expensive grounded revision.

Contract: node(state) -> {"chapter_prose": improved}. Never raises on LLM error
(returns the prose unchanged). Disabled entirely when NOVELGEN_SELF_REFINE=0.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from llm_client import chat_text
from llm_json import strip_think
from schema import ChapterGraphState

# Cap iterations hard — Self-Refine's gains flatten fast and over-editing is a
# real risk. Override with NOVELGEN_SELF_REFINE_ITERS.
MAX_ITERS = int(os.environ.get("NOVELGEN_SELF_REFINE_ITERS", "2"))
# Sentinel the model emits when the draft needs no craft changes.
_CLEAN_SENTINEL = "NO_CHANGES"
# If a "refined" draft barely differs from the input, treat it as a no-op and
# stop, rather than churning near-identical text.
_MIN_CHANGE_RATIO = 0.02


def is_enabled() -> bool:
    return os.environ.get("NOVELGEN_SELF_REFINE", "1") != "0"


def _fmt_world_rules(state: ChapterGraphState) -> str:
    pack = state.context_pack
    if pack is None or not pack.relevant_world_rules:
        return "  (none)"
    return "\n".join(
        f"  • [{r.rule_type.upper()}] {r.title}: {r.content}"
        for r in pack.relevant_world_rules
    )


def build_self_refine_prompt(state: ChapterGraphState) -> str:
    # Anchor the length band to the CURRENT draft, not the plan target — a
    # refine pass should preserve length, and pinning to the plan target can
    # contradict the actual draft (e.g. "keep 340–460 words" on a short draft),
    # which makes the model punt instead of editing.
    wc = max(len((state.chapter_prose or "").split()), 1)
    lo, hi = int(wc * 0.85), int(wc * 1.15)
    return f"""You are a demanding line editor doing a craft pass on a chapter draft.
Assume it has weaknesses and find them. Look hard at ONLY writing craft, not
continuity:
  • show-don't-tell — emotions/traits STATED outright ("she was scared", "it was
    bad") instead of dramatized through action, sensation, or dialogue. This is
    the most common flaw — flag every instance.
  • pacing — parts that drag or rush past what should land.
  • tension — flat scenes with no conflict, stakes, or forward pull.
  • dialogue — stilted, expository, or interchangeable voices.
  • voice — drift in tense or POV.

Rewrite the weak passages so they show rather than tell and carry real tension.
Keep the passages that already work. Do not rewrite wholesale; keep length
roughly {lo}–{hi} words. Do NOT change plot facts, character knowledge, or
anything that would break the world rules below (guardrails, not your job here):
{_fmt_world_rules(state)}

Respond with ONLY the full revised chapter prose — every word of it, no
commentary or headings. ONLY if the prose is already strong on every craft axis
above with nothing worth improving, respond with exactly: {_CLEAN_SENTINEL}

=== DRAFT TO EDIT ===
{state.chapter_prose}"""


def _changed_enough(before: str, after: str) -> bool:
    """Cheap change test: did the refined text differ enough to be a real edit?"""
    if not after or after == before:
        return False
    # Character-length delta as a fast proxy; avoids importing a diff lib.
    denom = max(len(before), 1)
    return abs(len(after) - len(before)) / denom >= _MIN_CHANGE_RATIO or after.strip() != before.strip()


def make_self_refine_node(
    model: Optional[str] = None,
    ollama_client=None,
    db_path: Optional[Path] = None,
    max_iters: int = MAX_ITERS,
    print_fn=print,
) -> Callable[[ChapterGraphState], dict]:
    _p = print_fn

    def node(state: ChapterGraphState) -> dict:
        if not is_enabled():
            _p("  [self-refine] disabled (NOVELGEN_SELF_REFINE=0) — skipping")
            return {"chapter_prose": state.chapter_prose}
        if not state.chapter_prose:
            return {"chapter_prose": state.chapter_prose}

        prose = state.chapter_prose
        current = state
        for i in range(1, max_iters + 1):
            prompt = build_self_refine_prompt(current)
            try:
                raw = chat_text(
                    prompt, model=model, max_tokens=4096, timeout=900,
                    client=ollama_client, label=f"Self-refine (iter {i})",
                )
            except Exception as e:
                _p(f"  [self-refine] iter {i} failed ({e}) — keeping current draft")
                break

            out = strip_think(raw).strip()
            if not out or out.upper().startswith(_CLEAN_SENTINEL):
                _p(f"  [self-refine] iter {i}: no craft changes — stopping")
                break
            if not _changed_enough(prose, out):
                _p(f"  [self-refine] iter {i}: change negligible — stopping")
                break

            before_wc, after_wc = len(prose.split()), len(out.split())
            _p(f"  [self-refine] iter {i}: revised ({before_wc}->{after_wc} words)")
            prose = out
            current = current.model_copy(update={"chapter_prose": prose})

        return {"chapter_prose": prose}

    return node
