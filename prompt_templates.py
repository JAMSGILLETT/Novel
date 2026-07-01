"""
Editable prompt templates, backing the Prompts tab.

Scope: only the static instructional/task wording of each prompt is editable
here. The per-chapter dynamic data blocks each prompt needs (character
lists, plotline tables, world rules, etc.) are still assembled in Python by
each node's existing `_fmt_*` helpers and inserted into the template at a
named placeholder (e.g. `{characters_block}`) — editing a template changes
the wording around that data, not the data itself.

JSON output-format blocks (the literal schema shown to the model) are
deliberately kept OUT of the editable template and appended in code after
`.format()` — editing JSON-shaped instructions is exactly the kind of change
that silently breaks downstream parsing, so it isn't exposed here. This
mirrors a pattern already used by the story planner and character reasoner
prompts before this file existed.

Every `build_*_prompt` function takes an optional `template: str | None`
parameter. When None, it falls back to DEFAULT_TEMPLATES[key] — so existing
callers that don't care about overrides need no changes. Node factories that
want to honor a user's saved override resolve one with get_template(...) and
pass it through explicitly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import db

# ---------------------------------------------------------------------------
# Registry: (group, key, description) in pipeline order — drives the Prompts tab
# ---------------------------------------------------------------------------

PROMPT_REGISTRY: list[tuple[str, str, str]] = [
    ("Input Router", "input_router_classification", "Classify continuation vs. a user-injected plot event"),
    ("Outline Manager", "outline_init", "Generate the initial story outline (cold start)"),
    ("Outline Manager", "outline_revision", "Periodic full outline rewrite"),
    ("Story Planner", "story_planner", "Plan the chapter (scenes, conflicts, constraints, length)"),
    ("Character Reasoner", "character_reasoner", "Per-character intentions/dialogue/emotion (shared template)"),
    ("Story Writer", "story_writer_first", "Write the chapter prose (first draft)"),
    ("Story Writer", "story_writer_revision", "Revise prose to fix canon/craft feedback"),
    ("Canon Check", "canon_check", "Check prose against world rules and established facts"),
    ("Craft Check", "craft_check", "Check prose for pacing/tension/dialogue/voice quality"),
    ("Chapter Summarizer", "chapter_summarizer", "Produce short/medium summary + timeline events"),
    ("Summary Update", "book_summary_continuation", "Roll the new chapter into the running novel summary"),
    ("Summary Update", "book_summary_first", "Write the first novel summary (chapter 1)"),
    ("Summary Update", "act_summary_compression", "Compress a completed act into a permanent summary"),
    ("Memory Extractor", "memory_character", "Extract character state changes"),
    ("Memory Extractor", "memory_plotline", "Extract plotline state changes"),
    ("Memory Extractor", "memory_location", "Extract location state changes"),
    ("Memory Extractor", "memory_pov", "Extract POV state changes"),
    ("Memory Extractor", "memory_new_characters", "Discover newly introduced characters"),
    ("Memory Extractor", "memory_new_locations", "Discover newly introduced locations"),
    ("Memory Extractor", "memory_new_plotlines", "Discover newly introduced plotlines"),
    ("Memory Extractor", "memory_new_world_rules", "Discover newly revealed world rules"),
    ("Memory Extractor", "memory_new_world_lore", "Discover newly revealed world lore"),
]

PROMPT_KEYS: set[str] = {key for _, key, _ in PROMPT_REGISTRY}


# ---------------------------------------------------------------------------
# Default templates
# ---------------------------------------------------------------------------

DEFAULT_TEMPLATES: dict[str, str] = {

    "input_router_classification": """Given this user input to a chapter-generation system, classify whether it is:
  (a) "continuation" — a generic request to continue the story with no specific new event
      (e.g. "continue", "write the next chapter", "what happens next", "go on")
  (b) "user_event_injection" — an explicit instruction that should be treated as a forced
      plot event this chapter (e.g. introducing a new character, forcing an action or
      encounter, changing a character's state directly, or otherwise steering the plot
      in a specific direction)

USER INPUT: "{user_input}\"""",

    "outline_init": """You are a story architect creating a high-level outline for a new novel before writing begins.
This outline will guide chapter-by-chapter planning for the whole book — think in terms of the entire arc, not just the opening.

USER'S OPENING DIRECTIVE: "{user_input}"

EXISTING WORLD RULES:
{rules_text}

EXISTING WORLD LORE:
{lore_text}

PRE-ESTABLISHED CHARACTERS (reference by their exact id if you give them an arc):
{chars_text}

Produce beats as a list of 5 to 10 major story beats spanning the novel's likely arc (not scene-level detail —
book-level turning points). Only give character_arcs to characters listed above; leave the list empty if none are
pre-established yet (arcs will be added automatically as new characters are introduced in later chapters).""",

    "outline_revision": """You are revising the story outline for an ongoing novel, given everything that has happened so far.
This outline guides chapter-by-chapter planning — keep it book-level, not scene-level detail.

=== CURRENT OUTLINE ===
Premise: {premise}
Theme: {theme}
Planned ending: {planned_ending}
Beats:
{beats_text}
Character arcs:
{arcs_text}

=== STORY SO FAR (chapter {chapter_number}) ===
{history}

=== CHARACTER ROSTER (reference by exact id) ===
{roster_text}

=== YOUR TASK ===
Rewrite the full outline:
  - Update beat statuses based on what has actually happened (upcoming → in_progress → completed)
  - Add any new beats the story has earned given how it's developed
  - Update each character's current_stage in their arc based on recent chapters
  - Refine planned_ending if the story's trajectory suggests a clearer resolution
  - Keep premise/theme stable unless the story has clearly diverged from them""",

    "story_planner": """You are the Story Director for a collaborative novel. Your job is to PLAN the next chapter — not write it. The Story Writer executes your plan.

=== MODE: {mode_note} ===

=== STORY OUTLINE (your north star — advance toward this, don't just react to the last chapter) ===
{outline_block}

=== WORLD RULES (ABSOLUTE — never violate, never have a character or scene violate these) ===
{world_rules_block}

=== RELEVANT WORLD LORE ===
{world_lore_block}

=== ACTIVE PLOTLINES (advance at least one) ===
{plotlines_block}

=== CHARACTERS IN SCENE (full profiles) ===
{characters_block}

=== POV STATE ===
{pov_block}

=== NEARBY LOCATIONS ===
{locations_block}

=== STORY HISTORY ===
{history_block}
{dep_section}{stale_plotlines_block}
=== USER DIRECTIVE ===
"{user_input}"

=== FULL CHARACTER ROSTER (all characters including off-screen) ===
{roster_block}
  To bring an off-screen character back, put their exact ID in requested_offscreen_character_ids.
  Do NOT include deceased characters unless their return is plot-justified.

=== YOUR TASK — plan chapter {chapter_number} ===
Produce a StoryPlan with:
  • scenes: 3–5 ordered scene descriptions forming a coherent chapter arc
  • pacing_notes: tone and rhythm guidance (e.g. "slow burn, end on a revelation")
  • conflicts: specific tensions to dramatize this chapter
  • narrative_goals: what this chapter must accomplish (theme, plot, character)
  • character_constraints: for each character who appears:
      - forbidden_actions: things they CANNOT do (consistency or world-rule constraints)
      - required_callbacks: specific beats or plotline IDs they MUST address
  • required_callbacks: story-level mandatory beats independent of any single character
  • target_word_count: choose a length that fits this chapter's role — not every chapter should be the same length:
      - ~500-700 words for a short, punchy, high-tension or climactic chapter
      - ~800-1200 words for a standard chapter
      - ~1300-1800 words for a slow-burn or world-building chapter with more ground to cover
  • requested_offscreen_character_ids: roster IDs you want brought in from off-screen (empty list if none)

Rules:
  - Never violate World Rules
  - Do not kill a character without a required_callback justifying it
  - Reference plotlines by their ID in required_callbacks so the canon checker can verify""",

    "character_reasoner": """You are reasoning from inside the mind of a single character in an ongoing novel.
Do not write prose. Reason about intentions, dialogue goals, and emotional state only.
{history_section}
=== CHARACTER: {character_name} ===
Personality: {personality}
Goals: {goals}
Current objectives: {objectives}
Emotional state: {emotional_state}
Knowledge: {knowledge}
Secrets: {secrets}
Relationships:
{relationships}

=== CHAPTER PLAN (context for your reasoning) ===
Scenes:
{scenes_block}

Pacing: {pacing_notes}

Active conflicts:
{conflicts_block}

=== YOUR CONSTRAINTS THIS CHAPTER ===
Forbidden actions (you CANNOT do these, no matter what):
{forbidden_block}

Required callbacks (beats you MUST address):
{required_block}

=== SCENE CONTEXT ===
{pov_section}
{prior_section}
=== YOUR TASK ===
Reason through how {character_name} will behave this chapter. Produce:
  • action_intentions: list of specific things {character_name} plans to do
  • dialogue_intent: what {character_name} wants to communicate, and what they are hiding or deflecting
  • emotional_response: how {character_name} feels about the situation and the other people present
  • constraint_acknowledgement: for each forbidden action and required callback, one sentence on
    how {character_name}'s own goals and personality lead them to (or away from) that constraint

Stay true to the character's personality and existing relationships. Do not invent new facts.""",

    "story_writer_first": """You are a literary fiction author writing a chapter of an ongoing novel.

HARD REQUIREMENTS — failure on any of these means the output is wrong:
  1. POINT OF VIEW: close third-person limited. Use "he/she/they" — NEVER "I" or "me".
     Stay inside {pov_name}'s head only. Do not show other characters' inner thoughts.
  2. LENGTH: {min_words} to {max_words} words (target ~{target_word_count}, chosen by the Story Planner
     for this chapter's pacing). Count carefully. Do not stop early.
  3. FORMAT: output only the prose. No chapter title, no word count, no commentary.
{style_section}
Do not summarize — show the scene through action, dialogue, and {pov_name}'s internal experience.

=== WORLD RULES (ABSOLUTE — never violate these in the prose) ===
{world_rules_block}

=== STORY HISTORY ===
{history_block}

=== ACTIVE PLOTLINES (at least one must advance) ===
{plotlines_block}

=== SCENE CONTEXT ===
{pov_block}

=== NEARBY LOCATIONS ===
{locations_block}

=== CHARACTERS IN SCENE ===
{characters_block}

=== CHAPTER PLAN ===
{plan_block}

=== HARD CONSTRAINTS (non-negotiable — prose must satisfy all of these) ===
{constraints_block}

=== CHARACTER INNER STATES (use to inform behaviour and dialogue — do not copy verbatim) ===
{reasonings_block}

=== USER DIRECTIVE ===
"{user_input}"

REMINDER BEFORE YOU WRITE:
  - Third-person only ("Kael moved", not "I moved")
  - {min_words}–{max_words} words
  - No title, no meta-commentary — begin the prose directly

Now write chapter {chapter_number}.""",

    "story_writer_revision": """You are revising a chapter of a novel to fix specific canon violations.
{style_section}
=== WORLD RULES (ABSOLUTE — these were violated; fix them) ===
{world_rules_block}

=== VIOLATIONS TO FIX ===
{violations_block}

=== ORIGINAL PROSE (revise this — keep what works, fix what violates) ===
{original_prose}

=== HARD CONSTRAINTS (still apply in the revision) ===
{constraints_block}

Rewrite the chapter fixing every violation listed above while preserving the scene structure and pacing.
HARD REQUIREMENTS: third-person only ("he/she" not "I"), {min_words}–{max_words} words, no title or commentary.
Output only the prose.""",

    "canon_check": """You are a continuity editor reviewing a chapter of a novel.
Your job is to catch genuine canon violations — not to find problems where none exist.
If the chapter is consistent with the rules and character knowledge below, return passed=true with an empty violations list. That is a valid and common outcome.

IMPORTANT GUIDELINES:
  - A character REFERENCING a past event (e.g. "his hand still ached from last night's spell") is NOT a violation.
  - Only flag a world rule violation if magic/abilities are actively USED in a way that breaks the rule.
  - Only flag a knowledge_leak if a character states something they COULD NOT POSSIBLY know — vague or ambiguous dialogue is NOT a leak.
  - Only flag a forbidden_action if the prose explicitly shows the character doing the forbidden thing, not just mentioning it in context.
  - When in doubt, do NOT flag it. False positives waste revision cycles.

=== WORLD RULES (flag only if actively broken in the prose) ===
{world_rules_block}

=== HARD CONSTRAINTS FOR THIS CHAPTER ===
{constraints_block}

=== STORY-LEVEL REQUIRED BEATS (flag only if completely absent from the prose) ===
{required_callbacks_block}

=== CHARACTER PROFILES (use to judge whether behaviour fits personality and knowledge) ===
{characters_block}

=== ESTABLISHED LOCATIONS ===
{locations_block}

=== CHAPTER PROSE ===
{chapter_prose}

=== YOUR TASK ===
Read the prose carefully. Only report a violation if you can quote the specific offending passage and state the exact rule it breaks. If nothing is broken, return passed=true and an empty violations list.""",

    "craft_check": """You are a developmental editor reviewing a chapter of a novel for craft quality — not continuity.
Your job is to catch genuine engagement problems — not to nitpick style choices or find problems where none exist.
If the chapter reads well, return passed=true with an empty issues list. That is a valid and common outcome.

IMPORTANT GUIDELINES:
  - Only flag "pacing" if a scene meaningfully drags or a key beat is rushed past without room to land.
  - Only flag "tension" if a scene genuinely lacks any conflict, stakes, or forward pull — quiet scenes are fine if they serve the story.
  - Only flag "show_dont_tell" for a passage that summarizes an important emotional beat instead of dramatizing it, not for ordinary narrative transitions.
  - Only flag "dialogue" if lines are genuinely stilted, purely expository, or interchangeable between speakers.
  - Only flag "voice_consistency" if the narration actually breaks POV or tense, not for minor stylistic variation.
  - When in doubt, do NOT flag it. False positives waste revision cycles and this is a matter of taste as much as craft.

=== CHAPTER PLAN (what this chapter was meant to accomplish) ===
{plan_block}

=== CHAPTER PROSE ===
{chapter_prose}

=== YOUR TASK ===
Read the prose carefully. Only report an issue if you can point to the specific passage and explain what's weak about it. If nothing is genuinely weak, return passed=true and an empty issues list.""",

    "chapter_summarizer": """You are summarizing a chapter of a novel for use as persistent story memory.
Be precise and concrete — future chapters will rely on this summary to stay consistent.

=== CHAPTER {chapter_number} PROSE ===
{chapter_prose}

=== YOUR TASK ===
Produce three things:

1. short_summary — one sentence (max 30 words). What happened and why it matters.
2. medium_summary — 2 to 4 sentences. Key events, character decisions, and emotional beats. Written in past tense, third person.
3. timeline_events — a list of 3 to 8 concrete facts that future chapters must not contradict. Format: "[Character] did [action] at/in [location]." Only include things that actually happened in the prose.""",

    "book_summary_continuation": """You are updating a running summary of a novel.

CURRENT NOVEL SUMMARY:
{current_summary}

NEW CHAPTER (Chapter {chapter_number}):
{medium_summary}

Key events this chapter:
{timeline_events}

Rewrite the novel summary to incorporate the new chapter. Keep it under 500 words.
Focus on what has happened, not speculation. Third person, past tense.
Output only the updated summary text.""",

    "book_summary_first": """Write a brief novel summary based on Chapter {chapter_number}.

{medium_summary}

Key events:
{timeline_events}

Keep it under 200 words. Third person, past tense. Output only the summary text.""",

    "act_summary_compression": """Compress this section of a novel's rolling summary into a permanent, compact act summary.
This summary will never be re-compressed again, so keep the details that matter for the rest of the book.

ROLLING SUMMARY:
{rolling_summary}

Produce:
1. A compact summary (150 to 200 words, third person, past tense) capturing what happened and why it matters going forward.
2. 3 to 6 key_events: concrete, permanent facts future chapters must not contradict.""",

    "memory_character": """You are extracting memory updates for a novel character after a chapter.
Only report fields that CLEARLY CHANGED in this chapter's prose. Leave everything else null.

CHARACTER: {character_name} (id: {character_id})
Current personality: {personality}
Current emotional state: {emotional_state}
Current goals: {goals}
Current knowledge: {knowledge}
Current objectives: {objectives}

CHAPTER SUMMARY: {chapter_summary}

CHAPTER PROSE:
{chapter_prose}""",

    "memory_plotline": """You are extracting memory updates for a story plotline after a chapter.
Only report fields that CLEARLY CHANGED. Leave everything else null.

PLOTLINE: {plotline_name} (id: {plotline_id})
Current status: {status}
Current stage: {progress_stage}
Current tension (0-10): {current_tension}
Next possible developments: {next_developments}

CHAPTER SUMMARY: {chapter_summary}

CHAPTER PROSE:
{chapter_prose}""",

    "memory_location": """You are extracting memory updates for a location after a novel chapter.
Only report fields that CLEARLY CHANGED. Leave everything else null.

LOCATION: {location_name} (id: {location_id})
Current description: {description}
Current tone: {tone}
Recent events: {recent_events}

CHAPTER SUMMARY: {chapter_summary}

CHAPTER PROSE:
{chapter_prose}""",

    "memory_pov": """You are extracting POV state updates after a novel chapter.
Only report fields that CLEARLY CHANGED. Leave everything else null.

CURRENT POV STATE:
  Location: {location_name}
  Companions: {companions}
  Inventory: {inventory}
  Emotional state: {emotional_state}
  Injuries: {injuries}
  Knowledge: {knowledge}

KNOWN LOCATION IDs (use these for location_id):
{location_options}

CHAPTER SUMMARY: {chapter_summary}

CHAPTER PROSE:
{chapter_prose}""",

    "memory_new_characters": """Read this chapter and identify any NAMED characters who appear but are NOT in the existing roster.
Do not re-list existing characters. Only include genuinely new, named people.

EXISTING CHARACTERS (do not include these): {existing_names}

CHAPTER PROSE:
{chapter_prose}

If new characters exist, return a JSON array. If none, return an empty array [].""",

    "memory_new_locations": """Read this chapter and identify any NAMED locations that appear but are NOT in the existing list.
Do not re-list existing locations. Only include genuinely new, named places.

EXISTING LOCATIONS (do not include these): {existing_names}

CHAPTER PROSE:
{chapter_prose}

If new locations exist, return a JSON array. If none, return an empty array [].""",

    "memory_new_plotlines": """Read this chapter and identify any NEW story threads or conflicts that emerge which are NOT already tracked.
A plotline is a meaningful, ongoing narrative tension — not a single scene beat.

EXISTING PLOTLINES (do not include these): {existing_names}

CHAPTER PROSE:
{chapter_prose}

If new plotlines exist, return a JSON array. If none, return an empty array [].""",

    "memory_new_world_rules": """Read this chapter and identify any new HARD RULES about how this world works that are revealed or established.
A world rule is an absolute constraint — a law of physics, magic system mechanic, or social rule that always applies.
Do NOT include character opinions, plot events, or soft conventions.

EXISTING RULES (do not include these): {existing_names}

CHAPTER PROSE:
{chapter_prose}

If new world rules exist, return a JSON array. If none, return an empty array [].""",

    "memory_new_world_lore": """Read this chapter and identify any new WORLD LORE that is revealed — historical facts, cultural details, political information, or canon facts about this world.
Do not include character-specific knowledge or plot events. Only world-level facts that any informed person in this world might know.

EXISTING LORE (do not include these): {existing_names}

CHAPTER PROSE:
{chapter_prose}

If new lore exists, return a JSON array. If none, return an empty array [].""",
}

assert set(DEFAULT_TEMPLATES) == PROMPT_KEYS, "DEFAULT_TEMPLATES and PROMPT_REGISTRY must list the same keys"


# ---------------------------------------------------------------------------
# Resolution / editing helpers
# ---------------------------------------------------------------------------

def get_template(key: str, story_id: str, db_path: Optional[Path] = None) -> str:
    """Returns the user's saved override for this prompt, or the built-in default."""
    override = db.get_prompt_override(key, story_id, db_path)
    return override if override is not None else DEFAULT_TEMPLATES[key]


def save_template(key: str, story_id: str, template: str, db_path: Optional[Path] = None) -> None:
    db.upsert_prompt_override(key, story_id, template, db_path)


def reset_template(key: str, story_id: str, db_path: Optional[Path] = None) -> None:
    db.delete_prompt_override(key, story_id, db_path)
