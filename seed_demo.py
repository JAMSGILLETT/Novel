"""
Seed a small, self-consistent demo world so you can test generation immediately.

Run once:

    python seed_demo.py

It creates a story called "Emberfall" (story_id: emberfall-demo) alongside any
existing stories — it does NOT touch "my-story". Launch the GUI, pick
"Emberfall" from the Story dropdown, go to the Chat tab, and generate.

What it seeds:
  • 3 world rules (the magic system + two hard constraints)
  • 5 pieces of world lore (history / politics / geography / culture / canon)
  • 4 characters (a POV lead, an ally, an antagonist, a mentor)
  • 3 locations
  • 3 active plotlines
  • the POV state (who/where the story opens on)
  • a story outline (premise / theme / beats / arcs) so chapter 1 skips the
    cold-start outline-generation LLM call
  • 1 canon dependency rule (pulls the magic system in whenever the lead appears)

Re-running is safe: every record has a fixed id, so it upserts in place.

Embedding into the Chroma vector store needs sentence-transformers + chromadb.
If those aren't installed the DB is still seeded (the mandatory-retrieval pass
and name matching work without vectors); it just prints a note and continues.
"""
from __future__ import annotations

from pathlib import Path

import db
from schema import (
    Character, Location, Plotline, POVState, WorldRule, WorldLore,
    StoryOutline, StoryBeat, CharacterArcNote, CanonRule,
)

DB_PATH     = Path(__file__).parent / "story.db"
CHROMA_PATH = Path(__file__).parent / "story_chroma"

STORY_ID   = "emberfall-demo"
BOOK_TITLE = "Emberfall"


# ---------------------------------------------------------------------------
# World rules — hard constraints the canon check will hold the prose to
# ---------------------------------------------------------------------------
WORLD_RULES = [
    WorldRule(
        id="rule-ashbinding",
        rule_type="magic_system",
        title="Ashbinding",
        content=(
            "A mage binds to a single ember drawn from the Emberwell. Channeling its "
            "power lets them shape fire and heat, but every binding burns years from "
            "the mage's own life — an ashbinder ages visibly with heavy use, and a "
            "spent ember can never be rebound."
        ),
    ),
    WorldRule(
        id="rule-the-verge",
        rule_type="hard_constraint",
        title="The Verge",
        content=(
            "The world ends at the Verge, a churning boundary of sky and ash-sea. "
            "Nothing living has ever crossed it and returned. The dead are given to "
            "the Verge; the living do not approach it."
        ),
    ),
    WorldRule(
        id="rule-guild-law",
        rule_type="social_rule",
        title="Guild Law of Licensure",
        content=(
            "Only the Ember Concord may license an ashbinder. Binding without a "
            "license is 'severance' — a capital crime punished by cutting the "
            "offender from their ember, which is invariably fatal."
        ),
    ),
]

# ---------------------------------------------------------------------------
# World lore — discoverable background the retriever can surface
# ---------------------------------------------------------------------------
WORLD_LORE = [
    WorldLore(
        id="lore-the-sundering",
        category="history",
        title="The Sundering",
        content=(
            "Two centuries ago the old kingdom overreached, binding a thousand embers "
            "at once to raise a bridge across the Verge. The backlash — the Sundering — "
            "burned the kingdom to cinders and left the Emberwell half-drowned."
        ),
    ),
    WorldLore(
        id="lore-ember-concord",
        category="politics",
        title="The Ember Concord",
        content=(
            "The council of licensed masters that rose from the ashes of the Sundering. "
            "It rations embers, licenses binders, and hunts the unlicensed through its "
            "Wardens. Publicly it preserves order; privately it hoards the strongest embers."
        ),
    ),
    WorldLore(
        id="lore-cindergate",
        category="geography",
        title="Cindergate",
        content=(
            "The last great city, built in the crater the Sundering left. Its tiers "
            "descend toward the drowned Emberwell at the bottom; the higher you live, "
            "the further from the ash and the closer to the Concord's Spire."
        ),
    ),
    WorldLore(
        id="lore-lantern-vigil",
        category="culture",
        title="The Lantern Vigil",
        content=(
            "When someone dies, the living set a paper lantern adrift toward the Verge "
            "so the dead find the boundary. To withhold a lantern is the deepest insult; "
            "to set one for the living is a curse."
        ),
    ),
    WorldLore(
        id="lore-embermarks",
        category="canon_fact",
        title="Embermarks",
        content=(
            "A binding leaves a mark on the skin that darkens with each channel — a "
            "visible ledger of how much life a mage has spent. Wardens read embermarks "
            "to judge how dangerous, and how desperate, an unlicensed binder has become."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Characters — ids are fixed so relationships / POV wiring line up
# ---------------------------------------------------------------------------
KAELEN = Character(
    id="char-kaelen",
    name="Kaelen Vos",
    personality=(
        "Guarded and quick-witted, hides fear behind dry humor. Fiercely loyal to the "
        "few he trusts, reckless with his own life, unwilling to ask for help."
    ),
    goals=[
        "Stay ahead of the Wardens' severance warrant",
        "Find a way to unbind an ember before it kills him",
    ],
    relationships={
        "Mira Sonne": "trusts her; owes her his life",
        "Warden Thecca": "hunted by her",
        "Old Bram": "former mentor, uneasy debt",
    },
    knowledge=[
        "Bound an unlicensed ember three years ago",
        "The Concord's records are kept in the Drowned Archive",
    ],
    emotional_state="wary, running on too little sleep",
    current_location_id="loc-cinderworks",
    current_objectives=["Reach the Drowned Archive before the Wardens close the tiers"],
    secrets=["His embermark has spread to his throat — he has far less time than he admits"],
)

MIRA = Character(
    id="char-mira",
    name="Mira Sonne",
    personality=(
        "Precise, principled archivist with a subversive streak. Believes knowledge "
        "should be free of the Concord. Calm under pressure, terrible liar."
    ),
    goals=["Expose what the Concord hid after the Sundering"],
    relationships={"Kaelen Vos": "protective of him; frustrated by his recklessness"},
    knowledge=["The Drowned Archive's flooded lower stacks hold pre-Sundering ledgers"],
    emotional_state="determined",
    current_location_id="loc-drowned-archive",
    current_objectives=["Copy the Concord's ember-rationing ledger before it's moved"],
    secrets=["She has already made one forbidden copy and hidden it"],
)

THECCA = Character(
    id="char-thecca",
    name="Warden Thecca",
    personality=(
        "Disciplined, unhurried, genuinely believes the Concord's control saves lives. "
        "Reads people like embermarks. Merciful only when mercy serves order."
    ),
    goals=["Execute the severance warrant on Kaelen Vos", "Keep the Archive's secret buried"],
    relationships={"Kaelen Vos": "a warrant to close; a puzzle she respects"},
    knowledge=["Kaelen's embermark is spreading — he'll grow desperate and predictable"],
    emotional_state="patient, certain",
    current_location_id="loc-wardens-spire",
    current_objectives=["Seal the tiers and drive Kaelen toward the Spire"],
    secrets=["She was there at the vault the night the Concord drowned the Archive's lower stacks"],
)

BRAM = Character(
    id="char-bram",
    name="Old Bram",
    personality=(
        "A retired binder turned fence, all warmth and bad knees. Hoards favors like "
        "coin. Cowardly in the open, unexpectedly steadfast when cornered."
    ),
    goals=["Stay invisible to the Concord", "Not lose another apprentice"],
    relationships={"Kaelen Vos": "the apprentice he failed once; wants to make it right"},
    knowledge=["Rumors of an unbinding ritual lost in the Sundering"],
    emotional_state="nervous",
    current_location_id="loc-cinderworks",
    current_objectives=[],
    secrets=["He knows the drowned route into the Archive but fears the water"],
)

CHARACTERS = [KAELEN, MIRA, THECCA, BRAM]

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
LOCATIONS = [
    Location(
        id="loc-cinderworks",
        name="The Cinderworks",
        description=(
            "A warren of soot-black forges and tenements on Cindergate's lowest dry tier, "
            "where unlicensed binders sell heat by the breath and everyone owes everyone."
        ),
        political_control="nominally the Concord; in practice the fences and gangs",
        tone="cramped, smoky, watchful",
        recent_events=["Wardens posted a severance warrant with Kaelen's embermark sketched on it"],
    ),
    Location(
        id="loc-drowned-archive",
        name="The Drowned Archive",
        description=(
            "The Concord's record-hall, its lower stacks deliberately flooded after the "
            "Sundering. Dry galleries above, black water below, and ledgers rotting in between."
        ),
        political_control="the Ember Concord",
        tone="hushed, damp, secretive",
        secrets=["The pre-Sundering ember-rationing ledgers survive in the flooded stacks"],
    ),
    Location(
        id="loc-wardens-spire",
        name="The Wardens' Spire",
        description=(
            "The Concord's tower crowning the highest tier, where warrants are issued and "
            "severances carried out. Its lamplit heights never see ash."
        ),
        political_control="the Ember Concord",
        tone="cold, orderly, imposing",
    ),
]

# ---------------------------------------------------------------------------
# Plotlines — all active so the context builder always includes them
# ---------------------------------------------------------------------------
PLOTLINES = [
    Plotline(
        id="plot-severance-warrant",
        name="The Severance Warrant",
        status="active",
        progress_stage="Wardens have posted the warrant and begun sealing the tiers",
        current_tension=6,
        involved_character_ids=["char-kaelen", "char-thecca"],
        next_possible_developments=[
            "A checkpoint forces Kaelen off the direct route",
            "Thecca offers a false bargain",
            "Kaelen is recognized by his embermark",
        ],
        last_touched_chapter=0,
    ),
    Plotline(
        id="plot-drowned-ledger",
        name="The Drowned Ledger",
        status="active",
        progress_stage="Mira has located the ledger but can't reach the flooded stacks alone",
        current_tension=5,
        involved_character_ids=["char-mira", "char-kaelen", "char-bram"],
        next_possible_developments=[
            "Bram reveals the drowned route",
            "A copy surfaces where it shouldn't",
            "The Concord moves the ledger",
        ],
        last_touched_chapter=0,
    ),
    Plotline(
        id="plot-ember-debt",
        name="Ember Debt",
        status="active",
        progress_stage="Kaelen's embermark is spreading; each binding costs him more",
        current_tension=7,
        involved_character_ids=["char-kaelen", "char-bram"],
        next_possible_developments=[
            "Kaelen is forced to bind and visibly ages",
            "Bram hints at a lost unbinding ritual",
        ],
        last_touched_chapter=0,
    ),
]

# ---------------------------------------------------------------------------
# POV state — where/who the story opens on
# ---------------------------------------------------------------------------
POV = POVState(
    location_id="loc-cinderworks",
    pov_character_id="char-kaelen",
    companions=[],
    inventory=["a fence's forged tier-pass", "a cold ember stub", "Mira's coded note"],
    emotional_state="tense, hunted",
    goals=["Reach the Drowned Archive and find Mira before the tiers seal"],
    knowledge=["A severance warrant is out for him", "Mira is waiting at the Archive"],
)

# ---------------------------------------------------------------------------
# Story outline — seeding this lets chapter 1 skip the cold-start LLM outline call
# ---------------------------------------------------------------------------
OUTLINE = StoryOutline(
    story_id=STORY_ID,
    premise=(
        "A dying unlicensed mage and a rogue archivist race to expose the secret the "
        "Ember Concord drowned after the Sundering — before a Warden's severance "
        "warrant, and his own spreading embermark, run out his time."
    ),
    theme="What we burn of ourselves for the people and truths we refuse to abandon.",
    planned_ending=(
        "Kaelen spends the last of his ember to bring the drowned ledger into the light, "
        "breaking the Concord's grip even as it costs him everything."
    ),
    beats=[
        StoryBeat(description="Kaelen flees the Cinderworks toward the Archive as the tiers seal",
                  status="upcoming", related_plotline_id="plot-severance-warrant"),
        StoryBeat(description="Kaelen and Mira reach the flooded stacks and recover the ledger",
                  status="upcoming", related_plotline_id="plot-drowned-ledger"),
        StoryBeat(description="Thecca corners them with the truth of the drowned Archive",
                  status="upcoming", related_plotline_id="plot-severance-warrant"),
        StoryBeat(description="Kaelen confronts the true cost of unbinding his ember",
                  status="upcoming", related_plotline_id="plot-ember-debt"),
    ],
    character_arcs=[
        CharacterArcNote(character_id="char-kaelen",
                         arc_summary="From surviving alone to spending himself for something larger",
                         current_stage="running, trusting no one"),
        CharacterArcNote(character_id="char-mira",
                         arc_summary="From cautious archivist to open defiance of the Concord",
                         current_stage="gathering proof"),
        CharacterArcNote(character_id="char-thecca",
                         arc_summary="From certain enforcer to doubt about what order costs",
                         current_stage="fully certain"),
    ],
)

# ---------------------------------------------------------------------------
# Canon dependency rule — whenever Kaelen appears, force the magic system into
# context (its aging cost is central to every scene he's in)
# ---------------------------------------------------------------------------
CANON_RULES = [
    CanonRule(
        rule_id="rule-kaelen-ashbinding",
        story_id=STORY_ID,
        trigger_entity_type="character",
        trigger_entity_id="char-kaelen",
        inject_entity_type="world_rule",
        inject_entity_id="rule-ashbinding",
        reason="Kaelen is an ashbinder; the life-cost of binding must constrain every scene he's in.",
    ),
]


def _seed_db() -> None:
    db.init_db(DB_PATH)
    db.create_story(STORY_ID, BOOK_TITLE, DB_PATH)

    for r in WORLD_RULES:
        db.upsert_world_rule(r, DB_PATH)
    for l in WORLD_LORE:
        db.upsert_world_lore(l, DB_PATH)
    for c in CHARACTERS:
        db.upsert_character(c, STORY_ID, DB_PATH)
    for loc in LOCATIONS:
        db.upsert_location(loc, STORY_ID, DB_PATH)
    for p in PLOTLINES:
        db.upsert_plotline(p, STORY_ID, DB_PATH)
    db.upsert_pov_state(POV, STORY_ID, DB_PATH)
    db.upsert_story_outline(OUTLINE, DB_PATH)
    for rule in CANON_RULES:
        db.insert_canon_rule(rule, DB_PATH)

    print(f"  DB seeded: {len(WORLD_RULES)} rules, {len(WORLD_LORE)} lore, "
          f"{len(CHARACTERS)} characters, {len(LOCATIONS)} locations, {len(PLOTLINES)} plotlines.")


def _seed_vectors() -> bool:
    """Embed entities into Chroma so vector retrieval finds them. Returns False
    (with a note) if the optional embedding/vector deps aren't installed."""
    try:
        import vector_store as vs
        from embeddings import get_default_embedder
    except Exception as e:  # pragma: no cover - depends on optional deps
        print(f"  [note] Skipped vector embedding ({e}). The DB seed still works: "
              f"world rules/lore, the POV character + location, and active plotlines are "
              f"always included, and characters named in your prompt are matched by name.")
        return False

    emb = get_default_embedder()
    client = vs.get_chroma_client(CHROMA_PATH)

    def embed(entity_type: str, entity, story_id=None):
        text = vs.ENTITY_TEXT_FN[entity_type](entity)
        vs.upsert_entity(client, entity_type, entity.id, emb.embed(text), text, story_id=story_id)

    for r in WORLD_RULES:
        embed("world_rule", r)
    for l in WORLD_LORE:
        embed("world_lore", l)
    for c in CHARACTERS:
        embed("character", c, STORY_ID)
    for loc in LOCATIONS:
        embed("location", loc, STORY_ID)
    for p in PLOTLINES:
        embed("plotline", p, STORY_ID)

    print("  Vectors embedded into Chroma (retrieval will surface relevant entities).")
    return True


if __name__ == "__main__":
    print(f"Seeding demo story “{BOOK_TITLE}” (story_id: {STORY_ID})…")
    _seed_db()
    _seed_vectors()
    print(
        "\nDone. Launch the app (python main.py), pick “Emberfall” from the Story\n"
        "dropdown, open the Chat tab, and try a prompt like:\n"
        '  "Kaelen slips out of the Cinderworks toward the Drowned Archive as the Wardens seal the tiers."\n'
    )
