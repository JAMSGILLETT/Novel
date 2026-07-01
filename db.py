"""
SQLite layer. Grows incrementally as nodes are built.

Storage model: each entity type gets its own table with a json_data TEXT column.
This avoids schema churn as domain models evolve — the Pydantic model is the
schema, and SQLite is just a keyed blob store.

Tables:
  chapters          — node 1 (cold-start detection)
  characters        — node 2+ (context builder, persistence)
  plotlines
  locations
  world_rules
  world_lore
  pov_state         — one row per story (no id column, story_id IS the PK)
  chapter_summaries — node 7 output
  canon_rules       — manually authored dependency-graph rules
  story_outline     — one row per story: premise/theme/beats/character arcs
  act_summaries     — permanent per-act summaries (hierarchical book summary)
  story_metadata    — key/value store per story (book_summary, style_sample, ...)
  prompt_overrides  — user-edited prompt templates (Prompts tab), per story + template name

Write helpers accept an optional `conn`: when given, the caller owns the
transaction (used by node_persistence.py so a whole chapter's writes commit or
roll back atomically); when omitted, each call opens/commits/closes its own
connection exactly as before, so existing callers are unaffected.
"""
import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from schema import (
    ActSummary, CanonRule, Character, ChapterSummary, Location,
    Plotline, POVState, StoryOutline, WorldEntity, WorldLore, WorldRule,
)

DB_PATH = Path(__file__).parent / "novelgen.db"


def _conn_and_owned(db_path: Optional[Path], conn: Optional[sqlite3.Connection]):
    """Returns (connection, owns_it). If conn is passed in, the caller manages
    commit/close; otherwise a fresh connection is opened and this call owns it."""
    if conn is not None:
        return conn, False
    return get_connection(db_path), True


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path if db_path is not None else DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    conn = get_connection(db_path)
    stmts = [
        """CREATE TABLE IF NOT EXISTS chapters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            story_id TEXT NOT NULL,
            chapter_number INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(story_id, chapter_number)
        )""",
        """CREATE TABLE IF NOT EXISTS characters (
            id TEXT PRIMARY KEY,
            story_id TEXT NOT NULL,
            json_data TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS plotlines (
            id TEXT PRIMARY KEY,
            story_id TEXT NOT NULL,
            json_data TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS locations (
            id TEXT PRIMARY KEY,
            story_id TEXT NOT NULL,
            json_data TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS world_rules (
            id TEXT PRIMARY KEY,
            json_data TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS world_lore (
            id TEXT PRIMARY KEY,
            json_data TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS pov_state (
            story_id TEXT PRIMARY KEY,
            json_data TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS chapter_summaries (
            story_id TEXT NOT NULL,
            chapter_number INTEGER NOT NULL,
            json_data TEXT NOT NULL,
            PRIMARY KEY (story_id, chapter_number)
        )""",
        """CREATE TABLE IF NOT EXISTS world_entities (
            id TEXT NOT NULL,
            story_id TEXT NOT NULL,
            category TEXT NOT NULL,
            json_data TEXT NOT NULL,
            PRIMARY KEY (id)
        )""",
        """CREATE TABLE IF NOT EXISTS story_metadata (
            story_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (story_id, key)
        )""",
        """CREATE TABLE IF NOT EXISTS canon_rules (
            rule_id TEXT PRIMARY KEY,
            story_id TEXT NOT NULL,
            trigger_entity_type TEXT NOT NULL,
            trigger_entity_id TEXT NOT NULL,
            inject_entity_type TEXT NOT NULL,
            inject_entity_id TEXT NOT NULL,
            reason TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS story_outline (
            story_id TEXT PRIMARY KEY,
            json_data TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS act_summaries (
            story_id TEXT NOT NULL,
            act_number INTEGER NOT NULL,
            json_data TEXT NOT NULL,
            PRIMARY KEY (story_id, act_number)
        )""",
        """CREATE TABLE IF NOT EXISTS prompt_overrides (
            story_id TEXT NOT NULL,
            name TEXT NOT NULL,
            template TEXT NOT NULL,
            PRIMARY KEY (story_id, name)
        )""",
    ]
    for stmt in stmts:
        conn.execute(stmt)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Chapter helpers (node 1)
# ---------------------------------------------------------------------------

def has_any_chapters(story_id: str, db_path: Optional[Path] = None) -> bool:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT 1 FROM chapters WHERE story_id = ? LIMIT 1", (story_id,)
    ).fetchone()
    conn.close()
    return row is not None


def get_latest_chapter_number(story_id: str, db_path: Optional[Path] = None) -> Optional[int]:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT MAX(chapter_number) AS max_ch FROM chapters WHERE story_id = ?",
        (story_id,),
    ).fetchone()
    conn.close()
    return row["max_ch"] if row and row["max_ch"] is not None else None


# ---------------------------------------------------------------------------
# Generic entity getter (fetches one row by id + story_id)
# ---------------------------------------------------------------------------

def _get_entity_json(
    table: str, entity_id: str, story_id: str, db_path: Optional[Path]
) -> Optional[str]:
    conn = get_connection(db_path)
    row = conn.execute(
        f"SELECT json_data FROM {table} WHERE id = ? AND story_id = ?",
        (entity_id, story_id),
    ).fetchone()
    conn.close()
    return row["json_data"] if row else None


def _get_all_json(
    table: str, story_id: str, db_path: Optional[Path]
) -> List[str]:
    conn = get_connection(db_path)
    rows = conn.execute(
        f"SELECT json_data FROM {table} WHERE story_id = ?", (story_id,)
    ).fetchall()
    conn.close()
    return [r["json_data"] for r in rows]


# ---------------------------------------------------------------------------
# Typed getters (node 2: context builder)
# ---------------------------------------------------------------------------

def get_character_by_id(
    entity_id: str, story_id: str, db_path: Optional[Path] = None
) -> Optional[Character]:
    raw = _get_entity_json("characters", entity_id, story_id, db_path)
    return Character.model_validate_json(raw) if raw else None


def get_plotline_by_id(
    entity_id: str, story_id: str, db_path: Optional[Path] = None
) -> Optional[Plotline]:
    raw = _get_entity_json("plotlines", entity_id, story_id, db_path)
    return Plotline.model_validate_json(raw) if raw else None


def get_location_by_id(
    entity_id: str, story_id: str, db_path: Optional[Path] = None
) -> Optional[Location]:
    raw = _get_entity_json("locations", entity_id, story_id, db_path)
    return Location.model_validate_json(raw) if raw else None


def get_world_rule_by_id(
    entity_id: str, db_path: Optional[Path] = None
) -> Optional[WorldRule]:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT json_data FROM world_rules WHERE id = ?", (entity_id,)
    ).fetchone()
    conn.close()
    return WorldRule.model_validate_json(row["json_data"]) if row else None


def get_world_lore_by_id(
    entity_id: str, db_path: Optional[Path] = None
) -> Optional[WorldLore]:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT json_data FROM world_lore WHERE id = ?", (entity_id,)
    ).fetchone()
    conn.close()
    return WorldLore.model_validate_json(row["json_data"]) if row else None


def get_all_characters(story_id: str, db_path: Optional[Path] = None) -> List[Character]:
    rows = _get_all_json("characters", story_id, db_path)
    return [Character.model_validate_json(r) for r in rows]


def get_all_locations(story_id: str, db_path: Optional[Path] = None) -> List[Location]:
    rows = _get_all_json("locations", story_id, db_path)
    return [Location.model_validate_json(r) for r in rows]


def get_all_plotlines(story_id: str, db_path: Optional[Path] = None) -> List[Plotline]:
    rows = _get_all_json("plotlines", story_id, db_path)
    return [Plotline.model_validate_json(r) for r in rows]


def get_all_world_rules(db_path: Optional[Path] = None) -> List[WorldRule]:
    conn = get_connection(db_path)
    rows = conn.execute("SELECT json_data FROM world_rules").fetchall()
    conn.close()
    return [WorldRule.model_validate_json(r["json_data"]) for r in rows]


def get_all_world_lore(db_path: Optional[Path] = None) -> List[WorldLore]:
    conn = get_connection(db_path)
    rows = conn.execute("SELECT json_data FROM world_lore").fetchall()
    conn.close()
    return [WorldLore.model_validate_json(r["json_data"]) for r in rows]


def get_pov_state(story_id: str, db_path: Optional[Path] = None) -> Optional[POVState]:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT json_data FROM pov_state WHERE story_id = ?", (story_id,)
    ).fetchone()
    conn.close()
    return POVState.model_validate_json(row["json_data"]) if row else None


def get_last_chapter_summary(
    story_id: str, before_chapter: int, db_path: Optional[Path] = None
) -> Optional[ChapterSummary]:
    """Returns the most recent chapter summary with chapter_number < before_chapter."""
    conn = get_connection(db_path)
    row = conn.execute(
        """SELECT json_data FROM chapter_summaries
           WHERE story_id = ? AND chapter_number < ?
           ORDER BY chapter_number DESC LIMIT 1""",
        (story_id, before_chapter),
    ).fetchone()
    conn.close()
    return ChapterSummary.model_validate_json(row["json_data"]) if row else None


def get_canon_rules_triggered_by(
    story_id: str, entity_ids: List[str], db_path: Optional[Path] = None
) -> List[CanonRule]:
    if not entity_ids:
        return []
    conn = get_connection(db_path)
    placeholders = ",".join("?" * len(entity_ids))
    rows = conn.execute(
        f"""SELECT * FROM canon_rules
            WHERE story_id = ? AND trigger_entity_id IN ({placeholders})""",
        [story_id] + list(entity_ids),
    ).fetchall()
    conn.close()
    return [CanonRule(**dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Write helpers (used by tests and node 10)
#
# All accept an optional `conn`. Pass one to make several writes part of a
# single caller-owned transaction (node_persistence.py does this); omit it to
# get the old standalone open/commit/close behavior.
# ---------------------------------------------------------------------------

def upsert_character(
    c: Character, story_id: str, db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    _conn, owned = _conn_and_owned(db_path, conn)
    _conn.execute(
        "INSERT OR REPLACE INTO characters (id, story_id, json_data) VALUES (?, ?, ?)",
        (c.id, story_id, c.model_dump_json()),
    )
    if owned:
        _conn.commit()
        _conn.close()


def upsert_plotline(
    p: Plotline, story_id: str, db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    _conn, owned = _conn_and_owned(db_path, conn)
    _conn.execute(
        "INSERT OR REPLACE INTO plotlines (id, story_id, json_data) VALUES (?, ?, ?)",
        (p.id, story_id, p.model_dump_json()),
    )
    if owned:
        _conn.commit()
        _conn.close()


def upsert_location(
    loc: Location, story_id: str, db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    _conn, owned = _conn_and_owned(db_path, conn)
    _conn.execute(
        "INSERT OR REPLACE INTO locations (id, story_id, json_data) VALUES (?, ?, ?)",
        (loc.id, story_id, loc.model_dump_json()),
    )
    if owned:
        _conn.commit()
        _conn.close()


def upsert_world_rule(
    r: WorldRule, db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    _conn, owned = _conn_and_owned(db_path, conn)
    _conn.execute(
        "INSERT OR REPLACE INTO world_rules (id, json_data) VALUES (?, ?)",
        (r.id, r.model_dump_json()),
    )
    if owned:
        _conn.commit()
        _conn.close()


def upsert_world_lore(
    l: WorldLore, db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    _conn, owned = _conn_and_owned(db_path, conn)
    _conn.execute(
        "INSERT OR REPLACE INTO world_lore (id, json_data) VALUES (?, ?)",
        (l.id, l.model_dump_json()),
    )
    if owned:
        _conn.commit()
        _conn.close()


def upsert_pov_state(
    pov: POVState, story_id: str, db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    _conn, owned = _conn_and_owned(db_path, conn)
    _conn.execute(
        "INSERT OR REPLACE INTO pov_state (story_id, json_data) VALUES (?, ?)",
        (story_id, pov.model_dump_json()),
    )
    if owned:
        _conn.commit()
        _conn.close()


def upsert_chapter_summary(
    s: ChapterSummary, story_id: str, db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    _conn, owned = _conn_and_owned(db_path, conn)
    _conn.execute(
        """INSERT OR REPLACE INTO chapter_summaries (story_id, chapter_number, json_data)
           VALUES (?, ?, ?)""",
        (story_id, s.chapter_number, s.model_dump_json()),
    )
    if owned:
        _conn.commit()
        _conn.close()


def insert_canon_rule(
    rule: CanonRule, db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    _conn, owned = _conn_and_owned(db_path, conn)
    _conn.execute(
        """INSERT OR REPLACE INTO canon_rules
           (rule_id, story_id, trigger_entity_type, trigger_entity_id,
            inject_entity_type, inject_entity_id, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (rule.rule_id, rule.story_id, rule.trigger_entity_type, rule.trigger_entity_id,
         rule.inject_entity_type, rule.inject_entity_id, rule.reason),
    )
    if owned:
        _conn.commit()
        _conn.close()


def upsert_world_entity(
    e: WorldEntity, story_id: str, db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    _conn, owned = _conn_and_owned(db_path, conn)
    _conn.execute(
        "INSERT OR REPLACE INTO world_entities (id, story_id, category, json_data) VALUES (?, ?, ?, ?)",
        (e.id, story_id, e.category, e.model_dump_json()),
    )
    if owned:
        _conn.commit()
        _conn.close()


def get_world_entities_by_category(
    category: str, story_id: str, db_path: Optional[Path] = None
) -> List[WorldEntity]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT json_data FROM world_entities WHERE category = ? AND story_id = ?",
        (category, story_id),
    ).fetchall()
    conn.close()
    return [WorldEntity.model_validate_json(r["json_data"]) for r in rows]


def get_all_world_entities(story_id: str, db_path: Optional[Path] = None) -> List[WorldEntity]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT json_data FROM world_entities WHERE story_id = ?", (story_id,)
    ).fetchall()
    conn.close()
    return [WorldEntity.model_validate_json(r["json_data"]) for r in rows]


def delete_world_entity(entity_id: str, db_path: Optional[Path] = None) -> None:
    conn = get_connection(db_path)
    conn.execute("DELETE FROM world_entities WHERE id = ?", (entity_id,))
    conn.commit()
    conn.close()


def get_book_summary(story_id: str, db_path: Optional[Path] = None) -> Optional[str]:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT value FROM story_metadata WHERE story_id = ? AND key = 'book_summary'",
        (story_id,),
    ).fetchone()
    conn.close()
    return row["value"] if row else None


def upsert_book_summary(
    story_id: str, summary: str, db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    _conn, owned = _conn_and_owned(db_path, conn)
    _conn.execute(
        "INSERT OR REPLACE INTO story_metadata (story_id, key, value) VALUES (?, 'book_summary', ?)",
        (story_id, summary),
    )
    if owned:
        _conn.commit()
        _conn.close()


# ---------------------------------------------------------------------------
# Story outline (node_outline_manager)
# ---------------------------------------------------------------------------

def get_story_outline(story_id: str, db_path: Optional[Path] = None) -> Optional[StoryOutline]:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT json_data FROM story_outline WHERE story_id = ?", (story_id,)
    ).fetchone()
    conn.close()
    return StoryOutline.model_validate_json(row["json_data"]) if row else None


def upsert_story_outline(
    outline: StoryOutline, db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    _conn, owned = _conn_and_owned(db_path, conn)
    _conn.execute(
        "INSERT OR REPLACE INTO story_outline (story_id, json_data) VALUES (?, ?)",
        (outline.story_id, outline.model_dump_json()),
    )
    if owned:
        _conn.commit()
        _conn.close()


# ---------------------------------------------------------------------------
# Act summaries (hierarchical book summary — permanent, one per act)
# ---------------------------------------------------------------------------

def get_all_act_summaries(story_id: str, db_path: Optional[Path] = None) -> List[ActSummary]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT json_data FROM act_summaries WHERE story_id = ? ORDER BY act_number ASC",
        (story_id,),
    ).fetchall()
    conn.close()
    return [ActSummary.model_validate_json(r["json_data"]) for r in rows]


def insert_act_summary(
    story_id: str, act: ActSummary, db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    _conn, owned = _conn_and_owned(db_path, conn)
    _conn.execute(
        "INSERT OR REPLACE INTO act_summaries (story_id, act_number, json_data) VALUES (?, ?, ?)",
        (story_id, act.act_number, act.model_dump_json()),
    )
    if owned:
        _conn.commit()
        _conn.close()


# ---------------------------------------------------------------------------
# Style sample (user-provided prose reference for the writer prompt)
# ---------------------------------------------------------------------------

def get_style_sample(story_id: str, db_path: Optional[Path] = None) -> Optional[str]:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT value FROM story_metadata WHERE story_id = ? AND key = 'style_sample'",
        (story_id,),
    ).fetchone()
    conn.close()
    return row["value"] if row else None


def upsert_style_sample(
    story_id: str, sample: str, db_path: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    _conn, owned = _conn_and_owned(db_path, conn)
    _conn.execute(
        "INSERT OR REPLACE INTO story_metadata (story_id, key, value) VALUES (?, 'style_sample', ?)",
        (story_id, sample),
    )
    if owned:
        _conn.commit()
        _conn.close()


# ---------------------------------------------------------------------------
# Prompt template overrides (Prompts tab)
# ---------------------------------------------------------------------------

def get_prompt_override(name: str, story_id: str, db_path: Optional[Path] = None) -> Optional[str]:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT template FROM prompt_overrides WHERE story_id = ? AND name = ?",
        (story_id, name),
    ).fetchone()
    conn.close()
    return row["template"] if row else None


def upsert_prompt_override(
    name: str, story_id: str, template: str, db_path: Optional[Path] = None,
) -> None:
    conn = get_connection(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO prompt_overrides (story_id, name, template) VALUES (?, ?, ?)",
        (story_id, name, template),
    )
    conn.commit()
    conn.close()


def delete_prompt_override(name: str, story_id: str, db_path: Optional[Path] = None) -> None:
    conn = get_connection(db_path)
    conn.execute(
        "DELETE FROM prompt_overrides WHERE story_id = ? AND name = ?", (story_id, name)
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized DB at {DB_PATH}")
