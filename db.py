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

Connection sharing: every read and write helper takes an optional `conn`. When
given, the caller owns the transaction/connection (node_persistence.py commits a
whole chapter atomically; node_context_builder.py reuses one connection for its
dozen-plus reads instead of opening a fresh one each time). When omitted, each
call opens/commits/closes its own connection exactly as before.
"""
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from schema import (
    ActSummary, CanonRule, Character, ChapterSummary, Location,
    Plotline, POVState, StoryOutline, WorldEntity, WorldLore, WorldRule,
)

DB_PATH = Path(__file__).parent / "novelgen.db"

BACKUP_RETENTION = 10  # keep the last N snapshots in backups/


def _conn_and_owned(db_path: Optional[Path], conn: Optional[sqlite3.Connection]):
    """Returns (connection, owns_it). If conn is passed in, the caller manages
    commit/close; otherwise a fresh connection is opened and this call owns it."""
    if conn is not None:
        return conn, False
    return get_connection(db_path), True


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path if db_path is not None else DB_PATH)
    conn.row_factory = sqlite3.Row
    # WAL lets readers and the single writer proceed without blocking each other,
    # and synchronous=NORMAL drops the per-write fsync stall (safe under WAL).
    # Both are cheap no-ops once set; journal_mode persists in the DB file.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# ---------------------------------------------------------------------------
# Low-level primitives — every getter/upsert routes through these so the
# open→execute→(commit)→close dance lives in exactly one place and every call
# transparently supports a caller-supplied connection.
# ---------------------------------------------------------------------------

def _fetch_one(sql: str, params, db_path: Optional[Path], conn: Optional[sqlite3.Connection]):
    _conn, owned = _conn_and_owned(db_path, conn)
    try:
        return _conn.execute(sql, params).fetchone()
    finally:
        if owned:
            _conn.close()


def _fetch_all(sql: str, params, db_path: Optional[Path], conn: Optional[sqlite3.Connection]):
    _conn, owned = _conn_and_owned(db_path, conn)
    try:
        return _conn.execute(sql, params).fetchall()
    finally:
        if owned:
            _conn.close()


def _write(sql: str, params, db_path: Optional[Path], conn: Optional[sqlite3.Connection]) -> None:
    _conn, owned = _conn_and_owned(db_path, conn)
    _conn.execute(sql, params)
    if owned:
        _conn.commit()
        _conn.close()


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
        """CREATE TABLE IF NOT EXISTS stories (
            story_id TEXT PRIMARY KEY,
            book_title TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )""",
        # In-flight chapter checkpoint: full pipeline state serialized after each
        # completed node, so a crash mid-chapter resumes instead of restarting.
        # One row per story — only one chapter generates at a time.
        """CREATE TABLE IF NOT EXISTS chapter_checkpoints (
            story_id TEXT PRIMARY KEY,
            chapter_number INTEGER NOT NULL,
            last_stage TEXT NOT NULL,
            state_json TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        # story_id is the hot filter on nearly every read but isn't a PK on
        # these tables — index it so lookups stay O(log n) as a story grows.
        "CREATE INDEX IF NOT EXISTS idx_characters_story ON characters(story_id)",
        "CREATE INDEX IF NOT EXISTS idx_plotlines_story ON plotlines(story_id)",
        "CREATE INDEX IF NOT EXISTS idx_locations_story ON locations(story_id)",
        "CREATE INDEX IF NOT EXISTS idx_world_entities_story ON world_entities(story_id)",
        "CREATE INDEX IF NOT EXISTS idx_canon_rules_trigger ON canon_rules(story_id, trigger_entity_id)",
    ]
    for stmt in stmts:
        conn.execute(stmt)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Chapter helpers (node 1)
# ---------------------------------------------------------------------------

def has_any_chapters(story_id: str, db_path: Optional[Path] = None, conn=None) -> bool:
    row = _fetch_one(
        "SELECT 1 FROM chapters WHERE story_id = ? LIMIT 1", (story_id,), db_path, conn
    )
    return row is not None


def get_latest_chapter_number(story_id: str, db_path: Optional[Path] = None, conn=None) -> Optional[int]:
    row = _fetch_one(
        "SELECT MAX(chapter_number) AS max_ch FROM chapters WHERE story_id = ?",
        (story_id,), db_path, conn,
    )
    return row["max_ch"] if row and row["max_ch"] is not None else None


# ---------------------------------------------------------------------------
# Generic entity JSON getters (id + story_id keyed tables)
# ---------------------------------------------------------------------------

def _get_entity_json(table: str, entity_id: str, story_id: str, db_path, conn=None) -> Optional[str]:
    row = _fetch_one(
        f"SELECT json_data FROM {table} WHERE id = ? AND story_id = ?",
        (entity_id, story_id), db_path, conn,
    )
    return row["json_data"] if row else None


def _get_all_json(table: str, story_id: str, db_path, conn=None) -> List[str]:
    rows = _fetch_all(
        f"SELECT json_data FROM {table} WHERE story_id = ?", (story_id,), db_path, conn
    )
    return [r["json_data"] for r in rows]


# ---------------------------------------------------------------------------
# Typed getters (node 2: context builder)
# ---------------------------------------------------------------------------

def get_character_by_id(entity_id: str, story_id: str, db_path: Optional[Path] = None, conn=None) -> Optional[Character]:
    raw = _get_entity_json("characters", entity_id, story_id, db_path, conn)
    return Character.model_validate_json(raw) if raw else None


def get_plotline_by_id(entity_id: str, story_id: str, db_path: Optional[Path] = None, conn=None) -> Optional[Plotline]:
    raw = _get_entity_json("plotlines", entity_id, story_id, db_path, conn)
    return Plotline.model_validate_json(raw) if raw else None


def get_location_by_id(entity_id: str, story_id: str, db_path: Optional[Path] = None, conn=None) -> Optional[Location]:
    raw = _get_entity_json("locations", entity_id, story_id, db_path, conn)
    return Location.model_validate_json(raw) if raw else None


def get_world_rule_by_id(entity_id: str, db_path: Optional[Path] = None, conn=None) -> Optional[WorldRule]:
    row = _fetch_one("SELECT json_data FROM world_rules WHERE id = ?", (entity_id,), db_path, conn)
    return WorldRule.model_validate_json(row["json_data"]) if row else None


def get_world_lore_by_id(entity_id: str, db_path: Optional[Path] = None, conn=None) -> Optional[WorldLore]:
    row = _fetch_one("SELECT json_data FROM world_lore WHERE id = ?", (entity_id,), db_path, conn)
    return WorldLore.model_validate_json(row["json_data"]) if row else None


def get_all_characters(story_id: str, db_path: Optional[Path] = None, conn=None) -> List[Character]:
    return [Character.model_validate_json(r) for r in _get_all_json("characters", story_id, db_path, conn)]


def get_all_locations(story_id: str, db_path: Optional[Path] = None, conn=None) -> List[Location]:
    return [Location.model_validate_json(r) for r in _get_all_json("locations", story_id, db_path, conn)]


def get_all_plotlines(story_id: str, db_path: Optional[Path] = None, conn=None) -> List[Plotline]:
    return [Plotline.model_validate_json(r) for r in _get_all_json("plotlines", story_id, db_path, conn)]


def get_all_world_rules(db_path: Optional[Path] = None, conn=None) -> List[WorldRule]:
    rows = _fetch_all("SELECT json_data FROM world_rules", (), db_path, conn)
    return [WorldRule.model_validate_json(r["json_data"]) for r in rows]


def get_all_world_lore(db_path: Optional[Path] = None, conn=None) -> List[WorldLore]:
    rows = _fetch_all("SELECT json_data FROM world_lore", (), db_path, conn)
    return [WorldLore.model_validate_json(r["json_data"]) for r in rows]


def get_pov_state(story_id: str, db_path: Optional[Path] = None, conn=None) -> Optional[POVState]:
    row = _fetch_one("SELECT json_data FROM pov_state WHERE story_id = ?", (story_id,), db_path, conn)
    return POVState.model_validate_json(row["json_data"]) if row else None


def get_last_chapter_summary(
    story_id: str, before_chapter: int, db_path: Optional[Path] = None, conn=None
) -> Optional[ChapterSummary]:
    """Returns the most recent chapter summary with chapter_number < before_chapter."""
    row = _fetch_one(
        """SELECT json_data FROM chapter_summaries
           WHERE story_id = ? AND chapter_number < ?
           ORDER BY chapter_number DESC LIMIT 1""",
        (story_id, before_chapter), db_path, conn,
    )
    return ChapterSummary.model_validate_json(row["json_data"]) if row else None


def get_canon_rules_triggered_by(
    story_id: str, entity_ids: List[str], db_path: Optional[Path] = None, conn=None
) -> List[CanonRule]:
    if not entity_ids:
        return []
    placeholders = ",".join("?" * len(entity_ids))
    rows = _fetch_all(
        f"""SELECT * FROM canon_rules
            WHERE story_id = ? AND trigger_entity_id IN ({placeholders})""",
        [story_id] + list(entity_ids), db_path, conn,
    )
    return [CanonRule(**dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Write helpers (used by tests and node 10)
#
# All accept an optional `conn`. Pass one to make several writes part of a
# single caller-owned transaction (node_persistence.py does this); omit it to
# get the standalone open/commit/close behavior.
# ---------------------------------------------------------------------------

def upsert_character(c: Character, story_id: str, db_path: Optional[Path] = None, conn=None) -> None:
    _write("INSERT OR REPLACE INTO characters (id, story_id, json_data) VALUES (?, ?, ?)",
           (c.id, story_id, c.model_dump_json()), db_path, conn)


def upsert_plotline(p: Plotline, story_id: str, db_path: Optional[Path] = None, conn=None) -> None:
    _write("INSERT OR REPLACE INTO plotlines (id, story_id, json_data) VALUES (?, ?, ?)",
           (p.id, story_id, p.model_dump_json()), db_path, conn)


def upsert_location(loc: Location, story_id: str, db_path: Optional[Path] = None, conn=None) -> None:
    _write("INSERT OR REPLACE INTO locations (id, story_id, json_data) VALUES (?, ?, ?)",
           (loc.id, story_id, loc.model_dump_json()), db_path, conn)


def upsert_world_rule(r: WorldRule, db_path: Optional[Path] = None, conn=None) -> None:
    _write("INSERT OR REPLACE INTO world_rules (id, json_data) VALUES (?, ?)",
           (r.id, r.model_dump_json()), db_path, conn)


def upsert_world_lore(l: WorldLore, db_path: Optional[Path] = None, conn=None) -> None:
    _write("INSERT OR REPLACE INTO world_lore (id, json_data) VALUES (?, ?)",
           (l.id, l.model_dump_json()), db_path, conn)


def upsert_pov_state(pov: POVState, story_id: str, db_path: Optional[Path] = None, conn=None) -> None:
    _write("INSERT OR REPLACE INTO pov_state (story_id, json_data) VALUES (?, ?)",
           (story_id, pov.model_dump_json()), db_path, conn)


def upsert_chapter_summary(s: ChapterSummary, story_id: str, db_path: Optional[Path] = None, conn=None) -> None:
    _write("""INSERT OR REPLACE INTO chapter_summaries (story_id, chapter_number, json_data)
              VALUES (?, ?, ?)""",
           (story_id, s.chapter_number, s.model_dump_json()), db_path, conn)


def insert_canon_rule(rule: CanonRule, db_path: Optional[Path] = None, conn=None) -> None:
    _write("""INSERT OR REPLACE INTO canon_rules
              (rule_id, story_id, trigger_entity_type, trigger_entity_id,
               inject_entity_type, inject_entity_id, reason)
              VALUES (?, ?, ?, ?, ?, ?, ?)""",
           (rule.rule_id, rule.story_id, rule.trigger_entity_type, rule.trigger_entity_id,
            rule.inject_entity_type, rule.inject_entity_id, rule.reason), db_path, conn)


def upsert_world_entity(e: WorldEntity, story_id: str, db_path: Optional[Path] = None, conn=None) -> None:
    _write("INSERT OR REPLACE INTO world_entities (id, story_id, category, json_data) VALUES (?, ?, ?, ?)",
           (e.id, story_id, e.category, e.model_dump_json()), db_path, conn)


def get_world_entities_by_category(
    category: str, story_id: str, db_path: Optional[Path] = None, conn=None
) -> List[WorldEntity]:
    rows = _fetch_all(
        "SELECT json_data FROM world_entities WHERE category = ? AND story_id = ?",
        (category, story_id), db_path, conn,
    )
    return [WorldEntity.model_validate_json(r["json_data"]) for r in rows]


def get_all_world_entities(story_id: str, db_path: Optional[Path] = None, conn=None) -> List[WorldEntity]:
    rows = _fetch_all(
        "SELECT json_data FROM world_entities WHERE story_id = ?", (story_id,), db_path, conn
    )
    return [WorldEntity.model_validate_json(r["json_data"]) for r in rows]


def delete_world_entity(entity_id: str, db_path: Optional[Path] = None, conn=None) -> None:
    _write("DELETE FROM world_entities WHERE id = ?", (entity_id,), db_path, conn)


def get_book_summary(story_id: str, db_path: Optional[Path] = None, conn=None) -> Optional[str]:
    row = _fetch_one(
        "SELECT value FROM story_metadata WHERE story_id = ? AND key = 'book_summary'",
        (story_id,), db_path, conn,
    )
    return row["value"] if row else None


def upsert_book_summary(story_id: str, summary: str, db_path: Optional[Path] = None, conn=None) -> None:
    _write("INSERT OR REPLACE INTO story_metadata (story_id, key, value) VALUES (?, 'book_summary', ?)",
           (story_id, summary), db_path, conn)


# ---------------------------------------------------------------------------
# Story outline (node_outline_manager)
# ---------------------------------------------------------------------------

def get_story_outline(story_id: str, db_path: Optional[Path] = None, conn=None) -> Optional[StoryOutline]:
    row = _fetch_one("SELECT json_data FROM story_outline WHERE story_id = ?", (story_id,), db_path, conn)
    return StoryOutline.model_validate_json(row["json_data"]) if row else None


def upsert_story_outline(outline: StoryOutline, db_path: Optional[Path] = None, conn=None) -> None:
    _write("INSERT OR REPLACE INTO story_outline (story_id, json_data) VALUES (?, ?)",
           (outline.story_id, outline.model_dump_json()), db_path, conn)


# ---------------------------------------------------------------------------
# Act summaries (hierarchical book summary — permanent, one per act)
# ---------------------------------------------------------------------------

def get_all_act_summaries(story_id: str, db_path: Optional[Path] = None, conn=None) -> List[ActSummary]:
    rows = _fetch_all(
        "SELECT json_data FROM act_summaries WHERE story_id = ? ORDER BY act_number ASC",
        (story_id,), db_path, conn,
    )
    return [ActSummary.model_validate_json(r["json_data"]) for r in rows]


def insert_act_summary(story_id: str, act: ActSummary, db_path: Optional[Path] = None, conn=None) -> None:
    _write("INSERT OR REPLACE INTO act_summaries (story_id, act_number, json_data) VALUES (?, ?, ?)",
           (story_id, act.act_number, act.model_dump_json()), db_path, conn)


# ---------------------------------------------------------------------------
# Style sample (user-provided prose reference for the writer prompt)
# ---------------------------------------------------------------------------

def get_style_sample(story_id: str, db_path: Optional[Path] = None, conn=None) -> Optional[str]:
    row = _fetch_one(
        "SELECT value FROM story_metadata WHERE story_id = ? AND key = 'style_sample'",
        (story_id,), db_path, conn,
    )
    return row["value"] if row else None


def upsert_style_sample(story_id: str, sample: str, db_path: Optional[Path] = None, conn=None) -> None:
    _write("INSERT OR REPLACE INTO story_metadata (story_id, key, value) VALUES (?, 'style_sample', ?)",
           (story_id, sample), db_path, conn)


# ---------------------------------------------------------------------------
# Prompt template overrides (Prompts tab)
# ---------------------------------------------------------------------------

def get_prompt_override(name: str, story_id: str, db_path: Optional[Path] = None, conn=None) -> Optional[str]:
    row = _fetch_one(
        "SELECT template FROM prompt_overrides WHERE story_id = ? AND name = ?",
        (story_id, name), db_path, conn,
    )
    return row["template"] if row else None


def upsert_prompt_override(name: str, story_id: str, template: str, db_path: Optional[Path] = None, conn=None) -> None:
    _write("INSERT OR REPLACE INTO prompt_overrides (story_id, name, template) VALUES (?, ?, ?)",
           (story_id, name, template), db_path, conn)


def delete_prompt_override(name: str, story_id: str, db_path: Optional[Path] = None, conn=None) -> None:
    _write("DELETE FROM prompt_overrides WHERE story_id = ? AND name = ?", (story_id, name), db_path, conn)


# ---------------------------------------------------------------------------
# Stories registry (GUI story switcher)
# ---------------------------------------------------------------------------

def create_story(story_id: str, book_title: str, db_path: Optional[Path] = None, conn=None) -> None:
    """Register a story so the GUI can list/switch to it. Idempotent — updates
    the title if the story already exists."""
    _write("INSERT OR REPLACE INTO stories (story_id, book_title) VALUES (?, ?)",
           (story_id, book_title), db_path, conn)


def get_story_title(story_id: str, db_path: Optional[Path] = None, conn=None) -> Optional[str]:
    row = _fetch_one("SELECT book_title FROM stories WHERE story_id = ?", (story_id,), db_path, conn)
    return row["book_title"] if row else None


def list_stories(db_path: Optional[Path] = None, conn=None) -> List[dict]:
    """Return [{story_id, book_title}, ...]. Includes any story that has chapters
    but was never registered (older DBs), so nothing gets orphaned."""
    rows = _fetch_all(
        """SELECT s.story_id AS story_id, s.book_title AS book_title
             FROM stories s
           UNION
           SELECT c.story_id AS story_id, c.story_id AS book_title
             FROM chapters c
            WHERE c.story_id NOT IN (SELECT story_id FROM stories)
           ORDER BY book_title COLLATE NOCASE""",
        (), db_path, conn,
    )
    return [{"story_id": r["story_id"], "book_title": r["book_title"]} for r in rows]


# ---------------------------------------------------------------------------
# App settings (global key/value, e.g. the selected generation model)
# ---------------------------------------------------------------------------

def get_setting(key: str, default: Optional[str] = None, db_path: Optional[Path] = None, conn=None) -> Optional[str]:
    row = _fetch_one("SELECT value FROM app_settings WHERE key = ?", (key,), db_path, conn)
    return row["value"] if row else default


def set_setting(key: str, value: str, db_path: Optional[Path] = None, conn=None) -> None:
    _write("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)", (key, value), db_path, conn)


# ---------------------------------------------------------------------------
# Database backups (full-file snapshots in backups/, used by the GUI + pipeline)
#
# ---------------------------------------------------------------------------
# Chapter checkpoints — crash recovery for an in-flight chapter.
# pipeline.run_chapter saves the full serialized ChapterGraphState here after
# every completed node; on the next run it resumes from the last completed
# node instead of regenerating the whole chapter. Deleted when the chapter
# finishes normally or the user cancels deliberately.
# ---------------------------------------------------------------------------

_SAVE_CHECKPOINT_SQL = """INSERT INTO chapter_checkpoints (story_id, chapter_number, last_stage, state_json, updated_at)
           VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(story_id) DO UPDATE SET
             chapter_number = excluded.chapter_number,
             last_stage = excluded.last_stage,
             state_json = excluded.state_json,
             updated_at = CURRENT_TIMESTAMP"""


def save_chapter_checkpoint(
    story_id: str, chapter_number: int, last_stage: str, state_json: str,
    db_path: Optional[Path] = None, conn=None,
) -> None:
    if conn is not None:
        _write(_SAVE_CHECKPOINT_SQL, (story_id, chapter_number, last_stage, state_json), db_path, conn)
        return
    # This row exists to survive the machine dying — synchronous=NORMAL (the
    # connection default, see get_connection) lets a power loss / hard crash
    # discard recently committed WAL frames, which is precisely when the
    # checkpoint is needed. FULL fsyncs this commit to disk; one fsync per
    # pipeline node is noise next to the minutes each LLM call takes.
    _conn = get_connection(db_path)
    try:
        _conn.execute("PRAGMA synchronous=FULL")
        _conn.execute(_SAVE_CHECKPOINT_SQL, (story_id, chapter_number, last_stage, state_json))
        _conn.commit()
    finally:
        _conn.close()


def get_chapter_checkpoint(story_id: str, db_path: Optional[Path] = None, conn=None) -> Optional[dict]:
    row = _fetch_one(
        "SELECT chapter_number, last_stage, state_json FROM chapter_checkpoints WHERE story_id = ?",
        (story_id,), db_path, conn,
    )
    if row is None:
        return None
    return {
        "chapter_number": row["chapter_number"],
        "last_stage": row["last_stage"],
        "state_json": row["state_json"],
    }


def delete_chapter_checkpoint(story_id: str, db_path: Optional[Path] = None, conn=None) -> None:
    _write("DELETE FROM chapter_checkpoints WHERE story_id = ?", (story_id,), db_path, conn)


# ---------------------------------------------------------------------------
# A snapshot is a copy of the whole SQLite file, so it covers ALL stories at
# once — restoring rolls the entire database back to that point. Filenames are
# {stem}_{tag}_{YYYYmmdd_HHMMSS}.db where tag is "ch<N>" (auto, before a chapter),
# "manual", or "prerestore" (the safety copy taken before a restore).
# ---------------------------------------------------------------------------

def _backups_dir(db_path: Optional[Path]) -> Path:
    base = db_path if db_path is not None else DB_PATH
    return base.parent / "backups"


def _pretty_tag(tag: str) -> str:
    if tag.startswith("ch") and tag[2:].isdigit():
        return f"Chapter {tag[2:]}"
    return tag.replace("prerestore", "Pre-restore").capitalize()


def list_backups(db_path: Optional[Path] = None) -> List[dict]:
    """Snapshots for this DB, newest first: {path, tag, label, when, size_kb}."""
    base = db_path if db_path is not None else DB_PATH
    d = _backups_dir(base)
    if not d.exists():
        return []
    out = []
    for p in d.glob(f"{base.stem}_*.db"):
        rest = p.stem[len(base.stem) + 1:]        # strip "<stem>_"
        tag = rest.split("_", 1)[0]               # "ch3" / "manual" / "prerestore"
        st = p.stat()
        out.append({
            "path": p,
            "tag": tag,
            "label": _pretty_tag(tag),
            "when": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "size_kb": max(1, st.st_size // 1024),
            "mtime": st.st_mtime,
        })
    out.sort(key=lambda b: b["mtime"], reverse=True)
    return out


def _prune_backups(base: Path) -> None:
    existing = sorted(_backups_dir(base).glob(f"{base.stem}_*.db"), key=lambda p: p.stat().st_mtime)
    for stale in existing[:-BACKUP_RETENTION]:
        try:
            stale.unlink()
        except OSError:
            pass


def create_backup(db_path: Optional[Path] = None, tag: str = "manual") -> Optional[Path]:
    """Checkpoint the WAL (so the copy has every committed change) and snapshot
    the DB into backups/, pruning to BACKUP_RETENTION. Returns the new path."""
    base = db_path if db_path is not None else DB_PATH
    if not base.exists():
        return None
    d = _backups_dir(base)
    d.mkdir(parents=True, exist_ok=True)
    try:
        conn = get_connection(base)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
    except sqlite3.Error:
        pass  # checkpoint is best-effort; copy still proceeds
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = d / f"{base.stem}_{tag}_{stamp}.db"
    shutil.copy2(base, dest)
    _prune_backups(base)
    return dest


def restore_backup(backup_path, db_path: Optional[Path] = None) -> None:
    """Overwrite the live DB with a snapshot. Saves the current DB first (as a
    'prerestore' snapshot) and clears WAL sidecars so SQLite can't replay stale
    frames onto the restored file."""
    base = db_path if db_path is not None else DB_PATH
    backup_path = Path(backup_path)
    if not backup_path.exists():
        raise FileNotFoundError(backup_path)
    if base.exists():
        create_backup(base, tag="prerestore")
    for suffix in ("-wal", "-shm"):
        side = base.parent / (base.name + suffix)
        if side.exists():
            try:
                side.unlink()
            except OSError:
                pass
    shutil.copy2(backup_path, base)


if __name__ == "__main__":
    init_db()
    print(f"Initialized DB at {DB_PATH}")
