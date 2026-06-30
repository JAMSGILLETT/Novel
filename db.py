"""
Minimal SQLite bootstrap.

This is intentionally NOT the full persistence layer (that's node 10).
It exists only so node 1 (input router) has something real to query for
cold-start detection, instead of faking the result.

As we build later nodes (context builder, persistence, etc.) this file
will grow tables for characters/plotlines/locations/lore. For now it has
just enough to support: "does this story have any chapters yet?"
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "novelgen.db"


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path if db_path is not None else DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> None:
    conn = get_connection(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chapters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            story_id TEXT NOT NULL,
            chapter_number INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(story_id, chapter_number)
        )
    """)
    conn.commit()
    conn.close()


def has_any_chapters(story_id: str, db_path: Path | None = None) -> bool:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT 1 FROM chapters WHERE story_id = ? LIMIT 1", (story_id,)
    ).fetchone()
    conn.close()
    return row is not None


def get_latest_chapter_number(story_id: str, db_path: Path | None = None) -> int | None:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT MAX(chapter_number) AS max_ch FROM chapters WHERE story_id = ?",
        (story_id,),
    ).fetchone()
    conn.close()
    return row["max_ch"] if row and row["max_ch"] is not None else None


if __name__ == "__main__":
    init_db()
    print(f"Initialized DB at {DB_PATH}")
