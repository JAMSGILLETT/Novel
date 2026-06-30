"""
Test for node_input_router.py.

Uses a throwaway SQLite file (not the real novelgen.db) so this doesn't
pollute actual story data. Exercises all three classification branches.

Run: python test_input_router.py
"""
from pathlib import Path
import sqlite3

from schema import ChapterGraphState
import db as db_module
from node_input_router import input_router_node

TEST_DB = Path(__file__).parent / "test_novelgen.db"


def reset_test_db():
    if TEST_DB.exists():
        TEST_DB.unlink()
    db_module.init_db(TEST_DB)


def seed_chapter(story_id: str, chapter_number: int):
    conn = db_module.get_connection(TEST_DB)
    conn.execute(
        "INSERT INTO chapters (story_id, chapter_number) VALUES (?, ?)",
        (story_id, chapter_number),
    )
    conn.commit()
    conn.close()


def patch_db_path(monkeypatch_target=TEST_DB):
    """Point db.py's module-level DB_PATH at the test DB for this run."""
    db_module.DB_PATH = monkeypatch_target


def test_cold_start():
    reset_test_db()
    patch_db_path()
    state = ChapterGraphState(
        story_id="story-cold",
        chapter_number=0,  # not yet known, router will set it
        user_input="Begin the story.",
    )
    # input_router_node calls has_any_chapters(state.story_id) with default db_path=DB_PATH
    # which we've patched above.
    result = input_router_node(state)
    assert result["input_mode"] == "cold_start", result
    assert result["chapter_number"] == 1, result
    print("Cold start OK:", result)


def test_continuation():
    reset_test_db()
    patch_db_path()
    seed_chapter("story-cont", 1)
    seed_chapter("story-cont", 2)
    state = ChapterGraphState(
        story_id="story-cont",
        chapter_number=0,
        user_input="continue",
    )
    result = input_router_node(state)
    assert result["input_mode"] == "continuation", result
    assert result["chapter_number"] == 3, result
    print("Continuation OK:", result)


def test_user_event_injection():
    reset_test_db()
    patch_db_path()
    seed_chapter("story-inj", 1)
    state = ChapterGraphState(
        story_id="story-inj",
        chapter_number=0,
        user_input="Have a hooded stranger ambush Alice on her way out of the market.",
    )
    result = input_router_node(state)
    assert result["input_mode"] == "user_event_injection", result
    assert result["chapter_number"] == 2, result
    print("User event injection OK:", result)


if __name__ == "__main__":
    test_cold_start()
    test_continuation()
    test_user_event_injection()
    TEST_DB.unlink(missing_ok=True)
    print("\nAll input router tests passed.")
