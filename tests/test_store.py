import sqlite3
from pathlib import Path

from damselfish.store import Store, merge_messages


def test_merge_messages_uses_shared_history() -> None:
    stored = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    incoming = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    assert merge_messages(stored, incoming) == incoming


def test_store_persists_stats_and_memory(tmp_path: Path) -> None:
    path = tmp_path / "router.db"
    store = Store(path, ["fast"])
    store.record_success("fast", 100, 0.5)
    store.record_success("fast", 200, 0.5)
    store.save_session("session", [{"role": "user", "content": "hello"}], 10)
    assert store.stats("fast").ewma_latency_ms == 150
    assert store.get_session("session", 1)[0]["content"] == "hello"
    store.close()


def test_store_ignores_historical_target_stats_columns(tmp_path: Path) -> None:
    path = tmp_path / "router.db"
    store = Store(path, ["fast"])
    store.close()

    with sqlite3.connect(path) as connection:
        connection.execute(
            "ALTER TABLE target_stats ADD COLUMN prompt_tokens INTEGER NOT NULL DEFAULT 0"
        )
        connection.execute(
            "ALTER TABLE target_stats ADD COLUMN completion_tokens INTEGER NOT NULL DEFAULT 0"
        )
        connection.execute(
            "ALTER TABLE target_stats ADD COLUMN total_tokens INTEGER NOT NULL DEFAULT 0"
        )
        connection.execute(
            """
            UPDATE target_stats
            SET prompt_tokens = 10, completion_tokens = 5, total_tokens = 15
            WHERE target_id = 'fast'
            """
        )

    store = Store(path, ["fast"])
    assert store.stats("fast").target_id == "fast"
    assert store.all_stats()["fast"].target_id == "fast"
    store.close()
