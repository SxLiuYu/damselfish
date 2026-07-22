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
    """Store handles legacy/unknown columns gracefully.

    The ``_target_stats_from_row`` helper filters row columns through
    ``_TARGET_STATS_FIELDS``, so extra columns from legacy schemas don't
    break loading.  We simulate a legacy DB with an extra unknown column.
    """
    path = tmp_path / "router.db"
    store = Store(path, ["fast"])
    store.close()

    # Simulate a legacy database with an extra unknown column
    with sqlite3.connect(path) as connection:
        connection.execute(
            "ALTER TABLE target_stats ADD COLUMN legacy_extra TEXT"
        )
        connection.execute(
            """
            UPDATE target_stats
            SET legacy_extra = 'old data'
            WHERE target_id = 'fast'
            """
        )

    store = Store(path, ["fast"])
    assert store.stats("fast").target_id == "fast"
    assert store.all_stats()["fast"].target_id == "fast"
    store.close()


def test_store_records_cap_count(tmp_path: Path) -> None:
    """record_cap increments the cap_count counter."""
    store = Store(tmp_path / "router.db", ["fast"])
    assert store.stats("fast").cap_count == 0
    store.record_cap("fast")
    store.record_cap("fast")
    assert store.stats("fast").cap_count == 2
    # cap_count is exposed in public() output
    assert store.stats("fast").public()["cap_count"] == 2
    store.close()


def test_store_migrates_cap_count_on_existing_db(tmp_path: Path) -> None:
    """Existing databases without cap_count get the column added on open."""
    path = tmp_path / "router.db"
    store = Store(path, ["fast"])
    store.close()

    # Simulate an old database by dropping the cap_count column
    with sqlite3.connect(path) as connection:
        # Recreate the table without cap_count to simulate legacy schema
        connection.execute("DROP TABLE target_stats")
        connection.execute(
            """
            CREATE TABLE target_stats (
                target_id TEXT PRIMARY KEY,
                requests INTEGER NOT NULL DEFAULT 0,
                successes INTEGER NOT NULL DEFAULT 0,
                failures INTEGER NOT NULL DEFAULT 0,
                rate_limits INTEGER NOT NULL DEFAULT 0,
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                ewma_latency_ms REAL,
                last_latency_ms REAL,
                last_success_at REAL,
                last_failure_at REAL,
                last_probe_at REAL,
                circuit_open_until REAL NOT NULL DEFAULT 0,
                last_error TEXT
            )
            """
        )
        connection.execute("INSERT INTO target_stats(target_id) VALUES ('fast')")

    # Reopening should add cap_count via migration
    store = Store(path, ["fast"])
    assert store.stats("fast").cap_count == 0
    store.record_cap("fast")
    assert store.stats("fast").cap_count == 1
    store.close()


def test_store_records_usage(tmp_path: Path) -> None:
    """record_usage accumulates token counts."""
    store = Store(tmp_path / "router.db", ["fast"])
    assert store.stats("fast").prompt_tokens == 0
    assert store.stats("fast").completion_tokens == 0
    assert store.stats("fast").total_tokens == 0
    store.record_usage("fast", prompt_tokens=100, completion_tokens=50, total_tokens=150)
    store.record_usage("fast", prompt_tokens=200, completion_tokens=80, total_tokens=280)
    assert store.stats("fast").prompt_tokens == 300
    assert store.stats("fast").completion_tokens == 130
    assert store.stats("fast").total_tokens == 430
    # Usage is exposed in public() output
    pub = store.stats("fast").public()
    assert pub["prompt_tokens"] == 300
    assert pub["completion_tokens"] == 130
    assert pub["total_tokens"] == 430
    store.close()


def test_store_record_usage_skips_zeros(tmp_path: Path) -> None:
    """record_usage does nothing when all values are zero."""
    store = Store(tmp_path / "router.db", ["fast"])
    store.record_usage("fast", prompt_tokens=0, completion_tokens=0, total_tokens=0)
    assert store.stats("fast").prompt_tokens == 0
    store.close()


def test_store_migrates_token_columns_on_existing_db(tmp_path: Path) -> None:
    """Existing databases without token columns get them added on open."""
    path = tmp_path / "router.db"
    store = Store(path, ["fast"])
    store.close()

    # Simulate an old database by dropping the token columns
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE target_stats")
        connection.execute(
            """
            CREATE TABLE target_stats (
                target_id TEXT PRIMARY KEY,
                requests INTEGER NOT NULL DEFAULT 0,
                successes INTEGER NOT NULL DEFAULT 0,
                failures INTEGER NOT NULL DEFAULT 0,
                rate_limits INTEGER NOT NULL DEFAULT 0,
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                ewma_latency_ms REAL,
                last_latency_ms REAL,
                last_success_at REAL,
                last_failure_at REAL,
                last_probe_at REAL,
                circuit_open_until REAL NOT NULL DEFAULT 0,
                last_error TEXT,
                cap_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute("INSERT INTO target_stats(target_id) VALUES ('fast')")

    # Reopening should add token columns via migration
    store = Store(path, ["fast"])
    assert store.stats("fast").prompt_tokens == 0
    assert store.stats("fast").completion_tokens == 0
    assert store.stats("fast").total_tokens == 0
    store.record_usage("fast", prompt_tokens=50, completion_tokens=25, total_tokens=75)
    assert store.stats("fast").prompt_tokens == 50
    store.close()
