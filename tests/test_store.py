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
