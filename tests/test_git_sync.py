import asyncio
import json
import subprocess
from pathlib import Path

from damselfish.config import GitSyncConfig
from damselfish.git_sync import GitMemorySync
from damselfish.store import Store, project_context_message


def _git(*arguments: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True
    )


def _sync_config(repository: Path, remote_env: str, device: str) -> GitSyncConfig:
    return GitSyncConfig(
        enabled=True,
        repository=repository,
        remote_url_env=remote_env,
        branch="main",
        pull_interval_seconds=0,
        device_id_env=f"UNUSED_DEVICE_{device}",
    )


def test_project_sessions_share_context(tmp_path: Path) -> None:
    store = Store(tmp_path / "memory.db", [])
    store.save_session(
        "architecture",
        [{"role": "user", "content": "Use FastAPI and SQLite"}],
        20,
        project_id="damselfish",
    )
    store.save_session(
        "deployment",
        [{"role": "user", "content": "Deploy on the cloud"}],
        20,
        project_id="damselfish",
    )

    context = store.project_context("damselfish", "deployment")
    message = project_context_message("damselfish", context, 1000)

    assert store.get_project_session("damselfish", "architecture", 30)
    assert len(store.list_project_sessions("damselfish")) == 2
    assert message and "FastAPI and SQLite" in message["content"]
    store.close()


def test_import_event_is_idempotent_and_keeps_longer_snapshot(tmp_path: Path) -> None:
    store = Store(tmp_path / "memory.db", [])
    event = {
        "event_id": "first",
        "project_id": "project",
        "session_id": "session",
        "created_at": 1.0,
        "source_device": "device-a",
        "messages": [{"role": "user", "content": "one"}],
    }
    assert store.import_memory_event(event)
    assert not store.import_memory_event(event)
    assert store.import_memory_event(
        {
            **event,
            "event_id": "second",
            "created_at": 2.0,
            "messages": [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ],
        }
    )
    assert len(store.get_project_session("project", "session", 30_000)) == 2
    store.close()


def test_git_sync_round_trip_between_devices(tmp_path: Path, monkeypatch) -> None:
    remote = tmp_path / "memory.git"
    remote.mkdir()
    _git("init", "--bare", cwd=remote)
    monkeypatch.setenv("REMOTE_A", str(remote))
    monkeypatch.setenv("REMOTE_B", str(remote))
    monkeypatch.setenv("UNUSED_DEVICE_A", "device-a")
    monkeypatch.setenv("UNUSED_DEVICE_B", "device-b")

    store_a = Store(tmp_path / "a.db", [])
    store_b = Store(tmp_path / "b.db", [])
    config_a = _sync_config(tmp_path / "repo-a", "REMOTE_A", "A")
    config_b = _sync_config(tmp_path / "repo-b", "REMOTE_B", "B")
    sync_a = GitMemorySync(config_a, store_a)
    sync_b = GitMemorySync(config_b, store_b)

    async def run() -> None:
        await sync_a.startup_sync()
        store_a.save_session(
            "session-a",
            [{"role": "user", "content": "from A"}],
            20,
            project_id="../../unsafe project",
            source_device=sync_a.device_id,
        )
        assert await sync_a.sync_pending(force=True)
        await sync_b.startup_sync()
        assert store_b.get_project_session("../../unsafe project", "session-a", 30)

        store_b.save_session(
            "session-b",
            [{"role": "user", "content": "from B"}],
            20,
            project_id="../../unsafe project",
            source_device=sync_b.device_id,
        )
        assert await sync_b.sync_pending(force=True)
        assert await sync_a.pull_if_due(force=True)
        assert store_a.get_project_session("../../unsafe project", "session-b", 30)

    asyncio.run(run())
    files = list((tmp_path / "repo-a" / "memory").rglob("*.json"))
    assert files
    assert all(tmp_path / "repo-a" in path.parents for path in files)
    assert json.loads(files[0].read_text())["project_id"] == "../../unsafe project"
    assert store_a.pending_memory_event_count() == 0
    assert store_b.pending_memory_event_count() == 0
    store_a.close()
    store_b.close()
