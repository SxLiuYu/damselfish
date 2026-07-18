from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import GitSyncConfig
from .store import Store

log = logging.getLogger("damselfish.git_sync")


class GitMemorySync:
    def __init__(self, config: GitSyncConfig, store: Store) -> None:
        self.config = config
        self.store = store
        self.repository = config.repository.expanduser().resolve()
        self.device_id = config.device_id.strip() or _default_device_id()
        self._lock = asyncio.Lock()
        self._last_pull_at = 0.0
        self._last_push_at = 0.0
        self._last_error: str | None = None
        self._imported_events = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    async def startup_sync(self) -> None:
        if not self.enabled:
            return
        async with self._lock:
            await asyncio.to_thread(self._startup_sync)

    async def pull_if_due(self, force: bool = False) -> bool:
        if not self.enabled:
            return False
        if not force and time.time() - self._last_pull_at < self.config.pull_interval_seconds:
            return True
        async with self._lock:
            if not force and time.time() - self._last_pull_at < self.config.pull_interval_seconds:
                return True
            return await asyncio.to_thread(self._pull_and_import)

    async def sync_pending(self, force: bool = False) -> bool:
        if not self.enabled or (not force and not self.config.push_on_write):
            return False
        async with self._lock:
            return await asyncio.to_thread(self._sync_pending)

    async def sync_now(self) -> bool:
        if not self.enabled:
            return False
        async with self._lock:
            pulled = await asyncio.to_thread(self._pull_and_import)
            pushed = await asyncio.to_thread(self._sync_pending)
            return pulled and pushed

    async def sync_loop(self, stop: asyncio.Event) -> None:
        if not self.enabled:
            return
        interval = max(min(self.config.pull_interval_seconds, 30.0), 1.0)
        while not stop.is_set():
            await self.pull_if_due()
            await self.sync_pending(force=True)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except TimeoutError:
                pass

    def status(self) -> dict[str, Any]:
        pending = self.store.pending_memory_event_count() if self.enabled else 0
        return {
            "enabled": self.enabled,
            "device_id": self.device_id,
            "repository": str(self.repository) if self.enabled else None,
            "pending_events": pending,
            "last_pull_at": self._last_pull_at or None,
            "last_push_at": self._last_push_at or None,
            "last_error": self._last_error,
            "imported_events": self._imported_events,
        }

    def response_status(self) -> str:
        if not self.enabled:
            return "disabled"
        return "synced" if self.store.pending_memory_event_count() == 0 else "pending"

    def _startup_sync(self) -> None:
        try:
            self._ensure_repository()
            self._pull_and_import(ensure=False)
            self._sync_pending(ensure=False)
        except Exception as error:
            self._record_error(error)

    def _pull_and_import(self, ensure: bool = True) -> bool:
        try:
            if ensure:
                self._ensure_repository()
            if self._has_remote():
                self._pull_remote()
            imported = self._import_events()
            self._imported_events += imported
            self._last_pull_at = time.time()
            self._last_error = None
            return True
        except Exception as error:
            self._record_error(error)
            return False

    def _sync_pending(self, ensure: bool = True) -> bool:
        try:
            if ensure:
                self._ensure_repository()
            events = self.store.pending_memory_events()
            if not events:
                return True
            for event in events:
                self._write_event(event)
            self._run("add", "--", "memory", ".gitignore")
            staged = self._run("diff", "--cached", "--quiet", check=False)
            if staged.returncode == 1:
                self._run(
                    "commit",
                    "-m",
                    f"memory: sync {self.device_id} {len(events)} events",
                )
            elif staged.returncode != 0:
                raise RuntimeError(staged.stderr.strip() or "unable to inspect staged memory")

            if not self._has_remote():
                self._last_error = "memory repository has no Git remote"
                return False

            pushed = False
            attempts = max(self.config.push_retries, 1)
            for attempt in range(attempts):
                result = self._run(
                    "push",
                    "--set-upstream",
                    "origin",
                    f"HEAD:{self.config.branch}",
                    check=False,
                )
                if result.returncode == 0:
                    pushed = True
                    break
                if attempt + 1 < attempts:
                    self._pull_remote()
                    self._imported_events += self._import_events()
            if not pushed:
                raise RuntimeError(result.stderr.strip() or "Git push failed")

            self.store.mark_memory_events_synced(
                [str(event["event_id"]) for event in events]
            )
            self._last_push_at = time.time()
            self._last_error = None
            return True
        except Exception as error:
            self._record_error(error)
            return False

    def _ensure_repository(self) -> None:
        self.repository.mkdir(parents=True, exist_ok=True)
        if not (self.repository / ".git").is_dir():
            self._run("init", "-b", self.config.branch)
        self._run("config", "user.name", self.config.author_name)
        self._run("config", "user.email", self.config.author_email)
        self._run("config", "rebase.autoStash", "true")

        remote_url = self.config.remote_url.strip()
        remotes = self._run("remote", check=False).stdout.split()
        if remote_url and "origin" not in remotes:
            self._run("remote", "add", "origin", remote_url)
        elif remote_url:
            current = self._run("remote", "get-url", "origin").stdout.strip()
            if current != remote_url:
                self._run("remote", "set-url", "origin", remote_url)

        ignore_path = self.repository / ".gitignore"
        expected = ".DS_Store\n*.tmp\n"
        if not ignore_path.exists():
            ignore_path.write_text(expected, encoding="utf-8")

    def _has_remote(self) -> bool:
        return "origin" in self._run("remote", check=False).stdout.split()

    def _pull_remote(self) -> None:
        fetch = self._run("fetch", "origin", self.config.branch, check=False)
        if fetch.returncode != 0:
            message = fetch.stderr.lower()
            if "couldn't find remote ref" in message or "remote ref does not exist" in message:
                return
            raise RuntimeError(fetch.stderr.strip() or "Git fetch failed")

        remote_ref = f"refs/remotes/origin/{self.config.branch}"
        if self._run("rev-parse", "--verify", remote_ref, check=False).returncode != 0:
            return
        if self._run("rev-parse", "--verify", "HEAD", check=False).returncode != 0:
            self._run("reset", "--hard", remote_ref)
            self._run("checkout", "-B", self.config.branch)
            return
        rebase = self._run("rebase", remote_ref, check=False)
        if rebase.returncode != 0:
            self._run("rebase", "--abort", check=False)
            raise RuntimeError(rebase.stderr.strip() or "Git rebase failed")

    def _write_event(self, event: dict[str, Any]) -> Path:
        project = _safe_segment(str(event["project_id"]))
        session = _safe_segment(str(event["session_id"]))
        source = _safe_segment(str(event["source_device"]))
        event_id = _safe_segment(str(event["event_id"]))
        timestamp = datetime.fromtimestamp(
            float(event["created_at"]), timezone.utc
        ).strftime("%Y%m%dT%H%M%S%fZ")
        directory = self.repository / "memory" / "projects" / project / "sessions" / session
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{timestamp}-{source}-{event_id}.json"
        if destination.exists():
            return destination
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(event, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def _import_events(self) -> int:
        root = self.repository / "memory" / "projects"
        if not root.is_dir():
            return 0
        imported = 0
        for path in sorted(root.glob("*/sessions/*/*.json")):
            if path.is_symlink() or path.stat().st_size > 5_000_000:
                continue
            try:
                event = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(event, dict):
                    raise ValueError("memory event must be an object")
                imported += int(self.store.import_memory_event(event))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                log.warning("skipping invalid memory event %s: %s", path.name, error)
        return imported

    def _run(
        self, *arguments: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["GIT_TERMINAL_PROMPT"] = "0"
        result = subprocess.run(
            ["git", *arguments],
            cwd=self.repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if check and result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"Git command failed: {arguments[0]}")
        return result

    def _record_error(self, error: Exception) -> None:
        message = str(error)
        if self.config.remote_url:
            message = message.replace(self.config.remote_url, "<remote>")
        self._last_error = message[:1000]
        log.warning("memory sync failed: %s", self._last_error)


def _safe_segment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_")[:48]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{normalized or 'item'}-{digest}"


def _default_device_id() -> str:
    identity = f"{socket.gethostname()}:{socket.getfqdn()}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
    hostname = _safe_segment(socket.gethostname()).rsplit("-", 1)[0]
    return f"{hostname}-{digest}"
