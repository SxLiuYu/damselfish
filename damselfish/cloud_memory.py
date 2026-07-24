# -*- coding: utf-8 -*-
"""Cloud memory sync for cross-device session continuity.

When enabled, memories and sessions are pushed to a remote REST API
so that any device can resume conversations seamlessly.

Usage in config.yml::

    cloud_memory:
      enabled: true
      url: "https://123.57.107.21:8088/damselfish"
      api_key: "your-api-key-here"
      push_interval: 30
      pull_interval: 30
      device_id_env: "DAMSELFISH_DEVICE_ID"
      max_workers: 4
      batch_size: 100
      decision_history_limit: 200
      max_snapshot_chars: 50000
      mode: "push_pull"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("damselfish.cloud_memory")


@dataclass(frozen=True, slots=True)
class CloudMemoryConfig:
    """Configuration for the cloud memory sync service."""
    enabled: bool = False
    url: str = ""
    api_key: str = ""
    api_key_env: str | None = None
    push_interval: float = 30.0
    pull_interval: float = 30.0
    device_id_env: str = "DAMSELFISH_DEVICE_ID"
    max_workers: int = 4
    batch_size: int = 100
    decision_history_limit: int = 200
    max_snapshot_chars: int = 50000
    mode: str = "push_pull"

    @property
    def api_key_value(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""

    @property
    def device_id(self) -> str:
        return os.environ.get(self.device_id_env, "") or str(uuid.uuid4())[:8]

    @property
    def base_url(self) -> str:
        url = self.url.rstrip("/")
        return url if url.endswith("/") else f"{url}/"


class CloudMemorySync:
    """Async sync engine for pushing/pulling memories across devices."""

    def __init__(
        self,
        config: CloudMemoryConfig,
        store,
    ) -> None:
        self.config = config
        self.store = store
        self._last_push_at: float = 0.0
        self._last_pull_at: float = 0.0
        self._stop_event: asyncio.Event | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._last_pulled_timestamp: float = 0.0
        self._started_at: float = 0.0

    async def start(self) -> None:
        if not self.config.enabled:
            log.info("cloud_memory disabled, skipping")
            return
        timeout = httpx.Timeout(10.0, connect=5.0)
        self._http_client = httpx.AsyncClient(
            timeout=timeout,
            verify=False,
        )
        self._stop_event = asyncio.Event()
        self._started_at = time.time()
        try:
            await self.push_pending()
        except Exception:
            log.warning("initial cloud push failed, will retry on next interval")
        log.info(
            "cloud_memory started: device=%s url=%s mode=%s",
            self.config.device_id, self.config.base_url, self.config.mode,
        )
        push_task = asyncio.create_task(self._push_loop(self._stop_event))
        pull_task = asyncio.create_task(self._pull_loop(self._stop_event))
        self._tasks = [push_task, pull_task]

    async def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self.store and self._http_client:
            try:
                await self.push_pending()
            except Exception:
                log.warning("final cloud push on shutdown failed")
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        for t in getattr(self, "_tasks", []):
            t.cancel()
        try:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass

    async def push_pending(self) -> dict[str, Any]:
        """Push unsynced memory events to the remote server."""
        if not self._http_client or not self.config.enabled:
            return {"status": "skipped", "reason": "not enabled"}
        pending = self.store.pending_memory_events()
        if not pending:
            return {"status": "ok", "pushed": 0, "message": "nothing to push"}
        device_id = self.config.device_id
        payload = {
            "device_id": device_id,
            "events": [
                {
                    "event_id": e["event_id"],
                    "project_id": e["project_id"],
                    "session_id": e["session_id"],
                    "created_at": e["created_at"],
                    "source_device": e.get("source_device", device_id),
                    "snapshot_json": e["snapshot_json"],
                }
                for e in pending[:100]
            ],
        }
        now = time.time()
        result = await self._remote_post(
            "api/memory/events", payload, now
        )
        return result

    async def pull_remote(self) -> dict[str, Any]:
        """Pull new memory events from the remote server."""
        if not self._http_client or not self.config.enabled:
            return {"status": "skipped", "reason": "not enabled"}
        device_id = self.config.device_id
        params = {
            "since": self._last_pulled_timestamp,
            "batch_size": self.config.batch_size,
        }
        response = await self._remote_get("api/memory/events", params)
        if response.get("status") != "ok" or not response.get("events"):
            return {"status": "ok", "pulled": 0}
        events = response["events"]
        imported = 0
        for event in events:
            source = event.get("source_device", "")
            if source == device_id:
                continue
            ok = self.store.import_memory_event(event)
            if ok:
                imported += 1
            else:
                log.debug("import conflict, skipping event %s", event.get("event_id"))
        if events:
            latest_ts = max(e.get("created_at", 0) for e in events)
            self._last_pulled_timestamp = latest_ts
        log.info("pulled %d events from cloud (total since=%s)", imported, self._last_pulled_timestamp)
        return {"status": "ok", "pulled": imported}

    async def status(self) -> dict[str, Any]:
        """Return current sync status for monitoring."""
        pending_count = self.store.pending_memory_event_count() if self.store else 0
        return {
            "enabled": self.config.enabled,
            "device_id": self.config.device_id,
            "base_url": self.config.base_url,
            "mode": self.config.mode,
            "pending_local": pending_count,
            "last_push_at": self._last_push_at,
            "last_pull_at": self._last_pull_at,
            "uptime_seconds": time.time() - getattr(self, "_started_at", 0),
        }

    async def health_check(self) -> bool:
        """Check if remote server is reachable."""
        if not self._http_client:
            return False
        try:
            resp = await self._remote_get("health")
            return resp.get("status") == "ok"
        except Exception:
            return False

    async def _push_loop(self, stop: asyncio.Event) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=self.config.push_interval
                )
                break
            except TimeoutError:
                pass
            if not self.config.enabled:
                continue
            try:
                result = await self.push_pending()
                self._last_push_at = time.time()
                if result.get("pushed", 0) > 0:
                    log.info("pushed %d events to cloud", result["pushed"])
            except Exception:
                log.warning("cloud push failed", exc_info=True)

    async def _pull_loop(self, stop: asyncio.Event) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=self.config.pull_interval
                )
                break
            except TimeoutError:
                pass
            if not self.config.enabled:
                continue
            try:
                result = await self.pull_remote()
                self._last_pull_at = time.time()
                pulled = result.get("pulled", 0)
                if pulled > 0:
                    log.info("pulled %d events from cloud", pulled)
            except Exception:
                log.warning("cloud pull failed", exc_info=True)

    async def _remote_get(
        self, path: str, params: dict | None = None
    ) -> dict[str, Any]:
        if not self._http_client:
            raise RuntimeError("client not initialized")
        headers = {
            "Authorization": f"Bearer {self.config.api_key_value}",
            "Content-Type": "application/json",
        }
        url = f"{self.config.base_url}{path.lstrip('/')}"
        resp = await self._http_client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _remote_post(
        self, path: str, payload: dict[str, Any], ts: float
    ) -> dict[str, Any]:
        if not self._http_client:
            raise RuntimeError("client not initialized")
        headers = {
            "Authorization": f"Bearer {self.config.api_key_value}",
            "Content-Type": "application/json",
        }
        url = f"{self.config.base_url}{path.lstrip('/')}"
        resp = await self._http_client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()
        pushed = len(body.get("events", []))
        return {"status": "ok", "pushed": pushed}


def cloud_memory_config_from_raw(raw: dict[str, Any]) -> CloudMemoryConfig:
    """Parse cloud_memory section from YAML config dict."""
    if not raw:
        return CloudMemoryConfig()
    valid_keys = {f.name for f in fields(CloudMemoryConfig)}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return CloudMemoryConfig(**filtered)
