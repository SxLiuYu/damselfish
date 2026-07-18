from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import AppConfig, TargetConfig
from .selector import RouteContext, rank_targets
from .store import Store

log = logging.getLogger("damselfish.router")


@dataclass(slots=True)
class CompletionResult:
    body: dict[str, Any]
    target: TargetConfig
    latency_ms: float


class UpstreamFailure(Exception):
    def __init__(self, target: TargetConfig, status: int, message: str) -> None:
        super().__init__(message)
        self.target = target
        self.status = status


class NoTargetAvailable(Exception):
    pass


class ModelRouter:
    def __init__(
        self, config: AppConfig, store: Store, client: httpx.AsyncClient
    ) -> None:
        self.config = config
        self.store = store
        self.client = client
        self._semaphores = {
            target.id: asyncio.Semaphore(target.max_concurrency)
            for target in config.targets
        }

    def reconfigure(self, config: AppConfig) -> None:
        self.config = config
        self._semaphores = {
            target.id: self._semaphores.get(
                target.id, asyncio.Semaphore(target.max_concurrency)
            )
            for target in config.targets
        }

    async def complete(
        self,
        payload: dict[str, Any],
        context: RouteContext,
        session_id: str | None,
    ) -> CompletionResult:
        targets = rank_targets(
            self.config,
            context,
            self.store.all_stats(),
            str(payload.get("model", "auto")),
        )
        if not targets:
            raise NoTargetAvailable(
                f"no healthy target has required capabilities: {sorted(context.required)}"
            )

        failures: list[str] = []
        for target in targets:
            try:
                result = await self._call(target, payload)
            except UpstreamFailure as error:
                failures.append(f"{target.id}: HTTP {error.status} {error}")
                self.store.record_decision(
                    session_id, context.scenario, context.persona, target.id,
                    None, False, error.status, str(error),
                )
                continue
            self.store.record_decision(
                session_id, context.scenario, context.persona, target.id,
                result.latency_ms, True,
            )
            log.info(
                "route scenario=%s persona=%s target=%s latency_ms=%.1f",
                context.scenario, context.persona or "-", target.id, result.latency_ms,
            )
            return result
        raise NoTargetAvailable("all matching targets failed: " + "; ".join(failures))

    async def _call(
        self, target: TargetConfig, payload: dict[str, Any], probe: bool = False
    ) -> CompletionResult:
        request = _upstream_payload(payload, target, probe)
        headers = {"Content-Type": "application/json"}
        if target.api_key:
            headers["Authorization"] = f"Bearer {target.api_key}"
        started = time.monotonic()
        try:
            async with self._semaphores[target.id]:
                response = await self.client.post(
                    target.chat_url, headers=headers, json=request
                )
            latency_ms = (time.monotonic() - started) * 1000
            if response.status_code < 200 or response.status_code >= 300:
                raise UpstreamFailure(
                    target, response.status_code, _error_message(response)
                )
            body = response.json()
            if isinstance(body.get("data"), dict) and "choices" in body["data"]:
                body = body["data"]
            _validate_completion(body)
        except UpstreamFailure as error:
            self._record_failure(target, error.status, str(error), probe)
            raise
        except httpx.TimeoutException as error:
            failure = UpstreamFailure(target, 504, f"timeout: {error}")
            self._record_failure(target, failure.status, str(failure), probe)
            raise failure from error
        except (httpx.HTTPError, ValueError, TypeError) as error:
            failure = UpstreamFailure(target, 502, f"invalid upstream response: {error}")
            self._record_failure(target, failure.status, str(failure), probe)
            raise failure from error
        self.store.record_success(
            target.id, latency_ms, self.config.routing.ewma_alpha, probe
        )
        body["model"] = target.model
        return CompletionResult(body=body, target=target, latency_ms=latency_ms)

    def _record_failure(
        self, target: TargetConfig, status: int, message: str, probe: bool
    ) -> None:
        state = self.store.stats(target.id)
        count = state.consecutive_failures + 1
        if status == 429:
            delay = min(
                self.config.routing.circuit_base_seconds * (2 ** max(count, 1)),
                self.config.routing.circuit_max_seconds,
            )
        elif count >= self.config.routing.circuit_failures:
            delay = min(
                self.config.routing.circuit_base_seconds * count,
                self.config.routing.circuit_max_seconds,
            )
        else:
            delay = 0
        self.store.record_failure(
            target.id, status, message, time.time() + delay, probe
        )
        log.warning(
            "target=%s status=%s circuit_seconds=%.0f error=%s",
            target.id, status, delay, message[:200],
        )

    async def probe(self, target: TargetConfig) -> None:
        if not target.available or not target.probe:
            return
        state = self.store.stats(target.id)
        if state.circuit_open_until > time.time():
            return
        payload = {
            "messages": [{"role": "user", "content": target.probe_prompt}],
            "max_tokens": 4,
        }
        try:
            await self._call(target, payload, probe=True)
        except UpstreamFailure:
            return

    async def probe_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            stats = self.store.all_stats()
            now = time.time()
            stale = [
                target
                for target in self.config.targets
                if target.probe
                and target.available
                and now - (stats[target.id].last_probe_at or 0)
                >= self.config.routing.probe_stale_seconds
            ]
            if stale:
                await asyncio.gather(*(self.probe(target) for target in stale))
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=self.config.routing.probe_interval_seconds
                )
            except TimeoutError:
                pass


UPSTREAM_FIELDS = {
    "messages", "tools", "tool_choice", "temperature", "top_p", "max_tokens",
    "max_completion_tokens", "stop", "response_format", "seed",
    "presence_penalty", "frequency_penalty", "parallel_tool_calls", "n", "user",
}


def _upstream_payload(
    payload: dict[str, Any], target: TargetConfig, probe: bool
) -> dict[str, Any]:
    request = {key: value for key, value in payload.items() if key in UPSTREAM_FIELDS}
    request["model"] = target.model
    request["stream"] = False
    if probe:
        request.pop("tools", None)
        request.pop("tool_choice", None)
    return request


def _validate_completion(body: Any) -> None:
    if not isinstance(body, dict):
        raise ValueError("response is not a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("response has no assistant message")
    usable = message.get("content") or message.get("tool_calls") or message.get("function_call")
    if not usable:
        raise ValueError("assistant message has no usable content or tool call")


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
        error = body.get("error", body)
        if isinstance(error, dict):
            return str(error.get("message", error))[:500]
        return str(error)[:500]
    except (ValueError, TypeError):
        return response.text[:500]
