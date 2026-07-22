from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .config import AppConfig, TargetConfig
from .selector import RouteContext, rank_targets
from .store import Store
from .tokens import estimate_text_tokens, estimate_messages_tokens

log = logging.getLogger("damselfish.router")

# Maximum number of cached responses and TTL in seconds.
_CACHE_MAX = 128
_CACHE_TTL = 30.0


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
        self._raced_ids: set[str] = set()
        # Short-lived in-memory cache for identical requests (dedup).
        # Keyed by hashed payload, value is a CompletionResult.
        self._cache: OrderedDict[str, CompletionResult] = OrderedDict()

    def reconfigure(self, config: AppConfig) -> None:
        self.config = config
        self._semaphores = {
            target.id: self._semaphores.get(
                target.id, asyncio.Semaphore(target.max_concurrency)
            )
            for target in config.targets
        }
        self._cache.clear()

    @staticmethod
    def _cache_key(payload: dict[str, Any]) -> str:
        """Build a hash key from the request payload for dedup."""
        import hashlib
        # Only hash the fields that affect the upstream response.
        relevant = {
            k: v for k, v in payload.items()
            if k in ("messages", "tools", "tool_choice", "temperature", "top_p",
                     "max_tokens", "max_completion_tokens", "stop", "response_format",
                     "seed", "presence_penalty", "frequency_penalty", "n", "user")
        }
        raw = json.dumps(relevant, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, payload: dict[str, Any]) -> CompletionResult | None:
        key = self._cache_key(payload)
        entry = self._cache.get(key)
        if entry is None:
            return None
        result, ts = entry
        if time.time() - ts > _CACHE_TTL:
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return result

    def _cache_put(self, payload: dict[str, Any], result: CompletionResult) -> None:
        key = self._cache_key(payload)
        self._cache[key] = (result, time.time())
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)

    async def complete(
        self,
        payload: dict[str, Any],
        context: RouteContext,
        session_id: str | None,
    ) -> CompletionResult:
        # Short-lived cache hit for identical requests (dedup within 30s).
        cached = self._cache_get(payload)
        if cached is not None:
            log.info("cache hit for %s, returning cached result", context.scenario)
            return cached

        targets = rank_targets(
            self.config,
            context,
            self.store.all_stats(),
            str(payload.get("model", "auto")),
            max_new_tokens=_max_new_tokens(payload),
        )
        if not targets:
            raise NoTargetAvailable(
                f"no healthy target has required capabilities: {sorted(context.required)}"
            )

        # Phase 1: serial attempt on the best target. On 429/504 (rate limit /
        # timeout) we fall through to Phase 2 and race the remaining candidates
        # in parallel, returning the first successful response. If every parallel
        # candidate fails we fall back to serial retry on the leftover targets.
        primary = targets[0]
        try:
            result = await self._call(primary, payload)
        except UpstreamFailure as error:
            self.store.record_decision(
                session_id, context.scenario, context.persona, primary.id,
                None, False, error.status, str(error),
            )
            if (error.status not in (429, 504) and not _is_context_overflow(error)) or len(targets) < 2:
                raise NoTargetAvailable(
                    f"primary target {primary.id} failed: HTTP {error.status} {error}"
                ) from error
            result = await self._race_targets(
                targets[1:], payload, context, session_id,
            )
            if result is None:
                # All parallel candidates failed; try the rest serially.
                suffix = [
                    t for t in targets[1:]
                    if t.id not in self._raced_ids
                ]
                result = await self._serial_fallback(
                    suffix, payload, context, session_id,
                )
        self.store.record_decision(
            session_id, context.scenario, context.persona, result.target.id,
            result.latency_ms, True,
        )
        log.info(
            "route scenario=%s persona=%s target=%s latency_ms=%.1f",
            context.scenario, context.persona or "-", result.target.id, result.latency_ms,
        )
        self._cache_put(payload, result)
        return result

    async def stream_complete(
        self,
        payload: dict[str, Any],
        context: RouteContext,
        session_id: str | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming version of ``complete()``.

        Yields normalized SSE chunks.  On 429/504 before the first chunk,
        falls through to parallel race.  After the stream ends, the caller
        can read ``self._stream_result`` for the final ``CompletionResult``.
        """
        targets = rank_targets(
            self.config,
            context,
            self.store.all_stats(),
            str(payload.get("model", "auto")),
            max_new_tokens=_max_new_tokens(payload),
        )
        if not targets:
            raise NoTargetAvailable(
                f"no healthy target has required capabilities: {sorted(context.required)}"
            )

        self._stream_result: CompletionResult | None = None
        primary = targets[0]
        iterator = self._stream_call(primary, payload)
        try:
            first_chunk = await iterator.__anext__()
        except StopAsyncIteration:
            raise NoTargetAvailable(f"primary target {primary.id} returned empty stream")
        except UpstreamFailure as error:
            self.store.record_decision(
                session_id, context.scenario, context.persona, primary.id,
                None, False, error.status, str(error),
            )
            if (error.status not in (429, 504) and not _is_context_overflow(error)) or len(targets) < 2:
                raise NoTargetAvailable(
                    f"primary target {primary.id} failed: HTTP {error.status} {error}"
                ) from error
            # Phase 2: parallel race
            result = await self._race_stream(
                targets[1:], payload, context, session_id,
            )
            if result is None:
                # Phase 3: serial fallback on leftovers
                suffix = [t for t in targets[1:] if t.id not in self._raced_ids]
                result = await self._serial_fallback(suffix, payload, context, session_id)
            self._stream_result = result
            self.store.record_decision(
                session_id, context.scenario, context.persona, result.target.id,
                result.latency_ms, True,
            )
            log.info(
                "route scenario=%s persona=%s target=%s latency_ms=%.1f (stream fallback)",
                context.scenario, context.persona or "-", result.target.id, result.latency_ms,
            )
            # Non-streaming fallback result → single SSE chunk
            yield _normalize_stream_chunk(result.body, result.target.model)
            return

        # Phase 1 succeeded: yield first chunk, then continue streaming
        yield first_chunk
        async for chunk in iterator:
            yield chunk
        self._stream_result = CompletionResult(body={}, target=primary, latency_ms=0)
        self.store.record_decision(
            session_id, context.scenario, context.persona, primary.id,
            self._stream_result.latency_ms, True,
        )
        log.info(
            "route scenario=%s persona=%s target=%s latency_ms=%.1f (stream)",
            context.scenario, context.persona or "-", primary.id, self._stream_result.latency_ms,
        )

    async def _race_targets(
        self,
        candidates: list[TargetConfig],
        payload: dict[str, Any],
        context: RouteContext,
        session_id: str | None,
    ) -> CompletionResult | None:
        """Race up to ``parallel_fallback_count`` candidates in parallel.

        Returns the first successful ``CompletionResult`` and cancels the rest.
        Records a decision row for every attempted target and tracks the ids in
        ``self._raced_ids`` so the caller can serially retry the leftovers.
        Returns ``None`` if every parallel attempt fails or the race times out.
        """
        limit = max(1, self.config.routing.parallel_fallback_count)
        racing = candidates[:limit]
        if not racing:
            return None
        self._raced_ids = {target.id for target in racing}
        timeout = self.config.routing.parallel_fallback_timeout_seconds

        tasks: dict[asyncio.Task[CompletionResult], TargetConfig] = {}
        for target in racing:
            tasks[asyncio.create_task(self._call(target, payload))] = target

        pending = set(tasks)
        last_failure: UpstreamFailure | None = None
        failures: list[str] = []
        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    # Timed out waiting for any winner; cancel the rest and
                    # fall back to serial handling of the remaining candidates.
                    log.warning(
                        "parallel fallback timed out after %.1fs; tried %s",
                        timeout, ", ".join(t.id for t in racing),
                    )
                    return None
                for task in done:
                    target = tasks[task]
                    try:
                        result = task.result()
                    except UpstreamFailure as error:
                        last_failure = error
                        failures.append(
                            f"{target.id}: HTTP {error.status} {error}"
                        )
                        self.store.record_decision(
                            session_id, context.scenario, context.persona,
                            target.id, None, False, error.status, str(error),
                        )
                        continue
                    except Exception as error:  # pragma: no cover - defensive
                        failures.append(f"{target.id}: {error}")
                        self.store.record_decision(
                            session_id, context.scenario, context.persona,
                            target.id, None, False, 502, str(error),
                        )
                        continue
                    # Winner: cancel the remaining tasks and return.
                    for leftover in pending:
                        leftover.cancel()
                    return result
            log.warning(
                "all parallel fallback targets failed: %s",
                "; ".join(failures),
            )
            return None
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _race_stream(
        self,
        candidates: list[TargetConfig],
        payload: dict[str, Any],
        context: RouteContext,
        session_id: str | None,
    ) -> CompletionResult | None:
        """Race up to ``parallel_fallback_count`` streaming candidates.

        Returns the first candidate's ``CompletionResult`` and sets
        ``self._stream_result`` to the winner.  Returns ``None`` if every
        parallel attempt fails or times out.
        """
        limit = max(1, self.config.routing.parallel_fallback_count)
        racing = candidates[:limit]
        if not racing:
            return None
        self._raced_ids = {target.id for target in racing}
        timeout = self.config.routing.parallel_fallback_timeout_seconds

        first_chunk_tasks: dict[asyncio.Task, TargetConfig] = {}
        iterators: dict[TargetConfig, AsyncIterator[dict]] = {}
        for target in racing:
            iterator = self._stream_call(target, payload)
            iterators[target] = iterator
            task = asyncio.create_task(iterator.__anext__())
            first_chunk_tasks[task] = target

        pending = set(first_chunk_tasks)
        failures: list[str] = []
        winner_target: TargetConfig | None = None
        try:
            while pending and winner_target is None:
                done, pending = await asyncio.wait(
                    pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    log.warning(
                        "parallel stream race timed out after %.1fs; tried %s",
                        timeout, ", ".join(t.id for t in racing),
                    )
                    break
                for task in done:
                    target = first_chunk_tasks[task]
                    try:
                        task.result()  # first chunk already consumed
                    except UpstreamFailure as error:
                        failures.append(f"{target.id}: HTTP {error.status} {error}")
                        self.store.record_decision(
                            session_id, context.scenario, context.persona,
                            target.id, None, False, error.status, str(error),
                        )
                        continue
                    except StopAsyncIteration:
                        failures.append(f"{target.id}: empty stream")
                        continue
                    except Exception as error:
                        failures.append(f"{target.id}: {error}")
                        continue
                    # Winner!
                    winner_target = target
                    break
            if winner_target is None:
                log.warning(
                    "all parallel stream candidates failed: %s",
                    "; ".join(failures),
                )
                return None
            # Cancel remaining tasks and close losing iterators
            for task in pending:
                task.cancel()
            for t, it in iterators.items():
                if t is not winner_target:
                    await it.aclose()
            return CompletionResult(
                body={"choices": [{"message": {"content": ""}}]},
                target=winner_target,
                latency_ms=0,
            )
        finally:
            for task in first_chunk_tasks:
                if not task.done():
                    task.cancel()

    async def _serial_fallback(
        self,
        candidates: list[TargetConfig],
        payload: dict[str, Any],
        context: RouteContext,
        session_id: str | None,
    ) -> CompletionResult:
        """Serially try leftover candidates after a parallel race failure."""
        failures: list[str] = []
        for target in candidates:
            try:
                result = await self._call(target, payload)
            except UpstreamFailure as error:
                failures.append(f"{target.id}: HTTP {error.status} {error}")
                self.store.record_decision(
                    session_id, context.scenario, context.persona, target.id,
                    None, False, error.status, str(error),
                )
                continue
            return result
        raise NoTargetAvailable(
            "all matching targets failed: " + "; ".join(failures)
        )

    async def _call(
        self, target: TargetConfig, payload: dict[str, Any], probe: bool = False
    ) -> CompletionResult:
        request, capped = _upstream_payload(payload, target, probe)
        if capped:
            self.store.record_cap(target.id)
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
        usage = body.get("usage") if not probe else None
        self.store.record_success(
            target.id, latency_ms, self.config.routing.ewma_alpha, probe,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0) if isinstance(usage, dict) else 0,
            completion_tokens=int(usage.get("completion_tokens", 0) or 0) if isinstance(usage, dict) else 0,
            total_tokens=int(usage.get("total_tokens", 0) or 0) if isinstance(usage, dict) else 0,
        )
        body["model"] = target.model
        return CompletionResult(body=body, target=target, latency_ms=latency_ms)

    async def _stream_call(
        self, target: TargetConfig, payload: dict[str, Any], probe: bool = False
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a streaming request and yield normalized SSE chunks.

        Each yielded dict is a single SSE ``data:`` chunk normalized to the
        OpenAI chat.completion.chunk schema.  Raises ``UpstreamFailure``
        **before** the first chunk is yielded — after the first chunk the
        caller should consider the stream committed and not attempt fallback.
        """
        request, capped = _upstream_payload(payload, target, probe)
        if capped:
            self.store.record_cap(target.id)
        request["stream"] = True
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if target.api_key:
            headers["Authorization"] = f"Bearer {target.api_key}"
        _first_yielded = False
        started = time.monotonic()
        try:
            async with self._semaphores[target.id]:
                response = await self.client.post(
                    target.chat_url, headers=headers, json=request
                )
            if response.status_code < 200 or response.status_code >= 300:
                raise UpstreamFailure(
                    target, response.status_code, _error_message(response)
                )
            latency_ms = (time.monotonic() - started) * 1000
            content_type = response.headers.get("content-type", "").lower()
            json_response = "application/json" in content_type or response.content.lstrip().startswith(b"{")
            if "text/event-stream" not in content_type and json_response:
                body = response.json()
                if not isinstance(body, dict) or not isinstance(body.get("choices"), list):
                    raise ValueError("non-streaming upstream response has no choices")
                normalized = _normalize_stream_chunk(body, target.model)
                usage = body.get("usage") if not probe else None
                self.store.record_success(
                    target.id, latency_ms, self.config.routing.ewma_alpha, probe,
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0) if isinstance(usage, dict) else 0,
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0) if isinstance(usage, dict) else 0,
                    total_tokens=int(usage.get("total_tokens", 0) or 0) if isinstance(usage, dict) else 0,
                )
                _first_yielded = True
                yield normalized
                return
            async for line in response.aiter_lines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                normalized = _normalize_stream_chunk(chunk, target.model)
                usage = chunk.get("usage") if not probe else None
                if not _first_yielded:
                    _first_yielded = True
                    self.store.record_success(
                        target.id, latency_ms, self.config.routing.ewma_alpha, probe,
                        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0) if isinstance(usage, dict) else 0,
                        completion_tokens=int(usage.get("completion_tokens", 0) or 0) if isinstance(usage, dict) else 0,
                        total_tokens=int(usage.get("total_tokens", 0) or 0) if isinstance(usage, dict) else 0,
                    )
                elif isinstance(usage, dict):
                    # Subsequent chunk with usage (some providers send it on the last chunk)
                    self.store.record_usage(
                        target.id,
                        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                        total_tokens=int(usage.get("total_tokens", 0) or 0),
                    )
                yield normalized
        except UpstreamFailure as error:
            if not _first_yielded:
                self._record_failure(target, error.status, str(error), probe)
            raise
        except httpx.TimeoutException as error:
            failure = UpstreamFailure(target, 504, f"timeout: {error}")
            if not _first_yielded:
                self._record_failure(target, failure.status, str(failure), probe)
            raise failure from error
        except (httpx.HTTPError, ValueError, TypeError) as error:
            failure = UpstreamFailure(target, 502, f"invalid upstream response: {error}")
            if not _first_yielded:
                self._record_failure(target, failure.status, str(failure), probe)
            raise failure from error

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
        if delay > 0:
            jitter = random.uniform(0, delay * 0.2)
            delay = min(delay + jitter, self.config.routing.circuit_max_seconds)
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
) -> tuple[dict[str, Any], bool]:
    request = {key: value for key, value in payload.items() if key in UPSTREAM_FIELDS}
    request["model"] = target.model
    request["stream"] = False
    if probe:
        request.pop("tools", None)
        request.pop("tool_choice", None)
        return request, False
    # Cap max_new_tokens when max_context is set to avoid 400 errors
    capped = False
    if target.max_context is not None:
        inputs_tokens = estimate_messages_tokens(request.get("messages", []))
        max_new = request.get("max_tokens", request.get("max_completion_tokens", 1024))
        if max_new is None:
            max_new = 1024
        allowed = target.max_context - inputs_tokens
        if allowed < 1:
            allowed = 1  # Allow at least 1 token to avoid zero-value errors
        if max_new > allowed:
            capped = True
            log.warning(
                "capping max_new_tokens for %s: %d -> %d (inputs=%d, max_context=%d)",
                target.id, max_new, allowed, inputs_tokens, target.max_context,
            )
            if "max_tokens" in request:
                request["max_tokens"] = max(1, int(allowed))
            if "max_completion_tokens" in request:
                request["max_completion_tokens"] = max(1, int(allowed))
    return request, capped


def _max_new_tokens(payload: dict[str, Any]) -> int:
    """Extract max_new_tokens from payload, defaulting to 1024."""
    return int(payload.get("max_tokens", payload.get("max_completion_tokens", 1024)) or 1024)


_OVERFLOW_MARKERS = (
    "max_new_tokens",
    "must be <=",
    "tokens +",
    "context length",
    "maximum context",
    "too long",
    "too many tokens",
)


def _is_context_overflow(error: UpstreamFailure) -> bool:
    """Detect upstream 400 errors caused by input exceeding context window."""
    if error.status != 400:
        return False
    message = str(error).lower()
    return any(marker in message for marker in _OVERFLOW_MARKERS)


# Backward-compatible aliases (tests import these private names directly).
# They delegate to the shared damselfish.tokens module so behaviour stays
# identical across selector and router.
_estimate_text_tokens = estimate_text_tokens
_estimate_current_input_tokens = estimate_messages_tokens





def _validate_completion(body: Any) -> None:
    if not isinstance(body, dict):
        raise ValueError("response is not a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("response has no assistant message")
    usable = (
        message.get("content")
        or message.get("tool_calls")
        or message.get("function_call")
        or message.get("reasoning_content")
    )
    if not usable:
        raise ValueError("assistant message has no usable content, reasoning, or tool call")


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
        error = body.get("error", body)
        if isinstance(error, dict):
            return str(error.get("message", error))[:500]
        return str(error)[:500]
    except (ValueError, TypeError):
        return response.text[:500]


def _normalize_stream_chunk(chunk: dict, target_model: str) -> dict:
    """Normalize an upstream SSE chunk to OpenAI chat.completion.chunk format."""
    choices = chunk.get("choices", [])
    normalized_choices = []
    for c in choices:
        delta = c.get("delta", c.get("message", {}))
        normalized_choices.append({
            "index": c.get("index", 0),
            "delta": delta,
            "finish_reason": c.get("finish_reason"),
        })
    return {
        "id": chunk.get("id", ""),
        "object": "chat.completion.chunk",
        "created": chunk.get("created", int(time.time())),
        "model": target_model,
        "choices": normalized_choices,
    }
