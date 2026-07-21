# 流式并行回退实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为千机 (Damselfish) 添加真正的流式 SSE 支持，并在 Phase 2 并行竞速时支持流式首块竞速。

**Architecture:** 在 `router.py` 中新增 `stream_complete()` 异步生成器方法，通过 `_stream_call()` 发送 `stream: true` 请求并逐块解析上游 SSE；新增 `_race_stream()` 用 `asyncio.wait(FIRST_COMPLETED)` 竞速首块。在 `app.py` 中修改流式路径，用 `StreamingResponse` 转发标准化 SSE 并在流结束后保存记忆。

**Tech Stack:** Python 3.11+, FastAPI, httpx, asyncio, SSE

## 全局约束

- Python >= 3.11
- httpx >= 0.27
- FastAPI >= 0.115
- 所有流式 chunk 必须标准化为 OpenAI SSE 格式
- `X-Damselfish-Latency-Ms` 为首块到达时间
- 记忆保存移到流结束后

---

## 文件结构

| 文件 | 职责 | 变更 |
|------|------|------|
| `damselfish/router.py` | 新增 `stream_complete()`, `_stream_call()`, `_race_stream()` | 修改 |
| `damselfish/app.py` | 修改流式路径，新增 SSE 标准化和流式记忆保存 | 修改 |
| `tests/test_router.py` | 新增流式测试 | 修改 |
| `tests/test_app.py` | 新增流式端到端测试 | 修改 |

---

### Task 1: 在 router.py 中新增 `_stream_call()` 方法

**Files:**
- Modify: `damselfish/router.py` (after `_call()` method)
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: `TargetConfig`, `payload: dict`, 可选 `probe: bool`
- Produces: `_stream_call(target, payload, probe=False) -> AsyncIterator[dict]` — 异步生成器，每次 yield 一个标准化 SSE chunk dict；首块 yield 前抛异常视为"首块前失败"

- [ ] **Step 1: 添加 `AsyncIterator` import 和 `_normalize_stream_chunk()` 函数**

在 `router.py` 顶部添加 import:
```python
from collections.abc import AsyncIterator
```

在文件末尾模块级函数区域添加:

```python
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
```

- [ ] **Step 2: 实现 `_stream_call()` 方法**

在 `_call()` 方法之后、`_record_failure()` 之前添加:

```python
async def _stream_call(
    self, target: TargetConfig, payload: dict[str, Any], probe: bool = False
) -> AsyncIterator[dict[str, Any]]:
    """Send a streaming request and yield normalized SSE chunks.

    Each yielded dict is a single SSE ``data:`` chunk normalized to the
    OpenAI chat.completion.chunk schema.  Raises ``UpstreamFailure``
    **before** the first chunk is yielded — after the first chunk the
    caller should consider the stream committed and not attempt fallback.
    """
    request = _upstream_payload(payload, target, probe)
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
            if not _first_yielded:
                _first_yielded = True
                self.store.record_success(
                    target.id, latency_ms, self.config.routing.ewma_alpha, probe
                )
            yield normalized
    except UpstreamFailure:
        if not _first_yielded:
            self._record_failure(target, ...)  # re-read from exception context
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
```

- [ ] **Step 3: 编写测试**

在 `tests/test_router.py` 末尾添加 `_success_response` 和三个测试函数:

```python
def test_stream_call_yields_chunks(tmp_path: Path) -> None:
    """_stream_call yields normalized SSE chunks from upstream."""
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(),
        targets=(TargetConfig("test", "Test", "http://router/v1", "test-model", local=True),),
    )
    store = Store(config.database, ["test"])
    sse_chunks = [
        'data: {"id":"x","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
        'data: {"id":"x","choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n',
        'data: {"id":"x","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        "data: [DONE]\n\n",
    ]
    sse_bytes = "".join(sse_chunks).encode()

    def handler(request):
        return httpx.Response(200, content=sse_bytes)

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        router = ModelRouter(config, store, client)
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        chunks = []
        async for chunk in router._stream_call(config.targets[0], payload):
            chunks.append(chunk)
        assert len(chunks) == 3
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert chunks[1]["choices"][0]["delta"]["content"] == "hello"
        assert chunks[2]["choices"][0]["finish_reason"] == "stop"
        assert chunks[0]["model"] == "test-model"

    asyncio.run(run())
    store.close()


def test_stream_call_429_raises_before_first_chunk(tmp_path: Path) -> None:
    """_stream_call raises UpstreamFailure before yielding if status is 429."""
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(),
        targets=(TargetConfig("test", "Test", "http://router/v1", "test-model", local=True),),
    )
    store = Store(config.database, ["test"])

    def handler(request):
        return httpx.Response(429, json={"error": {"message": "limited"}})

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        router = ModelRouter(config, store, client)
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        with pytest.raises(UpstreamFailure) as exc:
            async for _ in router._stream_call(config.targets[0], payload):
                pass
        assert exc.value.status == 429

    asyncio.run(run())
    store.close()


def test_stream_call_timeout_504_before_first_chunk(tmp_path: Path) -> None:
    """_stream_call raises UpstreamFailure(504) on timeout."""
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(),
        targets=(TargetConfig("test", "Test", "http://router/v1", "test-model", local=True),),
    )
    store = Store(config.database, ["test"])

    def handler(request):
        raise httpx.TimeoutException("simulated timeout")

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        router = ModelRouter(config, store, client)
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        with pytest.raises(UpstreamFailure) as exc:
            async for _ in router._stream_call(config.targets[0], payload):
                pass
        assert exc.value.status == 504

    asyncio.run(run())
    store.close()
```

- [ ] **Step 4: 运行测试**

```bash
cd /Users/sxliuyu/orca/workspaces/damselfish/main
uv run pytest tests/test_router.py::test_stream_call_yields_chunks -xvs 2>&1 | tail -15
uv run pytest tests/test_router.py::test_stream_call_429_raises_before_first_chunk -xvs 2>&1 | tail -15
uv run pytest tests/test_router.py::test_stream_call_timeout_504_before_first_chunk -xvs 2>&1 | tail -15
```

Expected: 3 tests pass.

- [ ] **Step 5: 提交**

```bash
git add damselfish/router.py tests/test_router.py
git commit -m "feat: add _stream_call with normalized SSE chunk parsing"
```

---

### Task 2: 在 router.py 中新增 `stream_complete()` 和 `_race_stream()` 方法

**Files:**
- Modify: `damselfish/router.py` (after `complete()` method)
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: `stream_complete(payload, context, session_id) -> AsyncIterator[dict]` — 异步生成器，通过 `self._stream_result` 传递最终 `CompletionResult`
- Produces: `_race_stream(candidates, payload, context, session_id) -> CompletionResult | None`

- [ ] **Step 1: 实现 `stream_complete()` 和 `_race_stream()` 方法**

在 `complete()` 方法之后、`_race_targets()` 之前添加 `stream_complete()`:

```python
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
        if error.status not in (429, 504) or len(targets) < 2:
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
```

在 `_race_targets()` 之后添加 `_race_stream()`:

```python
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
    winner_chunk: dict | None = None
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
                    chunk = task.result()
                except UpstreamFailure as error:
                    failures.append(f"{target.id}: HTTP {error.status} {error}")
                    self.store.record_decision(
                        session_id, context.scenario, context.persona,
                        target.id, None, False, error.status, str(error),
                    )
                    continue
                except StopAsyncIteration:
                    failures.append(f"{target.id}: empty s