import asyncio
from pathlib import Path

import httpx
import pytest

from damselfish.config import AppConfig, RoutingConfig, TargetConfig
from damselfish.router import (
    ModelRouter,
    UpstreamFailure,
    _estimate_current_input_tokens,
    _estimate_text_tokens,
    _is_context_overflow,
    _max_new_tokens,
    _upstream_payload,
)
from damselfish.selector import RouteContext
from damselfish.store import Store


def test_router_falls_back_after_rate_limit(tmp_path: Path) -> None:
    config = AppConfig(
        host="127.0.0.1",
        port=8086,
        database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=1),
        targets=(
            TargetConfig("first", "First", "http://router/v1", "first", local=True, priority=1),
            TargetConfig("second", "Second", "http://router/v1", "second", local=True, priority=2),
        ),
    )
    store = Store(config.database, ["first", "second"])

    def handler(request: httpx.Request) -> httpx.Response:
        model = __import__("json").loads(request.content)["model"]
        if model == "first":
            return httpx.Response(429, json={"error": {"message": "limited"}})
        return httpx.Response(
            200,
            json={"id": "ok", "choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            result = await router.complete(
                {"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
                RouteContext("default", None, frozenset(), frozenset(), ()),
                "test",
            )
            assert result.target.id == "second"

    asyncio.run(run())
    assert store.stats("first").rate_limits == 1
    assert store.stats("first").circuit_open_until > 0
    store.close()


def test_router_accepts_reasoning_only_response(tmp_path: Path) -> None:
    config = AppConfig(
        host="127.0.0.1",
        port=8086,
        database=tmp_path / "test.db",
        routing=RoutingConfig(),
        targets=(
            TargetConfig(
                "reasoning",
                "Reasoning",
                "http://router/v1",
                "reasoning-model",
                local=False,
            ),
        ),
    )
    store = Store(config.database, ["reasoning"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "OK",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            result = await router.complete(
                {"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
                RouteContext("default", None, frozenset(), frozenset(), ()),
                "test",
            )
            assert result.body["choices"][0]["message"]["reasoning_content"] == "OK"

    asyncio.run(run())
    assert store.stats("reasoning").successes == 1
    store.close()


def _success_response(model: str) -> dict:
    return {
        "id": f"ok-{model}",
        "choices": [
            {"message": {"role": "assistant", "content": f"from {model}"}, "finish_reason": "stop"}
        ],
    }


def test_parallel_fallback_on_429(tmp_path: Path) -> None:
    """Primary returns 429; parallel race picks the fastest remaining target."""
    config = AppConfig(
        host="127.0.0.1",
        port=8086,
        database=tmp_path / "test.db",
        routing=RoutingConfig(
            priority_weight_ms=1,
            parallel_fallback_count=3,
            parallel_fallback_timeout_seconds=5.0,
        ),
        targets=(
            TargetConfig("primary", "Primary", "http://router/v1", "primary", local=True, priority=1),
            TargetConfig("fast", "Fast", "http://router/v1", "fast", local=True, priority=2),
            TargetConfig("slow", "Slow", "http://router/v1", "slow", local=True, priority=3),
            TargetConfig("last", "Last", "http://router/v1", "last", local=True, priority=4),
        ),
    )
    store = Store(config.database, ["primary", "fast", "slow", "last"])

    call_order: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = __import__("json").loads(request.content)["model"]
        call_order.append(model)
        if model == "primary":
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        if model == "fast":
            return httpx.Response(200, json=_success_response("fast"))
        if model == "slow":
            return httpx.Response(200, json=_success_response("slow"))
        return httpx.Response(200, json=_success_response("last"))

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            result = await router.complete(
                {"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
                RouteContext("default", None, frozenset(), frozenset(), ()),
                "test",
            )
            # Primary failed with 429, parallel race picks the fastest success
            assert result.target.id in {"fast", "slow", "last"}
            assert result.body["choices"][0]["message"]["content"].startswith("from ")

    asyncio.run(run())
    # Primary was called first, then parallel targets were all dispatched
    assert "primary" in call_order
    assert any(t in call_order for t in ("fast", "slow", "last"))
    assert store.stats("primary").rate_limits == 1
    store.close()


def test_parallel_fallback_on_timeout(tmp_path: Path) -> None:
    """Primary times out (504); parallel race picks the fastest remaining target."""
    config = AppConfig(
        host="127.0.0.1",
        port=8086,
        database=tmp_path / "test.db",
        routing=RoutingConfig(
            priority_weight_ms=1,
            parallel_fallback_count=2,
            parallel_fallback_timeout_seconds=5.0,
        ),
        targets=(
            TargetConfig("primary", "Primary", "http://router/v1", "primary", local=True, priority=1),
            TargetConfig("alt", "Alt", "http://router/v1", "alt", local=True, priority=2),
            TargetConfig("backup", "Backup", "http://router/v1", "backup", local=True, priority=3),
        ),
    )
    store = Store(config.database, ["primary", "alt", "backup"])

    call_order: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = __import__("json").loads(request.content)["model"]
        call_order.append(model)
        if model == "primary":
            # Simulate an upstream timeout
            raise httpx.TimeoutException("simulated timeout")
        return httpx.Response(200, json=_success_response(model))

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            result = await router.complete(
                {"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
                RouteContext("default", None, frozenset(), frozenset(), ()),
                "test",
            )
            # Primary timed out (504), parallel race should pick "alt" or "backup"
            assert result.target.id in {"alt", "backup"}

    asyncio.run(run())
    assert "primary" in call_order
    assert "alt" in call_order
    store.close()


def test_parallel_fallback_all_fail_falls_to_serial(tmp_path: Path) -> None:
    """Primary returns 429, all parallel candidates also fail; serial fallback
    continues with remaining targets beyond the parallel limit."""
    config = AppConfig(
        host="127.0.0.1",
        port=8086,
        database=tmp_path / "test.db",
        routing=RoutingConfig(
            priority_weight_ms=1,
            parallel_fallback_count=2,
            parallel_fallback_timeout_seconds=5.0,
        ),
        targets=(
            TargetConfig("primary", "Primary", "http://router/v1", "primary", local=True, priority=1),
            TargetConfig("fail1", "Fail1", "http://router/v1", "fail1", local=True, priority=2),
            TargetConfig("fail2", "Fail2", "http://router/v1", "fail2", local=True, priority=3),
            TargetConfig("winner", "Winner", "http://router/v1", "winner", local=True, priority=4),
        ),
    )
    store = Store(config.database, ["primary", "fail1", "fail2", "winner"])

    call_order: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = __import__("json").loads(request.content)["model"]
        call_order.append(model)
        if model in ("primary", "fail1", "fail2"):
            return httpx.Response(429, json={"error": {"message": "limited"}})
        return httpx.Response(200, json=_success_response("winner"))

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            result = await router.complete(
                {"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
                RouteContext("default", None, frozenset(), frozenset(), ()),
                "test",
            )
            # Parallel race (fail1, fail2) both fail, serial fallback picks winner
            assert result.target.id == "winner"

    asyncio.run(run())
    assert "primary" in call_order
    assert "fail1" in call_order
    assert "fail2" in call_order
    assert "winner" in call_order
    store.close()


# ─── Streaming tests ─────────────────────────────────────────────────


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


def test_stream_call_converts_non_streaming_json_response(tmp_path: Path) -> None:
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(),
        targets=(TargetConfig("test", "Test", "http://router/v1", "test-model", local=True),),
    )
    store = Store(config.database, ["test"])

    def handler(request):
        return httpx.Response(200, json={
            "id": "completion",
            "object": "chat.completion",
            "created": 1,
            "model": "upstream-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "OK"},
                "finish_reason": "stop",
            }],
        })

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        router = ModelRouter(config, store, client)
        chunks = []
        async for chunk in router._stream_call(
            config.targets[0], {"messages": [{"role": "user", "content": "hi"}]}
        ):
            chunks.append(chunk)
        await client.aclose()
        assert len(chunks) == 1
        assert chunks[0]["choices"][0]["delta"]["content"] == "OK"
        assert chunks[0]["choices"][0]["finish_reason"] == "stop"

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


def test_stream_complete_phase1_success(tmp_path: Path) -> None:
    """stream_complete yields chunks from primary target on success."""
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(),
        targets=(TargetConfig("test", "Test", "http://router/v1", "test-model", local=True),),
    )
    store = Store(config.database, ["test"])
    sse_bytes = (
        'data: {"id":"x","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        'data: {"id":"x","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
        'data: {"id":"x","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: [DONE]\n\n'
    ).encode()

    def handler(request):
        return httpx.Response(200, content=sse_bytes)

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        router = ModelRouter(config, store, client)
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        chunks = []
        async for chunk in router.stream_complete(
            payload,
            RouteContext("default", None, frozenset(), frozenset(), ()),
            "test",
        ):
            chunks.append(chunk)
        assert len(chunks) == 3
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert router._stream_result is not None
        assert router._stream_result.target.id == "test"

    asyncio.run(run())
    store.close()


def test_stream_complete_phase1_429_fallback(tmp_path: Path) -> None:
    """Phase 1 returns 429, Phase 2 race yields chunks from winner."""
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(
            priority_weight_ms=1,
            parallel_fallback_count=2,
            parallel_fallback_timeout_seconds=5.0,
        ),
        targets=(
            TargetConfig("primary", "Primary", "http://router/v1", "primary", local=True, priority=1),
            TargetConfig("alt", "Alt", "http://router/v1", "alt", local=True, priority=2),
            TargetConfig("backup", "Backup", "http://router/v1", "backup", local=True, priority=3),
        ),
    )
    store = Store(config.database, ["primary", "alt", "backup"])
    sse_bytes = (
        'data: {"id":"x","choices":[{"delta":{"content":"from alt"},"finish_reason":null}]}\n\n'
        'data: {"id":"x","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: [DONE]\n\n'
    ).encode()

    def handler(request):
        model = __import__("json").loads(request.content)["model"]
        if model == "primary":
            return httpx.Response(429, json={"error": {"message": "limited"}})
        return httpx.Response(200, content=sse_bytes)

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        router = ModelRouter(config, store, client)
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        chunks = []
        async for chunk in router.stream_complete(
            payload,
            RouteContext("default", None, frozenset(), frozenset(), ()),
            "test",
        ):
            chunks.append(chunk)
        assert len(chunks) >= 1
        assert router._stream_result is not None
        assert router._stream_result.target.id in {"alt", "backup"}

    asyncio.run(run())
    store.close()


def test_stream_complete_phase2_all_fail_falls_to_serial(tmp_path: Path) -> None:
    """Phase 1 429, Phase 2 all fail, Phase 3 serial fallback succeeds."""
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(
            priority_weight_ms=1,
            parallel_fallback_count=2,
            parallel_fallback_timeout_seconds=5.0,
        ),
        targets=(
            TargetConfig("primary", "Primary", "http://router/v1", "primary", local=True, priority=1),
            TargetConfig("fail1", "Fail1", "http://router/v1", "fail1", local=True, priority=2),
            TargetConfig("fail2", "Fail2", "http://router/v1", "fail2", local=True, priority=3),
            TargetConfig("winner", "Winner", "http://router/v1", "winner", local=True, priority=4),
        ),
    )
    store = Store(config.database, ["primary", "fail1", "fail2", "winner"])

    def handler(request):
        model = __import__("json").loads(request.content)["model"]
        if model in ("primary", "fail1", "fail2"):
            return httpx.Response(429, json={"error": {"message": "limited"}})
        return httpx.Response(
            200,
            json={
                "id": "ok",
                "choices": [{
                    "message": {"role": "assistant", "content": "from winner"},
                    "finish_reason": "stop",
                }],
            },
        )

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        router = ModelRouter(config, store, client)
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        chunks = []
        async for chunk in router.stream_complete(
            payload,
            RouteContext("default", None, frozenset(), frozenset(), ()),
            "test",
        ):
            chunks.append(chunk)
        assert len(chunks) >= 1
        assert router._stream_result is not None
        assert router._stream_result.target.id == "winner"

    asyncio.run(run())
    store.close()


# ── _max_new_tokens tests ────────────────────────────────────────────


def test_max_new_tokens_default() -> None:
    """Default max_new_tokens is 1024."""
    assert _max_new_tokens({}) == 1024
    assert _max_new_tokens({"messages": []}) == 1024


def test_max_new_tokens_from_payload() -> None:
    """Extracts max_tokens from payload."""
    assert _max_new_tokens({"max_tokens": 2048}) == 2048


def test_max_new_tokens_from_max_completion_tokens() -> None:
    """Falls back to max_completion_tokens."""
    assert _max_new_tokens({"max_completion_tokens": 4096}) == 4096


def test_max_new_tokens_prefers_max_tokens() -> None:
    """max_tokens beats max_completion_tokens."""
    assert _max_new_tokens({"max_tokens": 512, "max_completion_tokens": 1024}) == 512


# ── _estimate_text_tokens / _estimate_current_input_tokens tests ─────────────


def test_router_estimate_text_tokens_cjk() -> None:
    """CJK text estimated correctly."""
    from damselfish import tokens
    if tokens._TIKTOKEN_ENCODING is not None:
        assert _estimate_text_tokens("中" * 1000) == 1000
    else:
        assert _estimate_text_tokens("中" * 1000) == 1650


def test_router_estimate_text_tokens_ascii() -> None:
    """ASCII text estimated correctly."""
    from damselfish import tokens
    if tokens._TIKTOKEN_ENCODING is not None:
        assert _estimate_text_tokens("hello world " * 100) == 201
    else:
        assert _estimate_text_tokens("hello world " * 100) == 330


def test_router_estimate_current_input_tokens() -> None:
    """Estimate input tokens from messages list."""
    messages = [
        {"role": "user", "content": "中" * 100},
        {"role": "assistant", "content": "hello"},
    ]
    estimated = _estimate_current_input_tokens(messages)
    assert estimated > 0


# ── _is_context_overflow tests ───────────────────────────────────────


def test_is_context_overflow_detects_zhipu_error() -> None:
    """Detects Zhipu-style context overflow error."""
    error = UpstreamFailure(
        TargetConfig("test", "Test", "http://t/v1", "t"),
        400,
        "Input validation error: `inputs` tokens + `max_new_tokens` must be <= 16384",
    )
    assert _is_context_overflow(error) is True


def test_is_context_overflow_detects_maximum_context() -> None:
    """Detects 'maximum context length' error."""
    error = UpstreamFailure(
        TargetConfig("test", "Test", "http://t/v1", "t"),
        400,
        "This model's maximum context length is 4096 tokens",
    )
    assert _is_context_overflow(error) is True


def test_is_context_overflow_detects_too_long() -> None:
    """Detects 'too long' error."""
    error = UpstreamFailure(
        TargetConfig("test", "Test", "http://t/v1", "t"),
        400,
        "text is too long for the model",
    )
    assert _is_context_overflow(error) is True


def test_is_context_overflow_rejects_429() -> None:
    """429 errors are not context overflow."""
    error = UpstreamFailure(
        TargetConfig("test", "Test", "http://t/v1", "t"),
        429,
        "rate limited",
    )
    assert _is_context_overflow(error) is False


def test_is_context_overflow_rejects_other_400() -> None:
    """Other 400 errors are not context overflow."""
    error = UpstreamFailure(
        TargetConfig("test", "Test", "http://t/v1", "t"),
        400,
        "invalid parameter: temperature must be between 0 and 2",
    )
    assert _is_context_overflow(error) is False


# ── _upstream_payload capping tests ──────────────────────────────────


def test_upstream_payload_caps_max_tokens() -> None:
    """max_tokens is capped when input + max_tokens > max_context."""
    # Use small max_context so cap triggers in both tiktoken and heuristic.
    # tiktoken: 1000 CJK = 1000 tokens; heuristic: 1000 CJK = 1650 tokens.
    # With max_context=1500 and max_tokens=1024, both backends exceed.
    target = TargetConfig(
        "test", "Test", "http://t/v1", "t",
        local=True, max_context=1500,
    )
    payload = {
        "messages": [{"role": "user", "content": "中" * 1000}],
        "max_tokens": 1024,
    }
    request, capped = _upstream_payload(payload, target, probe=False)
    assert request["max_tokens"] < 1024
    assert capped is True


def test_upstream_payload_no_cap_within_limit() -> None:
    """max_tokens is not capped when within max_context."""
    target = TargetConfig(
        "test", "Test", "http://t/v1", "t",
        local=True, max_context=4096,
    )
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 500,
    }
    request, capped = _upstream_payload(payload, target, probe=False)
    assert request["max_tokens"] == 500
    assert capped is False


def test_upstream_payload_no_max_context() -> None:
    """Without max_context, no capping occurs."""
    target = TargetConfig(
        "test", "Test", "http://t/v1", "t",
        local=True, max_context=None,
    )
    payload = {
        "messages": [{"role": "user", "content": "中" * 50000}],
        "max_tokens": 99999,
    }
    request, capped = _upstream_payload(payload, target, probe=False)
    assert request["max_tokens"] == 99999
    assert capped is False


def test_upstream_payload_probe_skips_capping() -> None:
    """Probe requests skip capping."""
    target = TargetConfig(
        "test", "Test", "http://t/v1", "t",
        local=True, max_context=4096,
    )
    payload = {
        "messages": [{"role": "user", "content": "中" * 5000}],
        "max_tokens": 99999,
    }
    request, capped = _upstream_payload(payload, target, probe=True)
    assert "tools" not in request
    assert request["max_tokens"] == 99999
    assert capped is False


def test_upstream_payload_caps_max_completion_tokens() -> None:
    """max_completion_tokens is also capped."""
    target = TargetConfig(
        "test", "Test", "http://t/v1", "t",
        local=True, max_context=2048,
    )
    payload = {
        "messages": [{"role": "user", "content": "中" * 1200}],
        "max_completion_tokens": 2048,
    }
    request, capped = _upstream_payload(payload, target, probe=False)
    assert request["max_completion_tokens"] < 2048
    assert capped is True


# ── Router fallback on context overflow 400 ─────────────────────────


def test_router_falls_back_on_context_overflow_400(tmp_path: Path) -> None:
    """Primary returns 400 context overflow; falls back to next target."""
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=1),
        targets=(
            TargetConfig("short", "Short", "http://router/v1", "short",
                         local=True, priority=1),
            TargetConfig("long", "Long", "http://router/v1", "long",
                         local=True, priority=2),
        ),
    )
    store = Store(config.database, ["short", "long"])

    def handler(request: httpx.Request) -> httpx.Response:
        model = __import__("json").loads(request.content)["model"]
        if model == "short":
            return httpx.Response(
                400,
                json={"error": {"message": "Input validation error: `inputs` tokens + `max_new_tokens` must be <= 16384"}},
            )
        return httpx.Response(200, json={"id": "ok", "choices": [{"message": {"role": "assistant", "content": "from long"}, "finish_reason": "stop"}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            result = await router.complete(
                {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                RouteContext("default", None, frozenset(), frozenset(), ()),
                "test",
            )
            assert result.target.id == "long"
            assert result.body["choices"][0]["message"]["content"] == "from long"

    asyncio.run(run())
    store.close()


def test_router_does_not_fallback_on_other_400(tmp_path: Path) -> None:
    """Non-context-overflow 400 does not trigger fallback."""
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=1),
        targets=(
            TargetConfig("primary", "Primary", "http://router/v1", "primary",
                         local=True, priority=1),
            TargetConfig("backup", "Backup", "http://router/v1", "backup",
                         local=True, priority=2),
        ),
    )
    store = Store(config.database, ["primary", "backup"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "invalid parameter"}})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            with pytest.raises(Exception) as exc:
                await router.complete(
                    {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                    RouteContext("default", None, frozenset(), frozenset(), ()),
                    "test",
                )
            assert "NoTargetAvailable" in type(exc.value).__name__ or "primary target primary failed" in str(exc.value)

    asyncio.run(run())
    store.close()


def test_stream_complete_phase1_400_context_overflow_fallback(tmp_path: Path) -> None:
    """Stream: Phase 1 returns 400 context overflow; Phase 2 race succeeds."""
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=1, parallel_fallback_count=2, parallel_fallback_timeout_seconds=5.0),
        targets=(
            TargetConfig("short", "Short", "http://router/v1", "short",
                         local=True, priority=1),
            TargetConfig("long", "Long", "http://router/v1", "long",
                         local=True, priority=2),
        ),
    )
    store = Store(config.database, ["short", "long"])
    sse_bytes = (
        'data: {"id":"x","choices":[{"delta":{"content":"from long"},"finish_reason":null}]}\n\n'
        'data: {"id":"x","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: [DONE]\n\n'
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        model = __import__("json").loads(request.content)["model"]
        if model == "short":
            return httpx.Response(400, json={"error": {"message": "`inputs` tokens + `max_new_tokens` must be <= 16384"}})
        return httpx.Response(200, content=sse_bytes)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            chunks = []
            async for chunk in router.stream_complete(
                {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                RouteContext("default", None, frozenset(), frozenset(), ()),
                "test",
            ):
                chunks.append(chunk)
            assert len(chunks) >= 1
            assert router._stream_result is not None
            assert router._stream_result.target.id == "long"

    asyncio.run(run())
    store.close()


# ── cap_count recording ────────────────────────────────────────────


def test_router_records_cap_count(tmp_path: Path) -> None:
    """When max_new_tokens is capped, store.record_cap is called."""
    from damselfish import tokens
    # Use a small max_context that triggers cap in both tiktoken and heuristic.
    # tiktoken: 1000 CJK = 1000 tokens; heuristic: 1000 CJK = 1650 tokens.
    # With max_context=1500 and max_tokens=1024, both backends exceed.
    max_context = 1500
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=1),
        targets=(
            TargetConfig("small", "Small", "http://router/v1", "small",
                         local=True, priority=1, max_context=max_context),
        ),
    )
    store = Store(config.database, ["small"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"id": "ok", "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            # Input ~1000 tokens (tiktoken) or ~1650 (heuristic) + max_tokens=1024 > 1500
            await router.complete(
                {"model": "auto", "messages": [{"role": "user", "content": "中" * 1000}], "max_tokens": 1024},
                RouteContext("default", None, frozenset(), frozenset(), ()),
                "test",
            )

    asyncio.run(run())
    assert store.stats("small").cap_count == 1
    assert store.stats("small").public()["cap_count"] == 1
    store.close()


# ── token usage recording ──────────────────────────────────────────


def test_router_records_usage_from_non_streaming(tmp_path: Path) -> None:
    """Router captures upstream ``usage`` and records it to the store."""
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=1),
        targets=(
            TargetConfig("t1", "T1", "http://router/v1", "m",
                         local=True, priority=1),
        ),
    )
    store = Store(config.database, ["t1"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "ok",
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            await router.complete(
                {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                RouteContext("default", None, frozenset(), frozenset(), ()),
                "test",
            )

    asyncio.run(run())
    stats = store.stats("t1")
    assert stats.prompt_tokens == 100
    assert stats.completion_tokens == 50
    assert stats.total_tokens == 150
    store.close()


def test_router_records_usage_from_streaming(tmp_path: Path) -> None:
    """Router captures ``usage`` from the last SSE chunk in streaming mode."""
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=1),
        targets=(
            TargetConfig("t1", "T1", "http://router/v1", "m",
                         local=True, priority=1),
        ),
    )
    store = Store(config.database, ["t1"])

    upstream_body = (
        'data: {"id":"x","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        'data: {"id":"x","choices":[{"delta":{"content":"OK"},"finish_reason":null}]}\n\n'
        'data: {"id":"x","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":80,"completion_tokens":40,"total_tokens":120}}\n\n'
        'data: [DONE]\n\n'
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=upstream_body)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            chunks = []
            async for chunk in router.stream_complete(
                {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                RouteContext("default", None, frozenset(), frozenset(), ()),
                "test",
            ):
                chunks.append(chunk)

    asyncio.run(run())
    stats = store.stats("t1")
    assert stats.prompt_tokens == 80
    assert stats.completion_tokens == 40
    assert stats.total_tokens == 120
    store.close()


def test_router_no_usage_does_not_break(tmp_path: Path) -> None:
    """Upstream without ``usage`` field doesn't break the router."""
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=1),
        targets=(
            TargetConfig("t1", "T1", "http://router/v1", "m",
                         local=True, priority=1),
        ),
    )
    store = Store(config.database, ["t1"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "ok",
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            await router.complete(
                {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
                RouteContext("default", None, frozenset(), frozenset(), ()),
                "test",
            )

    asyncio.run(run())
    stats = store.stats("t1")
    assert stats.prompt_tokens == 0
    assert stats.completion_tokens == 0
    assert stats.total_tokens == 0
    store.close()
