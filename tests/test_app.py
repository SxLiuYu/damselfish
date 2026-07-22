import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from damselfish.app import create_app
from damselfish.config import AppConfig, RoutingConfig, TargetConfig
from damselfish.router import CompletionResult, ModelRouter
from damselfish.store import Store


def test_project_memory_api_and_context_injection(
    tmp_path: Path, monkeypatch
) -> None:
    target = TargetConfig(
        "local", "Local", "http://unused/v1", "local-model", local=True, probe=False
    )
    config = AppConfig(
        host="127.0.0.1",
        port=18086,
        database=tmp_path / "app.db",
        routing=RoutingConfig(),
        targets=(target,),
    )
    seen_messages: list[list[dict[str, Any]]] = []

    async def complete(self, payload, context, session_id):
        seen_messages.append(payload["messages"])
        body = {
            "id": "completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "saved"},
                    "finish_reason": "stop",
                }
            ],
        }
        return CompletionResult(body=body, target=target, latency_ms=1.0)

    monkeypatch.setattr(ModelRouter, "complete", complete)
    app = create_app(config)
    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            json={
                "model": "damselfish/auto",
                "damselfish": {
                    "project_id": "damselfish",
                    "session_id": "architecture",
                },
                "messages": [{"role": "user", "content": "Use SQLite"}],
            },
        )
        second = client.post(
            "/v1/chat/completions",
            json={
                "model": "damselfish/auto",
                "damselfish": {
                    "project_id": "damselfish",
                    "session_id": "deployment",
                },
                "messages": [{"role": "user", "content": "Deploy it"}],
            },
        )
        projects = client.get("/v1/memory/projects").json()["data"]
        sessions = client.get(
            "/v1/memory/projects/damselfish/sessions"
        ).json()["data"]
        deployment = client.get(
            "/v1/memory/projects/damselfish/sessions/deployment"
        ).json()

    assert first.status_code == 200
    assert second.headers["X-Damselfish-Project"] == "damselfish"
    assert second.headers["X-Damselfish-Memory-Sync"] == "disabled"
    assert seen_messages[1][0]["role"] == "system"
    assert "Use SQLite" in seen_messages[1][0]["content"]
    assert projects[0]["session_count"] == 2
    assert len(sessions) == 2
    assert deployment["messages"][0]["content"] == "Deploy it"
    assert all(message["role"] != "system" for message in deployment["messages"])


def test_streaming_chat_completion(tmp_path: Path, monkeypatch) -> None:
    """Streaming request returns SSE chunks and saves memory."""
    target = TargetConfig(
        "local", "Local", "http://unused/v1", "local-model", local=True, probe=False
    )
    config = AppConfig(
        host="127.0.0.1",
        port=18086,
        database=tmp_path / "app.db",
        routing=RoutingConfig(),
        targets=(target,),
    )

    async def stream_complete(self, payload, context, session_id):
        # Simulate streaming chunks
        from damselfish.router import CompletionResult
        chunks = [
            {"id": "x", "object": "chat.completion.chunk", "created": 1,
             "model": "local-model",
             "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
            {"id": "x", "object": "chat.completion.chunk", "created": 1,
             "model": "local-model",
             "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]},
            {"id": "x", "object": "chat.completion.chunk", "created": 1,
             "model": "local-model",
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        for chunk in chunks:
            yield chunk
        self._stream_result = CompletionResult(
            body={"choices": [{"message": {"content": "hello"}}]},
            target=target, latency_ms=1.0,
        )

    monkeypatch.setattr(ModelRouter, "stream_complete", stream_complete)

    app = create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "damselfish/auto",
                "stream": True,
                "damselfish": {
                    "project_id": "stream-test",
                    "session_id": "sess1",
                },
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = response.text
        assert "data: " in body
        assert "[DONE]" in body
        assert "hello" in body
        # Verify memory was saved
        session = client.get(
            "/v1/memory/projects/stream-test/sessions/sess1"
        ).json()
        assert any(
            m.get("content") == "hello" for m in session["messages"]
            if m.get("role") == "assistant"
        )


def test_streaming_chat_completion_with_historical_stats_columns(
    tmp_path: Path,
) -> None:
    target = TargetConfig(
        "local", "Local", "http://unused/v1", "local-model", local=True, probe=False
    )
    config = AppConfig(
        host="127.0.0.1",
        port=18086,
        database=tmp_path / "app.db",
        routing=RoutingConfig(),
        targets=(target,),
    )
    store = Store(config.database, [target.id])
    store.close()
    # Simulate a legacy database without token columns
    with sqlite3.connect(config.database) as connection:
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
        connection.execute("INSERT INTO target_stats(target_id) VALUES (?)", (target.id,))

    upstream_body = (
        'data: {"id":"x","choices":[{"delta":{"role":"assistant"},'
        '"finish_reason":null}]}\n\n'
        'data: {"id":"x","choices":[{"delta":{"content":"OK"},'
        '"finish_reason":null}]}\n\n'
        'data: {"id":"x","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    ).encode()

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=upstream_body)

    app = create_app(config)
    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    with TestClient(app) as client:
        app.state.router.client = upstream_client
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "damselfish/auto",
                "stream": True,
                "messages": [{"role": "user", "content": "Reply with OK"}],
            },
        )

    asyncio.run(upstream_client.aclose())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"finish_reason": "stop"' in response.text
    assert "data: [DONE]" in response.text


def test_auto_fallback_to_longer_context_on_overflow(tmp_path: Path) -> None:
    """End-to-end: short-context target returns 400 overflow, router falls
    back to a long-context target automatically."""
    short_target = TargetConfig(
        "short", "Short", "http://short/v1", "short",
        local=True, priority=1, max_context=4096, probe=False,
    )
    long_target = TargetConfig(
        "long", "Long", "http://long/v1", "long",
        local=True, priority=2, max_context=128000, probe=False,
    )
    config = AppConfig(
        host="127.0.0.1", port=18086, database=tmp_path / "app.db",
        routing=RoutingConfig(priority_weight_ms=1),
        targets=(short_target, long_target),
    )

    def upstream(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        model = payload["model"]
        if model == "short":
            return httpx.Response(
                400,
                json={"error": {"message": "`inputs` tokens + `max_new_tokens` must be <= 4096"}},
            )
        return httpx.Response(
            200,
            json={"id": "ok", "choices": [{"message": {"role": "assistant", "content": "from long"}, "finish_reason": "stop"}]},
        )

    app = create_app(config)
    with TestClient(app) as client:
        app.state.router.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "damselfish/auto",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    assert response.headers["X-Damselfish-Target"] == "long"
    assert response.json()["choices"][0]["message"]["content"] == "from long"


# ── _compress_conversation tests ────────────────────────────────────


def test_compress_conversation_removes_hardcoded_target(tmp_path: Path) -> None:
    """Compression uses auto-routing, not hardcoded deepseek-v4-flash."""
    from damselfish.app import _compress_conversation
    from damselfish.store import Store
    from damselfish.router import ModelRouter

    target = TargetConfig(
        "fast", "Fast", "http://router/v1", "fast-model",
        local=True, priority=1, probe=False,
    )
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=1),
        targets=(target,),
    )
    store = Store(config.database, ["fast"])

    # Build a long conversation that would trigger compression
    messages = [{"role": "user", "content": f"msg {i}"} for i in range(35)]
    messages.append({"role": "assistant", "content": "ok"})

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        content = payload["messages"][0]["content"]
        # Verify it's a Chinese summary prompt, not hardcoded target
        assert "对话" in content or "summarize" in content.lower()
        return httpx.Response(
            200,
            json={"id": "ok", "choices": [{"message": {"role": "assistant", "content": "用户讨论了多个技术问题"}, "finish_reason": "stop"}]},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            await _compress_conversation(store, router, "test-session", messages, 10)
            # Verify messages were compressed
            saved = store.get_session("test-session", ttl_days=30)
            assert saved is not None
            assert len(saved) < len(messages)

    asyncio.run(run())
    store.close()


def test_compress_conversation_skips_when_no_reduction(tmp_path: Path) -> None:
    """Compression is skipped if token count doesn't decrease."""
    from damselfish.app import _compress_conversation
    from damselfish.store import Store
    from damselfish.router import ModelRouter

    target = TargetConfig(
        "fast", "Fast", "http://router/v1", "fast-model",
        local=True, priority=1, probe=False,
    )
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=1),
        targets=(target,),
    )
    store = Store(config.database, ["fast"])

    # Short conversation — should not compress
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            200,
            json={"id": "ok", "choices": [{"message": {"role": "assistant", "content": "summary"}, "finish_reason": "stop"}]},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            await _compress_conversation(store, router, "short-session", messages, 10)

    asyncio.run(run())
    # Should not have called the LLM (too short to compress)
    assert call_count == 0
    store.close()


def test_compress_conversation_uses_chinese_prompt(tmp_path: Path) -> None:
    """Compression prompt is in Chinese, not English."""
    from damselfish.app import _compress_conversation
    from damselfish.store import Store
    from damselfish.router import ModelRouter

    target = TargetConfig(
        "fast", "Fast", "http://router/v1", "fast-model",
        local=True, priority=1, probe=False,
    )
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=1),
        targets=(target,),
    )
    store = Store(config.database, ["fast"])

    messages = [{"role": "user", "content": f"问题 {i}"} for i in range(35)]
    messages.append({"role": "assistant", "content": "回答"})

    captured_prompt = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        captured_prompt.append(payload["messages"][0]["content"])
        return httpx.Response(
            200,
            json={"id": "ok", "choices": [{"message": {"role": "assistant", "content": "摘要"}, "finish_reason": "stop"}]},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            await _compress_conversation(store, router, "cn-session", messages, 10)

    asyncio.run(run())
    assert len(captured_prompt) == 1
    prompt = captured_prompt[0]
    # Verify Chinese prompt (check for Chinese characters)
    assert any('\u4e00' <= c <= '\u9fff' for c in prompt)
    # Verify it's a summary prompt, not English
    assert "summarize" not in prompt.lower()
    assert "Please" not in prompt
    store.close()


def test_compress_conversation_sets_estimated_input_tokens(tmp_path: Path) -> None:
    """Compression RouteContext includes estimated_input_tokens so that
    max_context filtering works correctly for the compression request."""
    from damselfish.app import _compress_conversation
    from damselfish.store import Store
    from damselfish.router import ModelRouter

    # A target with a small context window
    small_target = TargetConfig(
        "small", "Small", "http://router/v1", "small-model",
        local=True, priority=1, probe=False, max_context=4096,
    )
    # A target with a large context window
    large_target = TargetConfig(
        "large", "Large", "http://router/v1", "large-model",
        local=True, priority=2, probe=False, max_context=128000,
    )
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=1),
        targets=(small_target, large_target),
    )
    store = Store(config.database, ["small", "large"])

    # Build a long conversation that would trigger compression
    # Use long messages so the compression prompt exceeds the small target's
    # 4096-token context window, forcing routing to the large target.
    messages = [{"role": "user", "content": "中" * 200 + f" msg {i}"} for i in range(35)]
    messages.append({"role": "assistant", "content": "ok"})

    captured_model = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        captured_model.append(payload.get("model"))
        return httpx.Response(
            200,
            json={"id": "ok", "choices": [{"message": {"role": "assistant", "content": "摘要"}, "finish_reason": "stop"}]},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            await _compress_conversation(store, router, "est-session", messages, 10)

    asyncio.run(run())
    # Should have routed to the large-context target, not the small one
    assert captured_model == ["large-model"], f"expected large-model, got {captured_model}"
    store.close()


def test_compress_conversation_requires_chat_capability(tmp_path: Path) -> None:
    """Compression request requires 'chat' capability so that non-chat targets
    are filtered out."""
    from damselfish.app import _compress_conversation
    from damselfish.store import Store
    from damselfish.router import ModelRouter

    # A target without chat capability
    non_chat_target = TargetConfig(
        "embed", "Embed", "http://router/v1", "embed-model",
        local=True, priority=1, probe=False,
        capabilities=frozenset({"fast"}),
    )
    # A target with chat capability
    chat_target = TargetConfig(
        "chat", "Chat", "http://router/v1", "chat-model",
        local=True, priority=2, probe=False,
        capabilities=frozenset({"chat", "fast"}),
    )
    config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=1),
        targets=(non_chat_target, chat_target),
    )
    store = Store(config.database, ["embed", "chat"])

    messages = [{"role": "user", "content": f"msg {i}"} for i in range(35)]
    messages.append({"role": "assistant", "content": "ok"})

    captured_model = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        captured_model.append(payload.get("model"))
        return httpx.Response(
            200,
            json={"id": "ok", "choices": [{"message": {"role": "assistant", "content": "摘要"}, "finish_reason": "stop"}]},
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = ModelRouter(config, store, client)
            await _compress_conversation(store, router, "cap-session", messages, 10)

    asyncio.run(run())
    # Should have routed to the chat target, not the non-chat one
    assert captured_model == ["chat-model"], f"expected chat-model, got {captured_model}"
    store.close()


def test_streaming_meta_event_injected(tmp_path: Path, monkeypatch) -> None:
    """Streaming response includes a meta event before the first data chunk."""
    target = TargetConfig(
        "local", "Local", "http://unused/v1", "local-model", local=True, probe=False
    )
    config = AppConfig(
        host="127.0.0.1", port=18086, database=tmp_path / "app.db",
        routing=RoutingConfig(),
        targets=(target,),
    )

    async def stream_complete(self, payload, context, session_id):
        from damselfish.router import CompletionResult
        chunks = [
            {"id": "x", "object": "chat.completion.chunk", "created": 1,
             "model": "local-model",
             "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        ]
        for chunk in chunks:
            yield chunk
        self._stream_result = CompletionResult(
            body={"choices": [{"message": {"content": "ok"}}]},
            target=target, latency_ms=42.5,
        )

    monkeypatch.setattr(ModelRouter, "stream_complete", stream_complete)

    app = create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "damselfish/auto",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200
    body = response.text
    assert "event: meta" in body
    assert "latency_ms" in body
    assert "target" in body
    assert "local-model" in body


def test_health_endpoint_deep_check(tmp_path: Path) -> None:
    """Enhanced /health reports available_targets, healthy_targets, total_targets."""
    target = TargetConfig(
        "local", "Local", "http://unused/v1", "local-model", local=True, probe=False
    )
    config = AppConfig(
        host="127.0.0.1", port=18086, database=tmp_path / "app.db",
        routing=RoutingConfig(),
        targets=(target,),
    )
    app = create_app(config)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "available_targets" in data
    assert "healthy_targets" in data
    assert "total_targets" in data
    assert data["total_targets"] >= 1
