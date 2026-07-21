from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from damselfish.app import create_app
from damselfish.config import AppConfig, RoutingConfig, TargetConfig
from damselfish.router import CompletionResult, ModelRouter


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
