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
