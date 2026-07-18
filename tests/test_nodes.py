import json
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from damselfish.app import create_app
from damselfish.config import AppConfig, RoutingConfig, TargetConfig


def config(tmp_path: Path) -> AppConfig:
    target = TargetConfig(
        "static", "Static", "http://unused/v1", "static-model", local=True, probe=False
    )
    return AppConfig(
        host="127.0.0.1",
        port=18086,
        database=tmp_path / "app.db",
        routing=RoutingConfig(),
        targets=(target,),
        managed_nodes_file=tmp_path / "managed-nodes.json",
    )


def node_payload() -> dict[str, Any]:
    return {
        "id": "free-cloud",
        "label": "Free Cloud",
        "base_url": "https://free.example/v1",
        "model": "free-model",
        "api_key": "upstream-secret",
        "priority": 20,
        "capabilities": ["chat", "chinese", "tools"],
    }


def test_admin_requires_service_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DAMSELFISH_API_KEY", raising=False)
    with TestClient(create_app(config(tmp_path))) as client:
        page = client.get("/admin/nodes")
        response = client.get("/admin/api/nodes")
    assert page.status_code == 200
    assert response.status_code == 503


def test_admin_login_uses_http_only_session_cookie(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DAMSELFISH_API_KEY", "service-secret")
    with TestClient(create_app(config(tmp_path))) as client:
        rejected = client.post("/admin/login", json={"key": "wrong"})
        logged_in = client.post("/admin/login", json={"key": "service-secret"})
        nodes = client.get("/admin/api/nodes")
        logged_out = client.post("/admin/logout")
        rejected_again = client.get("/admin/api/nodes")
    assert rejected.status_code == 401
    assert logged_in.status_code == 200
    assert "HttpOnly" in logged_in.headers["set-cookie"]
    assert "SameSite=strict" in logged_in.headers["set-cookie"]
    assert nodes.status_code == 200
    assert logged_out.status_code == 200
    assert rejected_again.status_code == 401


def test_node_test_save_reload_and_delete(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DAMSELFISH_API_KEY", "service-secret")

    async def upstream(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer upstream-secret"
        return httpx.Response(
            200,
            json={
                "model": "free-model",
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            },
        )

    app = create_app(config(tmp_path))
    with TestClient(app) as client:
        app.state.router.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        unauthorized = client.get("/admin/api/nodes")
        tested = client.post(
            "/admin/api/nodes/test",
            headers={"Authorization": "Bearer service-secret"},
            json=node_payload(),
        )
        created = client.post(
            "/admin/api/nodes",
            headers={"Authorization": "Bearer service-secret"},
            json=node_payload(),
        )
        listed = client.get(
            "/admin/api/nodes",
            headers={"Authorization": "Bearer service-secret"},
        )
        models = client.get(
            "/v1/models", headers={"Authorization": "Bearer service-secret"}
        )
        deleted = client.delete(
            "/admin/api/nodes/free-cloud",
            headers={"Authorization": "Bearer service-secret"},
        )

    assert unauthorized.status_code == 401
    assert tested.json()["success"] is True
    assert created.status_code == 201
    assert created.json()["data"]["has_api_key"] is True
    free_cloud = next(node for node in listed.json()["data"] if node["id"] == "free-cloud")
    assert "api_key" not in free_cloud
    assert free_cloud["managed"] is True
    assert any(model["id"] == "free-cloud" for model in models.json()["data"])
    assert deleted.json()["deleted"] is True
    saved = (tmp_path / "managed-nodes.json").read_text(encoding="utf-8")
    assert "free-cloud" not in saved


def test_node_test_accepts_reasoning_content(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DAMSELFISH_API_KEY", "service-secret")

    async def upstream(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["max_tokens"] >= 128
        return httpx.Response(
            200,
            json={
                "model": "reasoning-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "OK",
                        },
                    }
                ],
            },
        )

    app = create_app(config(tmp_path))
    with TestClient(app) as client:
        app.state.router.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        response = client.post(
            "/admin/api/nodes/test",
            headers={"Authorization": "Bearer service-secret"},
            json=node_payload(),
        )

    result = response.json()
    assert result["success"] is True
    assert result["message"] == "已返回推理内容"
    assert result["latency_ms"] >= 0


def test_node_test_failure_includes_latency(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DAMSELFISH_API_KEY", "service-secret")

    async def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": ""}}]},
        )

    app = create_app(config(tmp_path))
    with TestClient(app) as client:
        app.state.router.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        response = client.post(
            "/admin/api/nodes/test",
            headers={"Authorization": "Bearer service-secret"},
            json=node_payload(),
        )

    result = response.json()
    assert result["success"] is False
    assert result["status"] == 502
    assert result["latency_ms"] >= 0


def test_edit_keeps_existing_upstream_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DAMSELFISH_API_KEY", "service-secret")
    headers = {"Authorization": "Bearer service-secret"}
    app = create_app(config(tmp_path))
    with TestClient(app) as client:
        client.post("/admin/api/nodes", headers=headers, json=node_payload())
        edited = node_payload()
        edited["label"] = "Renamed"
        edited["api_key"] = ""
        response = client.put("/admin/api/nodes/free-cloud", headers=headers, json=edited)
    assert response.status_code == 200
    assert response.json()["data"]["label"] == "Renamed"
    assert "upstream-secret" in (tmp_path / "managed-nodes.json").read_text()


def test_node_model_details_include_urls_and_owner(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DAMSELFISH_API_KEY", "service-secret")
    headers = {"Authorization": "Bearer service-secret"}

    async def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://free.example/v1/models"
        assert request.headers["authorization"] == "Bearer upstream-secret"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "auto",
                        "owned_by": "router",
                        "provider": "LLM Router",
                        "upstream_base_url": None,
                        "upstream_chat_url": None,
                    },
                    {
                        "id": "free-model",
                        "owned_by": "finna",
                        "provider": "Finna",
                        "upstream_base_url": "https://www.finna.com.cn/v1",
                        "upstream_chat_url": "https://www.finna.com.cn/v1/chat/completions",
                    },
                ],
            },
        )

    app = create_app(config(tmp_path))
    with TestClient(app) as client:
        client.post("/admin/api/nodes", headers=headers, json=node_payload())
        app.state.router.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        listed = client.get("/admin/api/nodes", headers=headers)
        response = client.get("/admin/api/nodes/free-cloud/models", headers=headers)

    node = next(item for item in listed.json()["data"] if item["id"] == "free-cloud")
    assert node["models_url"] == "https://free.example/v1/models"
    assert node["chat_url"] == "https://free.example/v1/chat/completions"
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "status": 200,
        "latency_ms": response.json()["latency_ms"],
        "models_url": "https://free.example/v1/models",
        "chat_url": "https://free.example/v1/chat/completions",
        "models": ["auto", "free-model"],
        "model_details": [
            {
                "id": "auto",
                "owned_by": "router",
                "provider": "LLM Router",
                "upstream_base_url": None,
                "upstream_chat_url": None,
                "access_base_url": "https://free.example/v1",
                "access_chat_url": "https://free.example/v1/chat/completions",
                "request_url": "https://free.example/v1/chat/completions",
                "automatic": True,
            },
            {
                "id": "free-model",
                "owned_by": "finna",
                "provider": "Finna",
                "upstream_base_url": "https://www.finna.com.cn/v1",
                "upstream_chat_url": "https://www.finna.com.cn/v1/chat/completions",
                "access_base_url": "https://free.example/v1",
                "access_chat_url": "https://free.example/v1/chat/completions",
                "request_url": "https://www.finna.com.cn/v1/chat/completions",
                "automatic": False,
            },
        ],
    }


def test_node_model_details_fall_back_for_standard_openai_response(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DAMSELFISH_API_KEY", "service-secret")
    headers = {"Authorization": "Bearer service-secret"}

    async def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "free-model", "owned_by": "cloud"}]},
        )

    app = create_app(config(tmp_path))
    with TestClient(app) as client:
        client.post("/admin/api/nodes", headers=headers, json=node_payload())
        app.state.router.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        response = client.get("/admin/api/nodes/free-cloud/models", headers=headers)

    model = response.json()["model_details"][0]
    assert model["provider"] == "cloud"
    assert model["upstream_base_url"] == "https://free.example/v1"
    assert model["upstream_chat_url"] == "https://free.example/v1/chat/completions"
    assert model["access_chat_url"] == "https://free.example/v1/chat/completions"


def test_node_model_discovery_failure_does_not_break_node_list(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DAMSELFISH_API_KEY", "service-secret")
    headers = {"Authorization": "Bearer service-secret"}

    async def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    app = create_app(config(tmp_path))
    with TestClient(app) as client:
        client.post("/admin/api/nodes", headers=headers, json=node_payload())
        app.state.router.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        models = client.get("/admin/api/nodes/free-cloud/models", headers=headers)
        listed = client.get("/admin/api/nodes", headers=headers)

    assert models.status_code == 200
    assert models.json()["success"] is False
    assert models.json()["status"] == 404
    assert listed.status_code == 200
    assert any(item["id"] == "free-cloud" for item in listed.json()["data"])
