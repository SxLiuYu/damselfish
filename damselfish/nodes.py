from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .config import TargetConfig, target_from_mapping

_NODE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$")
_ALLOWED_CAPABILITIES = {
    "chat", "chinese", "multilingual", "tools", "coding", "reasoning",
    "creative", "fast", "vision",
}


class NodeValidationError(ValueError):
    pass


class ManagedNodeStore:
    def __init__(self, path: Path | None) -> None:
        self.path = path

    def load_raw(self) -> list[dict[str, Any]]:
        if self.path is None or not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        nodes = document.get("nodes", []) if isinstance(document, dict) else []
        if not isinstance(nodes, list):
            raise NodeValidationError("节点文件格式无效")
        return [dict(node) for node in nodes]

    def targets(self) -> tuple[TargetConfig, ...]:
        return tuple(target_from_mapping(node, managed=True) for node in self.load_raw())

    def upsert(self, payload: dict[str, Any], node_id: str | None = None) -> TargetConfig:
        nodes = self.load_raw()
        existing = next((node for node in nodes if node.get("id") == node_id), None)
        normalized = normalize_node(payload, existing=existing, forced_id=node_id)
        duplicate = next(
            (node for node in nodes if node.get("id") == normalized["id"] and node is not existing),
            None,
        )
        if duplicate:
            raise NodeValidationError("节点 ID 已存在")
        if existing is None:
            nodes.append(normalized)
        else:
            nodes[nodes.index(existing)] = normalized
        self._write(nodes)
        return target_from_mapping(normalized, managed=True)

    def delete(self, node_id: str) -> bool:
        nodes = self.load_raw()
        filtered = [node for node in nodes if node.get("id") != node_id]
        if len(filtered) == len(nodes):
            return False
        self._write(filtered)
        return True

    def _write(self, nodes: list[dict[str, Any]]) -> None:
        if self.path is None:
            raise NodeValidationError("未配置节点持久化文件")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "nodes": nodes}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)
        os.chmod(self.path, 0o600)


def normalize_node(
    payload: dict[str, Any],
    *,
    existing: dict[str, Any] | None = None,
    forced_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise NodeValidationError("请求必须是 JSON 对象")
    node_id = str(forced_id or payload.get("id", "")).strip()
    if not _NODE_ID.fullmatch(node_id):
        raise NodeValidationError("节点 ID 只能包含字母、数字、点、下划线和短横线")
    label = _required_text(payload, "label", 120)
    base_url = _validate_base_url(_required_text(payload, "base_url", 500))
    model = _required_text(payload, "model", 200)
    api_key = str(payload.get("api_key", "")).strip()
    if not api_key and existing:
        api_key = str(existing.get("api_key", ""))
    capabilities = _string_list(payload.get("capabilities", ["chat"]), "能力")
    unknown = set(capabilities) - _ALLOWED_CAPABILITIES
    if unknown:
        raise NodeValidationError(f"未知能力：{', '.join(sorted(unknown))}")
    if "chat" not in capabilities:
        capabilities.insert(0, "chat")
    return {
        "id": node_id,
        "label": label,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "enabled": bool(payload.get("enabled", True)),
        "local": False,
        "free": bool(payload.get("free", True)),
        "priority": _bounded_int(payload.get("priority", 100), "优先级", 0, 10000),
        "capabilities": capabilities,
        "scenarios": _string_list(payload.get("scenarios", []), "场景"),
        "personas": _string_list(payload.get("personas", []), "人物"),
        "probe": bool(payload.get("probe", True)),
        "probe_prompt": str(payload.get("probe_prompt", "Reply OK")).strip()[:200] or "Reply OK",
        "max_concurrency": _bounded_int(
            payload.get("max_concurrency", 4), "最大并发", 1, 100
        ),
    }


def draft_target(payload: dict[str, Any], existing: dict[str, Any] | None = None) -> TargetConfig:
    return target_from_mapping(normalize_node(payload, existing=existing), managed=True)


def public_node(target: TargetConfig, *, managed: bool) -> dict[str, Any]:
    return {
        "id": target.id,
        "label": target.label,
        "base_url": target.base_url,
        "model": target.model,
        "enabled": target.enabled,
        "available": target.available,
        "local": target.local,
        "free": target.free,
        "priority": target.priority,
        "capabilities": sorted(target.capabilities),
        "scenarios": sorted(target.scenarios),
        "personas": sorted(target.personas),
        "probe": target.probe,
        "max_concurrency": target.max_concurrency,
        "has_api_key": bool(target.api_key),
        "managed": managed,
    }


async def test_node(client: httpx.AsyncClient, target: TargetConfig) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"
    started = time.monotonic()
    try:
        response = await client.post(
            target.chat_url,
            headers=headers,
            json={
                "model": target.model,
                "messages": [{"role": "user", "content": "仅回复 OK"}],
                "stream": False,
                "max_tokens": 16,
                "temperature": 0,
            },
        )
        latency_ms = (time.monotonic() - started) * 1000
        if not response.is_success:
            return {
                "success": False,
                "status": response.status_code,
                "latency_ms": round(latency_ms, 1),
                "message": _response_error(response),
            }
        body = response.json()
        if isinstance(body.get("data"), dict):
            body = body["data"]
        choices = body.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content")
        if not content and message.get("tool_calls"):
            content = "已返回工具调用"
        if not choices or not (content or message.get("tool_calls")):
            raise ValueError("响应中没有可用的 assistant 消息")
        return {
            "success": True,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 1),
            "model": body.get("model", target.model),
            "message": str(content or "已返回工具调用")[:300],
        }
    except httpx.TimeoutException:
        return {"success": False, "status": 504, "message": "连接或响应超时"}
    except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError) as error:
        return {"success": False, "status": 502, "message": f"上游响应无效：{str(error)[:300]}"}


async def discover_models(client: httpx.AsyncClient, target: TargetConfig) -> dict[str, Any]:
    headers: dict[str, str] = {}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"
    base = target.base_url.rstrip("/")
    models_url = f"{base}/models"
    started = time.monotonic()
    try:
        response = await client.get(models_url, headers=headers)
        latency_ms = (time.monotonic() - started) * 1000
        if not response.is_success:
            return {
                "success": False,
                "status": response.status_code,
                "latency_ms": round(latency_ms, 1),
                "message": _response_error(response),
            }
        body = response.json()
        data = body.get("data", []) if isinstance(body, dict) else []
        models = [str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")]
        return {
            "success": True,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 1),
            "models": models[:500],
        }
    except httpx.TimeoutException:
        return {"success": False, "status": 504, "message": "连接或响应超时"}
    except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError) as error:
        return {"success": False, "status": 502, "message": f"上游响应无效：{str(error)[:300]}"}


def _required_text(payload: dict[str, Any], name: str, maximum: int) -> str:
    value = str(payload.get(name, "")).strip()
    if not value:
        raise NodeValidationError(f"{name} 不能为空")
    if len(value) > maximum:
        raise NodeValidationError(f"{name} 过长")
    return value


def _validate_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise NodeValidationError("Base URL 必须是有效的 http/https 地址")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise NodeValidationError("Base URL 不能包含账号、查询参数或片段")
    url = value.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    return url.rstrip("/")


def _string_list(value: Any, name: str) -> list[str]:
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list):
        items = value
    else:
        raise NodeValidationError(f"{name}必须是数组或逗号分隔文本")
    return list(dict.fromkeys(str(item).strip().lower() for item in items if str(item).strip()))


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise NodeValidationError(f"{name}必须是整数") from error
    if number < minimum or number > maximum:
        raise NodeValidationError(f"{name}必须在 {minimum} 到 {maximum} 之间")
    return number


def _response_error(response: httpx.Response) -> str:
    try:
        body = response.json()
        error = body.get("error", body) if isinstance(body, dict) else body
        if isinstance(error, dict):
            return str(error.get("message", error))[:500]
        return str(error)[:500]
    except (ValueError, TypeError):
        return response.text[:500] or f"HTTP {response.status_code}"
