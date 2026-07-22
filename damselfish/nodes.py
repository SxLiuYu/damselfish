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
    # Preserve max_context from existing when not provided in payload
    max_context_raw = payload.get("max_context")
    if max_context_raw is None and existing and existing.get("max_context") is not None:
        max_context_raw = existing.get("max_context")
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
        "max_context": _optional_int(max_context_raw, "上下文上限"),
    }


def draft_target(payload: dict[str, Any], existing: dict[str, Any] | None = None) -> TargetConfig:
    return target_from_mapping(normalize_node(payload, existing=existing), managed=True)


def public_node(target: TargetConfig, *, managed: bool) -> dict[str, Any]:
    base_url = target.base_url.rstrip("/")
    return {
        "id": target.id,
        "label": target.label,
        "base_url": base_url,
        "models_url": f"{base_url}/models",
        "chat_url": target.chat_url,
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
        "max_context": target.max_context,
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
                "max_tokens": 256,
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
        if not isinstance(body, dict):
            raise ValueError("响应不是 JSON 对象")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("响应中没有 choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("响应中的 choice 格式无效")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("响应中没有 assistant 消息")
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        function_call = message.get("function_call")
        reasoning_content = message.get("reasoning_content")
        if not (content or tool_calls or function_call or reasoning_content):
            raise ValueError("响应中没有可用的 assistant 消息")
        if content:
            result_message = str(content)
        elif tool_calls or function_call:
            result_message = "已返回工具调用"
        else:
            result_message = "已返回推理内容"
        return {
            "success": True,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 1),
            "model": body.get("model", target.model),
            "message": result_message[:300],
        }
    except httpx.TimeoutException:
        return {
            "success": False,
            "status": 504,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "message": "连接或响应超时",
        }
    except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError) as error:
        return {
            "success": False,
            "status": 502,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "message": f"上游响应无效：{str(error)[:300]}",
        }


async def discover_models(client: httpx.AsyncClient, target: TargetConfig) -> dict[str, Any]:
    headers: dict[str, str] = {}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"
    base = target.base_url.rstrip("/")
    models_url = f"{base}/models"
    started = time.monotonic()
    try:
        response = await client.get(models_url, headers=headers, timeout=10.0)
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
        model_details = []
        for item in data:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            model_id = str(item["id"])
            automatic = model_id == "auto"
            upstream_base_url = _optional_url(item.get("upstream_base_url"))
            upstream_chat_url = _optional_url(
                item.get("upstream_chat_url"), preserve_path=True
            )
            if not automatic:
                upstream_base_url = upstream_base_url or base
                upstream_chat_url = upstream_chat_url or target.chat_url
            owned_by = str(item.get("owned_by", "unknown"))
            model_details.append({
                "id": model_id,
                "owned_by": owned_by,
                "provider": str(item.get("provider") or owned_by),
                "upstream_base_url": upstream_base_url,
                "upstream_chat_url": upstream_chat_url,
                "access_base_url": base,
                "access_chat_url": target.chat_url,
                "request_url": upstream_chat_url or target.chat_url,
                "automatic": automatic,
            })
            if len(model_details) == 500:
                break
        return {
            "success": True,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 1),
            "models_url": models_url,
            "chat_url": target.chat_url,
            "models": [model["id"] for model in model_details],
            "model_details": model_details,
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


def _optional_url(value: Any, *, preserve_path: bool = False) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        normalized = _validate_base_url(value.strip())
    except NodeValidationError:
        return None
    return value.strip().rstrip("/") if preserve_path else normalized


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


def _optional_int(value: Any, name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise NodeValidationError(f"{name}必须是整数或空值") from error


def _response_error(response: httpx.Response) -> str:
    try:
        body = response.json()
        error = body.get("error", body) if isinstance(body, dict) else body
        if isinstance(error, dict):
            return str(error.get("message", error))[:500]
        return str(error)[:500]
    except (ValueError, TypeError):
        return response.text[:500] or f"HTTP {response.status_code}"
