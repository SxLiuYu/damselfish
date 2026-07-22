from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from .config import AppConfig, load_config
from .git_sync import GitMemorySync
from .nodes import (
    ManagedNodeStore,
    NodeValidationError,
    discover_models,
    draft_target,
    public_node,
    test_node,
)
from .router import ModelRouter, NoTargetAvailable
from .selector import infer_context, RouteContext
from .store import Store, merge_messages, project_context_message

log = logging.getLogger("damselfish")


def create_app(config: AppConfig | None = None, config_path: str | Path | None = None) -> FastAPI:
    loaded = config or load_config(config_path)
    node_store = ManagedNodeStore(loaded.managed_nodes_file)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = Store(loaded.database, [target.id for target in loaded.targets])
        timeout = httpx.Timeout(
            loaded.routing.request_timeout_seconds,
            connect=loaded.routing.connect_timeout_seconds,
        )
        client = httpx.AsyncClient(timeout=timeout)
        router = ModelRouter(loaded, store, client)
        memory_sync = GitMemorySync(loaded.git_sync, store)
        await memory_sync.startup_sync()
        stop = asyncio.Event()
        probe_task = asyncio.create_task(router.probe_loop(stop))
        sync_task = asyncio.create_task(memory_sync.sync_loop(stop))
        app.state.config = loaded
        app.state.store = store
        app.state.router = router
        app.state.memory_sync = memory_sync
        app.state.started_at = time.time()
        app.state.node_store = node_store
        try:
            yield
        finally:
            stop.set()
            # Give in-flight requests up to 10 seconds to finish gracefully.
            try:
                await asyncio.wait_for(
                    asyncio.gather(probe_task, sync_task),
                    timeout=10.0,
                )
            except TimeoutError:
                log.warning("graceful shutdown timed out after 10s, forcing exit")
                probe_task.cancel()
                sync_task.cancel()
            await memory_sync.sync_pending(force=True)
            await client.aclose()
            store.close()

    app = FastAPI(title="Damselfish", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        expected = os.environ.get("DAMSELFISH_API_KEY")
        api_request = request.url.path.startswith("/v1/")
        admin_request = request.url.path.startswith("/admin/api/")
        if admin_request and not expected:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "DAMSELFISH_API_KEY is not configured"}},
            )
        if expected and (api_request or admin_request):
            supplied = request.headers.get("authorization", "")
            bearer_valid = hmac.compare_digest(supplied, f"Bearer {expected}")
            session_valid = admin_request and _valid_admin_session(
                request.cookies.get("damselfish_admin"), expected
            )
            if not bearer_valid and not session_valid:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "invalid Damselfish API key"}},
                )
        return await call_next(request)

    def refresh_managed_nodes(app: FastAPI) -> None:
        nonlocal loaded
        managed_targets = app.state.node_store.targets()
        static_targets = tuple(
            target for target in loaded.targets if target.id not in loaded.managed_target_ids
        )
        targets = static_targets + managed_targets
        ids = [target.id for target in targets]
        if len(ids) != len(set(ids)):
            raise NodeValidationError("节点 ID 与静态配置冲突")
        loaded = replace(
            loaded,
            targets=targets,
            managed_target_ids=frozenset(target.id for target in managed_targets),
        )
        app.state.config = loaded
        app.state.store.ensure_targets(ids)
        app.state.router.reconfigure(loaded)

    @app.get("/admin/nodes", include_in_schema=False)
    async def nodes_page() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "nodes.html")

    @app.post("/admin/login", include_in_schema=False)
    async def admin_login(request: Request) -> JSONResponse:
        expected = os.environ.get("DAMSELFISH_API_KEY", "")
        if not expected:
            raise HTTPException(status_code=503, detail="服务器尚未配置管理 Key")
        payload = await _json_object(request)
        supplied = str(payload.get("key", ""))
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="北京服务 Key 不正确")
        response = JSONResponse({"authenticated": True})
        response.set_cookie(
            "damselfish_admin",
            _admin_session_token(expected),
            max_age=12 * 60 * 60,
            httponly=True,
            secure=_request_is_https(request),
            samesite="strict",
            path="/",
        )
        return response

    @app.post("/admin/logout", include_in_schema=False)
    async def admin_logout() -> Response:
        response = JSONResponse({"authenticated": False})
        response.delete_cookie("damselfish_admin", path="/")
        return response

    @app.get("/admin/api/nodes")
    async def list_nodes(request: Request) -> dict[str, Any]:
        states = request.app.state.store.all_stats()
        return {
            "data": [
                {
                    **public_node(target, managed=target.id in loaded.managed_target_ids),
                    "stats": states[target.id].public(),
                }
                for target in loaded.targets
            ]
        }

    @app.post("/admin/api/nodes/test")
    async def test_node_draft(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        existing = _managed_node(request, str(payload.get("id", "")), optional=True)
        try:
            target = draft_target(payload, existing=existing)
        except NodeValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return await test_node(request.app.state.router.client, target)

    @app.post("/admin/api/nodes/discover")
    async def discover_node_models(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        existing = _managed_node(request, str(payload.get("id", "")), optional=True)
        try:
            target = draft_target(payload, existing=existing)
        except NodeValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return await discover_models(request.app.state.router.client, target)

    @app.get("/admin/api/nodes/{node_id}/models")
    async def node_models(node_id: str, request: Request) -> dict[str, Any]:
        target = next((item for item in loaded.targets if item.id == node_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail="节点不存在")
        return await discover_models(request.app.state.router.client, target)

    @app.post("/admin/api/nodes", status_code=201)
    async def create_node(request: Request) -> dict[str, Any]:
        payload = await _json_object(request)
        node_id = str(payload.get("id", "")).strip()
        if any(target.id == node_id for target in loaded.targets):
            raise HTTPException(status_code=409, detail="节点 ID 已存在")
        try:
            target = request.app.state.node_store.upsert(payload)
            refresh_managed_nodes(request.app)
        except NodeValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"data": public_node(target, managed=True)}

    @app.put("/admin/api/nodes/{node_id}")
    async def update_node(node_id: str, request: Request) -> dict[str, Any]:
        _managed_node(request, node_id)
        payload = await _json_object(request)
        try:
            target = request.app.state.node_store.upsert(payload, node_id=node_id)
            refresh_managed_nodes(request.app)
        except NodeValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"data": public_node(target, managed=True)}

    @app.delete("/admin/api/nodes/{node_id}")
    async def delete_node(node_id: str, request: Request) -> dict[str, Any]:
        _managed_node(request, node_id)
        request.app.state.node_store.delete(node_id)
        refresh_managed_nodes(request.app)
        return {"deleted": True, "id": node_id}

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        stats = request.app.state.store.all_stats()
        now = time.time()
        available_targets = []
        healthy_targets = []
        for target in loaded.targets:
            if not target.available:
                continue
            state = stats[target.id]
            if state.circuit_open_until > now:
                continue
            available_targets.append(target.id)
            # A target is "healthy" if it has had a recent successful request
            # or probe (within 2x the probe stale interval).
            recent_success = state.last_success_at and (now - state.last_success_at) < loaded.routing.probe_stale_seconds * 2
            if recent_success or state.successes == 0:
                healthy_targets.append(target.id)
        return {
            "status": "ok" if healthy_targets else "degraded",
            "uptime_seconds": int(now - request.app.state.started_at),
            "available_targets": available_targets,
            "healthy_targets": healthy_targets,
            "total_targets": len(loaded.targets),
            "memory_sync": request.app.state.memory_sync.status(),
        }

    @app.get("/stats")
    async def stats(request: Request) -> dict[str, Any]:
        states = request.app.state.store.all_stats()
        return {
            "targets": {
                target.id: {
                    "label": target.label,
                    "model": target.model,
                    "local": target.local,
                    "available": target.available,
                    "capabilities": sorted(target.capabilities),
                    **states[target.id].public(),
                }
                for target in loaded.targets
            },
            "recent_decisions": request.app.state.store.recent_decisions(),
            "memory_sync": request.app.state.memory_sync.status(),
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        created = int(time.time())
        entries = [{"id": "damselfish/auto", "object": "model", "created": created, "owned_by": "damselfish"}]
        entries.extend(
            {
                "id": target.id,
                "object": "model",
                "created": created,
                "owned_by": "local" if target.local else "upstream",
            }
            for target in loaded.targets
            if target.available
        )
        return {"object": "list", "data": entries}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        x_damselfish_session: str | None = Header(None),
        x_damselfish_session_title: str | None = Header(None),
        x_damselfish_project: str | None = Header(None),
        x_damselfish_project_title: str | None = Header(None),
        x_damselfish_scenario: str | None = Header(None),
        x_damselfish_persona: str | None = Header(None),
    ):
        try:
            payload = await request.json()
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=400, detail="request body must be JSON") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            raise HTTPException(status_code=400, detail="messages must be an array")

        extension = payload.pop("damselfish", {}) or {}
        if not isinstance(extension, dict):
            raise HTTPException(status_code=400, detail="damselfish must be an object")
        session_id = _identifier(
            x_damselfish_session or extension.get("session_id"), "session_id", optional=True
        )
        project_id = _identifier(
            x_damselfish_project or extension.get("project_id") or "default",
            "project_id",
        )
        project_title = _title(
            x_damselfish_project_title or extension.get("project_title")
        )
        session_title = _title(
            x_damselfish_session_title or extension.get("session_title")
        )
        scenario = x_damselfish_scenario or extension.get("scenario")
        persona = x_damselfish_persona or extension.get("persona")
        memory_enabled = bool(extension.get("memory", True)) and bool(session_id)
        project_memory_enabled = bool(extension.get("project_memory", True))
        incoming = payload["messages"]
        history = []
        transcript = list(incoming)
        if memory_enabled:
            await request.app.state.memory_sync.pull_if_due()
            history = request.app.state.store.get_project_session(
                project_id, session_id, loaded.routing.memory_ttl_days
            )
            transcript = merge_messages(history, incoming)
            payload["messages"] = transcript
            if project_memory_enabled:
                shared = request.app.state.store.project_context(
                    project_id,
                    session_id,
                    loaded.routing.project_memory_session_limit,
                    loaded.routing.project_memory_message_limit,
                )
                context_message = project_context_message(
                    project_id, shared, loaded.routing.project_memory_max_chars
                )
                if context_message:
                    payload["messages"] = [context_message, *transcript]

        context = infer_context(
            loaded, payload["messages"], payload.get("tools"), scenario, persona
        )
        wants_stream = bool(payload.get("stream"))
        try:
            decision_session = f"{project_id}/{session_id}" if session_id else None
            if wants_stream:
                return await _handle_streaming(
                    request, payload, context, decision_session,
                    memory_enabled, transcript, project_id, project_title,
                    session_title, session_id,
                )
            result = await request.app.state.router.complete(
                payload, context, decision_session
            )
        except NoTargetAvailable as error:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": str(error), "type": "router_unavailable"}},
            )

        if memory_enabled:
            assistant = result.body["choices"][0]["message"]
            request.app.state.store.save_session(
                session_id,
                transcript + [assistant],
                loaded.routing.memory_max_messages,
                project_id=project_id,
                project_title=project_title,
                session_title=session_title,
                source_device=request.app.state.memory_sync.device_id,
            )
            await request.app.state.memory_sync.sync_pending()
            # Background compression for long conversations
            if len(transcript) + 1 > loaded.routing.memory_compression_threshold:
                asyncio.create_task(_compress_conversation(
                    request.app.state.store, request.app.state.router,
                    session_id, transcript + [assistant],
                    loaded.routing.memory_compression_keep,
                ))
        headers = {
            "X-Damselfish-Target": result.target.id,
            "X-Damselfish-Model": result.target.model,
            "X-Damselfish-Latency-Ms": f"{result.latency_ms:.1f}",
            "X-Damselfish-Scenario": context.scenario,
            "X-Damselfish-Project": project_id,
            "X-Damselfish-Memory-Sync": request.app.state.memory_sync.response_status(),
        }
        if session_id:
            headers["X-Damselfish-Session"] = session_id
        return JSONResponse(content=result.body, headers=headers)

    @app.get("/v1/memory/projects")
    async def memory_projects(request: Request) -> dict[str, Any]:
        await request.app.state.memory_sync.pull_if_due()
        return {"data": request.app.state.store.list_projects()}

    @app.get("/v1/memory/projects/{project_id}/sessions")
    async def memory_sessions(project_id: str, request: Request) -> dict[str, Any]:
        await request.app.state.memory_sync.pull_if_due()
        return {"data": request.app.state.store.list_project_sessions(project_id)}

    @app.get("/v1/memory/projects/{project_id}/sessions/{session_id}")
    async def memory_session(
        project_id: str, session_id: str, request: Request
    ) -> dict[str, Any]:
        await request.app.state.memory_sync.pull_if_due()
        messages = request.app.state.store.get_project_session(
            project_id, session_id, loaded.routing.memory_ttl_days
        )
        if not messages:
            raise HTTPException(status_code=404, detail="memory session not found")
        return {
            "project_id": project_id,
            "session_id": session_id,
            "messages": messages,
        }

    @app.post("/v1/memory/sync")
    async def memory_sync(request: Request) -> dict[str, Any]:
        success = await request.app.state.memory_sync.sync_now()
        status = request.app.state.memory_sync.status()
        return {"success": success, **status}

    return app




async def _compress_conversation(store, router, session_id, messages, keep):
    """Compress old conversation messages using a lightweight model.

    Fixes:
    - Removed hardcoded preferred_targets (deepseek-v4-flash didn't exist on server)
    - Now uses auto-routing with "fast" preference
    - Verifies compression actually reduced token count
    """
    if not session_id or len(messages) <= keep + 5:
        return
    try:
        old = messages[:-keep]
        recent = messages[-keep:]
        text = "\n".join(
            str(m.get("content", ""))
            for m in old if isinstance(m.get("content"), str) and m.get("content")
        )
        if not text.strip():
            return
        prompt = (
            "请用中文简要总结以下对话。\n"
            "涵盖用户需求、已解决的问题和关键决策。\n"
            "保留足够细节以保持对话连续性。最多 200 字。\n\n" + text
        )
        from .selector import RouteContext
        from .tokens import estimate_messages_tokens
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400, "temperature": 0.5,
        }
        # Use auto-routing with "fast" preference instead of hardcoded target
        ctx = RouteContext(
            scenario="default", persona=None,
            required=frozenset({"chat"}), preferred=frozenset({"fast"}),
            preferred_targets=(),
            estimated_input_tokens=estimate_messages_tokens(payload["messages"]),
        )
        result = await router.complete(payload, ctx, None)
        summary = result.body["choices"][0]["message"].get("content", "")
        if not summary:
            return
        compressed = [
            {"role": "system", "content": "对话摘要：" + summary}
        ] + recent
        # Verify compression actually reduced token count
        old_tokens = estimate_messages_tokens(messages)
        new_tokens = estimate_messages_tokens(compressed)
        if new_tokens >= old_tokens:
            log.info("compression skipped for %s: tokens %d -> %d (no reduction)",
                     session_id[:8], old_tokens, new_tokens)
            return
        store.update_session_messages(session_id, compressed)
        log.info("compressed session %s: %d -> %d messages, tokens %d -> %d",
                 session_id[:8], len(messages), len(compressed), old_tokens, new_tokens)
    except Exception as e:
        log.warning("compression failed for %s: %s", session_id[:8], e)

async def _as_sse(body: dict[str, Any]) -> AsyncIterator[str]:
    choice = body["choices"][0]
    chunk = {
        "id": body.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
        "object": "chat.completion.chunk",
        "created": body.get("created", int(time.time())),
        "model": body.get("model", "damselfish/auto"),
        "choices": [
            {
                "index": choice.get("index", 0),
                "delta": choice["message"],
                "finish_reason": choice.get("finish_reason"),
            }
        ],
    }
    if body.get("usage"):
        chunk["usage"] = body["usage"]
    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def build_default_app() -> FastAPI:
    return create_app()


async def _handle_streaming(
    request: Request,
    payload: dict[str, Any],
    context: RouteContext,
    decision_session: str | None,
    memory_enabled: bool,
    transcript: list[dict[str, Any]],
    project_id: str,
    project_title: str | None,
    session_title: str | None,
    session_id: str | None,
) -> StreamingResponse:
    """Handle a streaming chat completion request.

    Calls ``router.stream_complete()`` and forwards normalized SSE chunks
    to the client.  Accumulates content and saves memory after the stream
    ends.
    """
    router = request.app.state.router
    loaded = request.app.state.config
    accumulated_content: list[str] = []
    accumulated_chars: list[int] = [0]
    _MAX_ACCUMULATED_CHARS = 50000
    first_chunk_time: list[float] = []

    async def stream_chunks() -> AsyncIterator[str]:
        target_id = ""
        target_model = ""
        try:
            first_meta_sent = False
            async for chunk in router.stream_complete(payload, context, decision_session):
                # Track latency from first chunk
                if not first_chunk_time:
                    first_chunk_time.append(time.monotonic())
                # Inject meta event before the first data chunk so clients
                # know which target/model is handling the stream (since SSE
                # headers can't be added after the response starts).
                if not first_meta_sent:
                    result = getattr(router, "_stream_result", None)
                    meta_target = result.target.id if result else ""
                    meta_model = result.target.model if result else ""
                    meta_latency = f"{result.latency_ms:.1f}" if result else "0.0"
                    meta = {
                        "target": meta_target,
                        "model": meta_model,
                        "latency_ms": meta_latency,
                        "scenario": context.scenario,
                    }
                    yield f"event: meta\n"
                    yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
                    first_meta_sent = True
                # Accumulate content for memory (with cap to avoid OOM on huge streams)
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if isinstance(delta, dict) and delta.get("content"):
                        content_piece = delta["content"]
                        if accumulated_chars[0] < _MAX_ACCUMULATED_CHARS:
                            accumulated_content.append(content_piece)
                            accumulated_chars[0] += len(content_piece)
                if chunk.get("model"):
                    target_model = chunk["model"]
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except NoTargetAvailable as error:
            error_chunk = {
                "error": {"message": str(error), "type": "router_unavailable"}
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            return
        yield "data: [DONE]\n\n"
        # Save memory after stream ends
        result = getattr(router, "_stream_result", None)
        if result is not None:
            target_id = result.target.id
            target_model = result.target.model
        if memory_enabled and session_id and accumulated_content:
            assistant = {"role": "assistant", "content": "".join(accumulated_content)}
            request.app.state.store.save_session(
                session_id,
                transcript + [assistant],
                loaded.routing.memory_max_messages,
                project_id=project_id,
                project_title=project_title,
                session_title=session_title,
                source_device=request.app.state.memory_sync.device_id,
            )
            await request.app.state.memory_sync.sync_pending()
            if len(transcript) + 1 > loaded.routing.memory_compression_threshold:
                asyncio.create_task(_compress_conversation(
                    request.app.state.store, request.app.state.router,
                    session_id, transcript + [assistant],
                    loaded.routing.memory_compression_keep,
                ))

    headers = {
        "X-Damselfish-Scenario": context.scenario,
        "X-Damselfish-Project": project_id,
        "X-Damselfish-Memory-Sync": request.app.state.memory_sync.response_status(),
    }
    if session_id:
        headers["X-Damselfish-Session"] = session_id
    return StreamingResponse(
        stream_chunks(), media_type="text/event-stream", headers=headers
    )


def _identifier(value: Any, name: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, (str, int)):
        raise HTTPException(status_code=400, detail=f"{name} must be a string")
    normalized = str(value).strip()
    if not normalized:
        if optional:
            return None
        raise HTTPException(status_code=400, detail=f"{name} cannot be empty")
    if len(normalized) > 200:
        raise HTTPException(status_code=400, detail=f"{name} is too long")
    return normalized


def _title(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="memory title must be a string")
    normalized = " ".join(value.split())
    return normalized[:200] or None


async def _json_object(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="请求必须是有效 JSON") from error
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="请求必须是 JSON 对象")
    return payload


def _managed_node(
    request: Request, node_id: str, *, optional: bool = False
) -> dict[str, Any] | None:
    node = next(
        (
            item
            for item in request.app.state.node_store.load_raw()
            if item.get("id") == node_id
        ),
        None,
    )
    if node is None and not optional:
        raise HTTPException(status_code=404, detail="可管理节点不存在")
    return node


def _admin_session_token(secret: str, expires_at: int | None = None) -> str:
    expiration = expires_at or int(time.time()) + 12 * 60 * 60
    payload = str(expiration)
    signature = hmac.new(
        secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}.{signature}".encode("ascii")).decode("ascii")


def _valid_admin_session(token: str | None, secret: str) -> bool:
    if not token:
        return False
    try:
        decoded = base64.urlsafe_b64decode(token.encode("ascii")).decode("ascii")
        payload, supplied_signature = decoded.split(".", 1)
        expiration = int(payload)
    except (ValueError, UnicodeDecodeError):
        return False
    if expiration < int(time.time()):
        return False
    expected_signature = hmac.new(
        secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(supplied_signature, expected_signature)


def _request_is_https(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "")
    return request.url.scheme == "https" or forwarded.split(",", 1)[0].strip() == "https"
