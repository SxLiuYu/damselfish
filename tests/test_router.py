import asyncio
from pathlib import Path

import httpx

from damselfish.config import AppConfig, RoutingConfig, TargetConfig
from damselfish.router import ModelRouter
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
