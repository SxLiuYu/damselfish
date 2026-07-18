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
