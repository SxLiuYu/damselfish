from pathlib import Path

from damselfish.config import AppConfig, RouteRule, RoutingConfig, TargetConfig
from damselfish.selector import RouteContext, infer_context, rank_targets
from damselfish.store import Store


def config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        host="127.0.0.1",
        port=8086,
        database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=0),
        targets=(
            TargetConfig("fast", "Fast", "http://fast/v1", "fast", local=True, capabilities=frozenset({"chat"})),
            TargetConfig("tools", "Tools", "http://tools/v1", "tools", local=True, capabilities=frozenset({"chat", "tools", "coding"})),
        ),
        scenarios={"default": RouteRule(preferred=frozenset({"chat"})), "tool": RouteRule(required=frozenset({"tools"}))},
    )


def test_rank_uses_latency(tmp_path: Path) -> None:
    app_config = config(tmp_path)
    store = Store(app_config.database, ["fast", "tools"])
    store.record_success("fast", 100, 1)
    store.record_success("tools", 500, 1)
    context = RouteContext("default", None, frozenset(), frozenset(), ())
    assert rank_targets(app_config, context, store.all_stats())[0].id == "fast"
    store.close()


def test_tools_require_capable_target(tmp_path: Path) -> None:
    app_config = config(tmp_path)
    store = Store(app_config.database, ["fast", "tools"])
    context = infer_context(app_config, [{"role": "user", "content": "run it"}], [{}])
    assert [target.id for target in rank_targets(app_config, context, store.all_stats())] == ["tools"]
    store.close()
