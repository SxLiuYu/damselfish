from pathlib import Path
from typing import Any

from damselfish.config import AppConfig, RouteRule, RoutingConfig, TargetConfig
from damselfish.selector import (
    RouteContext,
    _estimate_messages_tokens,
    _estimate_text_tokens,
    infer_context,
    rank_targets,
)
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


# ── Token estimation tests ───────────────────────────────────────────


def test_estimate_text_tokens_pure_chinese() -> None:
    """Pure Chinese: ~1.5 tokens/char × 1.1 safety margin."""
    from damselfish import tokens
    text = "中" * 1000
    estimated = _estimate_text_tokens(text)
    if tokens._TIKTOKEN_ENCODING is not None:
        assert estimated == 1000, f"tiktoken expected 1000, got {estimated}"
    else:
        assert estimated == 1650, f"expected 1650, got {estimated}"


def test_estimate_text_tokens_pure_english() -> None:
    """Pure English: ~0.25 tokens/char × 1.1 safety margin."""
    from damselfish import tokens
    text = "hello world " * 100  # 1200 chars
    estimated = _estimate_text_tokens(text)
    if tokens._TIKTOKEN_ENCODING is not None:
        assert estimated == 201, f"tiktoken expected 201, got {estimated}"
    else:
        assert estimated == 330, f"expected 330, got {estimated}"


def test_estimate_text_tokens_mixed() -> None:
    """Chinese + English: weighted sum."""
    from damselfish import tokens
    text = "中" * 500 + "hello " * 100  # 500 CJK + 600 ASCII = 1100 chars
    estimated = _estimate_text_tokens(text)
    if tokens._TIKTOKEN_ENCODING is not None:
        assert estimated == 601, f"tiktoken expected 601, got {estimated}"
    else:
        assert estimated == 990, f"expected 990, got {estimated}"


def test_estimate_text_tokens_empty() -> None:
    """Empty string yields 0."""
    assert _estimate_text_tokens("") == 0
    assert _estimate_text_tokens(None) == 0  # type: ignore[arg-type]


def test_estimate_messages_tokens_multimodal() -> None:
    """Multimodal messages: only text parts are counted."""
    from damselfish import tokens
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "图文并茂的描述"},
        {"role": "user", "content": [
            {"type": "text", "text": "图片说明"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]},
    ]
    estimated = _estimate_messages_tokens(messages)
    if tokens._TIKTOKEN_ENCODING is not None:
        # 7 + 2 + 2*4 overhead = 17
        assert estimated == 17, f"tiktoken got {estimated}"
    else:
        # heuristic: 7*1.5*1.1=11 + 4*1.5*1.1=6 + 8 = 25
        assert estimated == 25, f"heuristic got {estimated}"


def test_estimate_messages_tokens_empty() -> None:
    """Empty message list yields 0."""
    assert _estimate_messages_tokens([]) == 0


def test_estimate_messages_tokens_overestimates_safely() -> None:
    """Estimate should be >= actual token count for pure Chinese.

    For pure Chinese, 1 token ≈ 1.5 chars, so our estimate (1.5×1.1=1.65 chars/token)
    should be conservative (slightly overestimate).
    """
    from damselfish import tokens
    text = "中" * 1000
    estimated = _estimate_text_tokens(text)
    if tokens._TIKTOKEN_ENCODING is not None:
        # tiktoken: 1000 tokens. Heuristic should be >= that.
        heuristic = tokens._estimate_text_heuristic(text)
        assert heuristic > estimated, "heuristic should be >= tiktoken"
    else:
        # Real token count for 1000 CJK chars ≈ 1500. Our estimate = 1650 (10% overhead)
        assert estimated > 1500, "estimate should be conservative for CJK"


# ── max_context filtering tests ──────────────────────────────────────


def test_rank_targets_filters_by_max_context(tmp_path: Path) -> None:
    """Targets with insufficient max_context are filtered out."""
    app_config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=0),
        targets=(
            TargetConfig("short", "Short", "http://s/v1", "s", local=True,
                         priority=1, max_context=4096),
            TargetConfig("long", "Long", "http://l/v1", "l", local=True,
                         priority=2, max_context=128000),
        ),
    )
    store = Store(app_config.database, ["short", "long"])
    # ~4000 chars of Chinese → ~6600 tokens, exceeds short's 4096
    long_messages = [{"role": "user", "content": "中" * 4000}]
    context = infer_context(app_config, long_messages)
    ranked = rank_targets(app_config, context, store.all_stats())
    assert [t.id for t in ranked] == ["long"], f"got {[t.id for t in ranked]}"
    store.close()


def test_rank_targets_passes_within_context(tmp_path: Path) -> None:
    """Targets with sufficient max_context are kept."""
    app_config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=0),
        targets=(
            TargetConfig("short", "Short", "http://s/v1", "s", local=True,
                         priority=1, max_context=4096),
        ),
    )
    store = Store(app_config.database, ["short"])
    # Short message: ~100 tokens, well within 4096
    short_messages = [{"role": "user", "content": "hello"}]
    context = infer_context(app_config, short_messages)
    ranked = rank_targets(app_config, context, store.all_stats())
    assert [t.id for t in ranked] == ["short"]
    store.close()


def test_rank_targets_no_max_context(tmp_path: Path) -> None:
    """Targets without max_context are never filtered."""
    app_config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=0),
        targets=(
            TargetConfig("unlimited", "Unlimited", "http://u/v1", "u", local=True,
                         priority=1, max_context=None),
        ),
    )
    store = Store(app_config.database, ["unlimited"])
    huge_messages = [{"role": "user", "content": "中" * 50000}]
    context = infer_context(app_config, huge_messages)
    ranked = rank_targets(app_config, context, store.all_stats())
    assert [t.id for t in ranked] == ["unlimited"]
    store.close()


def test_rank_targets_max_new_tokens_parameter(tmp_path: Path) -> None:
    """Large max_new_tokens can cause filtering even with moderate input."""
    app_config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=0),
        targets=(
            TargetConfig("small", "Small", "http://s/v1", "s", local=True,
                         priority=1, max_context=2048),
            TargetConfig("big", "Big", "http://b/v1", "b", local=True,
                         priority=2, max_context=128000),
        ),
    )
    store = Store(app_config.database, ["small", "big"])
    # Input ~1000 tokens + max_new_tokens=2000 > 2048 → small filtered
    messages = [{"role": "user", "content": "中" * 600}]
    context = infer_context(app_config, messages)
    ranked = rank_targets(app_config, context, store.all_stats(), max_new_tokens=2000)
    assert [t.id for t in ranked] == ["big"], f"got {[t.id for t in ranked]}"
    # With max_new_tokens=500, small should pass
    ranked2 = rank_targets(app_config, context, store.all_stats(), max_new_tokens=500)
    assert {t.id for t in ranked2} == {"small", "big"}
    store.close()


def test_rank_dynamic_penalty_for_high_failure_rate(tmp_path: Path) -> None:
    """Targets with >50% failure rate get an extra penalty."""
    app_config = AppConfig(
        host="127.0.0.1", port=8086, database=tmp_path / "test.db",
        routing=RoutingConfig(priority_weight_ms=0, failure_penalty_ms=2000),
        targets=(
            TargetConfig("reliable", "Reliable", "http://r/v1", "r", local=True,
                         priority=1, capabilities=frozenset({"chat"})),
            TargetConfig("flaky", "Flaky", "http://f/v1", "f", local=True,
                         priority=2, capabilities=frozenset({"chat"})),
        ),
    )
    store = Store(app_config.database, ["reliable", "flaky"])
    # Reliable: 100 successes, 0 failures
    for _ in range(100):
        store.record_success("reliable", 100, 0.5)
    # Flaky: 10 successes, 90 failures (90% failure rate)
    for _ in range(10):
        store.record_success("flaky", 100, 0.5)
    for _ in range(90):
        store.record_failure("flaky", 500, "error", 0)

    context = RouteContext("default", None, frozenset({"chat"}), frozenset(), ())
    ranked = rank_targets(app_config, context, store.all_stats())
    # Reliable should be ranked first despite equal latency
    assert ranked[0].id == "reliable"
    store.close()
