from __future__ import annotations

from typing import Any

import pytest

from damselfish import tokens
from damselfish.tokens import estimate_messages_tokens, estimate_text_tokens


def _has_tiktoken() -> bool:
    return tokens._TIKTOKEN_ENCODING is not None


def test_text_tokens_empty() -> None:
    """Empty string yields 0."""
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens(None) == 0  # type: ignore[arg-type]


def test_text_tokens_pure_chinese() -> None:
    """Pure Chinese text yields a positive token count."""
    text = "中" * 1000
    estimated = estimate_text_tokens(text)
    if _has_tiktoken():
        # tiktoken: 1 CJK char ≈ 1 token
        assert estimated == 1000, f"tiktoken expected 1000, got {estimated}"
    else:
        # heuristic: 1000 × 1.5 × 1.1 = 1650
        assert estimated == 1650, f"heuristic expected 1650, got {estimated}"


def test_text_tokens_pure_english() -> None:
    """Pure English text yields a positive token count."""
    text = "hello world " * 100  # 1200 chars
    estimated = estimate_text_tokens(text)
    if _has_tiktoken():
        # tiktoken: ~201 tokens for this text
        assert estimated == 201, f"tiktoken expected 201, got {estimated}"
    else:
        # heuristic: 1200 × 0.25 × 1.1 = 330
        assert estimated == 330, f"heuristic expected 330, got {estimated}"


def test_text_tokens_mixed() -> None:
    """Chinese + English mixed text yields a positive token count."""
    text = "中" * 500 + "hello " * 100  # 500 CJK + 600 ASCII = 1100 chars
    estimated = estimate_text_tokens(text)
    if _has_tiktoken():
        # tiktoken: 500 CJK + ~100 ASCII words
        assert estimated == 601, f"tiktoken expected 601, got {estimated}"
    else:
        # heuristic: 500×1.5 + 600×0.25 = 900; ×1.1 = 990
        assert estimated == 990, f"heuristic expected 990, got {estimated}"


def test_messages_tokens_multimodal() -> None:
    """Multimodal messages: only text parts are counted."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "图文并茂的描述"},
        {"role": "user", "content": [
            {"type": "text", "text": "图片说明"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]},
    ]
    estimated = estimate_messages_tokens(messages)
    if _has_tiktoken():
        # 7 CJK + 2 CJK + 2*4 overhead = 9 + 8 = 17
        assert estimated == 17, f"tiktoken got {estimated}"
    else:
        # heuristic: 11 + 6 + 8 = 25
        assert estimated == 25, f"heuristic got {estimated}"


def test_messages_tokens_empty() -> None:
    """Empty message list yields 0."""
    assert estimate_messages_tokens([]) == 0


def test_messages_tokens_content_list_with_text_key() -> None:
    """Handles content as list with 'text' key."""
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "中"},
            ],
        },
    ]
    estimated = estimate_messages_tokens(messages)
    if _has_tiktoken():
        # "hello"=1 + "中"=1 + 4 overhead = 6
        assert estimated == 6, f"tiktoken got {estimated}"
    else:
        # heuristic: 1 + 1 + 4 = 6
        assert estimated == 6, f"heuristic got {estimated}"


def test_text_tokens_japanese() -> None:
    """Japanese kana characters are counted."""
    text = "こんにちは" * 200  # 1000 chars
    estimated = estimate_text_tokens(text)
    if _has_tiktoken():
        assert estimated == 200, f"tiktoken expected 200, got {estimated}"
    else:
        assert estimated == 1650, f"heuristic expected 1650, got {estimated}"


def test_text_tokens_korean() -> None:
    """Korean hangul characters are counted."""
    text = "안녕하세요" * 200  # 1000 chars
    estimated = estimate_text_tokens(text)
    if _has_tiktoken():
        assert estimated == 1000, f"tiktoken expected 1000, got {estimated}"
    else:
        assert estimated == 1650, f"heuristic expected 1650, got {estimated}"


def test_tiktoken_and_heuristic_both_positive() -> None:
    """Both backends produce positive counts for non-empty text."""
    text = "Hello 世界! This is a test."
    assert estimate_text_tokens(text) > 0
    # Also test the heuristic directly
    assert tokens._estimate_text_heuristic(text) > 0


def test_heuristic_overestimates_cjk() -> None:
    """The heuristic backend overestimates CJK (safety margin).

    This test only runs when tiktoken is NOT available, since when tiktoken
    is available the heuristic is not used.  When tiktoken IS available, we
    verify the heuristic still overestimates relative to tiktoken.
    """
    text = "中" * 1000
    tiktoken_count = len(tokens._TIKTOKEN_ENCODING.encode(text)) if _has_tiktoken() else 1000
    heuristic_count = tokens._estimate_text_heuristic(text)
    # Heuristic should be >= real count (conservative)
    assert heuristic_count >= tiktoken_count, (
        f"heuristic {heuristic_count} should be >= real {tiktoken_count}"
    )