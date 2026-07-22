from __future__ import annotations

import re
from typing import Any

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

# Optional tiktoken backend for accurate token counting
_TIKTOKEN_ENCODING = None
try:
    import tiktoken
    _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    pass


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens for a text string.

    When tiktoken is available, uses cl100k_base encoding for accurate counts.
    Falls back to CJK-aware heuristic: CJK ~1.5 tokens/char, other ~0.25,
    with a 10% safety margin.
    """
    if not text:
        return 0
    if _TIKTOKEN_ENCODING is not None:
        return len(_TIKTOKEN_ENCODING.encode(text))
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    estimate = cjk * 1.5 + other * 0.25
    return max(1, int(estimate * 1.1))


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate input tokens from messages.

    When tiktoken is available, uses cl100k_base encoding for accurate
    per-message counts.  Falls back to CJK-aware heuristic.

    Each message adds ~4 tokens of structural overhead (role markers, etc.).
    """
    if _TIKTOKEN_ENCODING is not None:
        return _estimate_messages_tiktoken(messages)
    return _estimate_messages_heuristic(messages)


def _estimate_messages_tiktoken(messages: list[dict[str, Any]]) -> int:
    """Accurate token count using tiktoken."""
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(_TIKTOKEN_ENCODING.encode(content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if not isinstance(text, str):
                        text = part.get("content")
                    if isinstance(text, str):
                        total += len(_TIKTOKEN_ENCODING.encode(text))
        total += 4  # structural overhead per message
    return total


def _estimate_messages_heuristic(messages: list[dict[str, Any]]) -> int:
    """Fallback CJK-aware heuristic token estimate."""
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += _estimate_text_heuristic(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if not isinstance(text, str):
                        text = part.get("content")
                    if isinstance(text, str):
                        total += _estimate_text_heuristic(text)
        total += 4
    return total


def _estimate_text_heuristic(text: str) -> int:
    """CJK-aware heuristic: CJK ~1.5 tokens/char, other ~0.25, +10% margin."""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    estimate = cjk * 1.5 + other * 0.25
    return max(1, int(estimate * 1.1))