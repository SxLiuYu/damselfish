from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .config import AppConfig, TargetConfig
from .store import TargetStats


@dataclass(frozen=True, slots=True)
class RouteContext:
    scenario: str
    persona: str | None
    required: frozenset[str]
    preferred: frozenset[str]
    preferred_targets: tuple[str, ...]
    estimated_input_tokens: int = 0


SCENARIO_KEYWORDS = {
    "coding": ("代码", "编程", "bug", "debug", "github", "部署", "api", "sql", "python", "javascript"),
    "reasoning": ("分析", "推理", "规划", "比较", "为什么", "方案", "架构", "研究"),
    "creative": ("创作", "文案", "故事", "诗", "营销", "广告"),
    "translation": ("翻译", "translate", "英文", "中文翻译"),
}


def infer_context(
    config: AppConfig,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    scenario: str | None = None,
    persona: str | None = None,
) -> RouteContext:
    text = "\n".join(
        str(message.get("content", ""))
        for message in messages[-6:]
        if isinstance(message.get("content"), str)
    ).lower()
    system_text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "system"
    ).lower()

    selected_persona = persona.lower() if persona else None
    if not selected_persona:
        for name, rule in config.personas.items():
            if any(keyword in system_text for keyword in rule.keywords):
                selected_persona = name
                break

    if scenario:
        selected_scenario = scenario.lower()
    elif tools:
        selected_scenario = "tool"
    elif _contains_image(messages):
        selected_scenario = "vision"
    else:
        selected_scenario = "default"
        for name, keywords in SCENARIO_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                selected_scenario = name
                break

    scenario_rule = config.scenarios.get(
        selected_scenario, config.scenarios.get("default")
    )
    required = set(scenario_rule.required if scenario_rule else ())
    preferred = set(scenario_rule.preferred if scenario_rule else ())
    preferred_targets = list(scenario_rule.targets if scenario_rule else ())
    if selected_persona and selected_persona in config.personas:
        persona_rule = config.personas[selected_persona]
        required.update(persona_rule.required)
        preferred.update(persona_rule.preferred)
        preferred_targets = list(persona_rule.targets) + preferred_targets
    return RouteContext(
        scenario=selected_scenario,
        persona=selected_persona,
        required=frozenset(required),
        preferred=frozenset(preferred),
        preferred_targets=tuple(dict.fromkeys(preferred_targets)),
        estimated_input_tokens=_estimate_messages_tokens(messages),
    )


def rank_targets(
    config: AppConfig,
    context: RouteContext,
    stats: dict[str, TargetStats],
    requested_model: str | None = None,
    max_new_tokens: int = 1024,
) -> list[TargetConfig]:
    now = time.time()
    ranked: list[tuple[float, TargetConfig]] = []
    for target in config.targets:
        state = stats[target.id]
        if not target.available or state.circuit_open_until > now:
            continue
        if not context.required.issubset(target.capabilities):
            continue
        # Filter out targets whose context window is too small for the input
        if target.max_context is not None:
            if context.estimated_input_tokens + max_new_tokens > target.max_context:
                continue
        latency = state.ewma_latency_ms or config.routing.unknown_latency_ms
        attempts = state.successes + state.failures
        failure_rate = state.failures / attempts if attempts else 0.0
        score = latency + failure_rate * config.routing.failure_penalty_ms
        score += target.priority * config.routing.priority_weight_ms
        score -= len(context.preferred & target.capabilities) * 100.0
        if context.scenario in target.scenarios:
            score -= 250.0
        if context.persona and context.persona in target.personas:
            score -= 250.0
        if target.id in context.preferred_targets:
            score -= 500.0 - context.preferred_targets.index(target.id) * 25.0
        if requested_model and requested_model not in {"auto", "damselfish", "damselfish/auto"}:
            if requested_model in {target.id, target.model}:
                score -= 10000.0
        ranked.append((score, target))
    ranked.sort(key=lambda item: (item[0], item[1].id))
    return [target for _, target in ranked]


def _estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate input tokens from messages (characters + overhead heuristic).

    A rough approximation: 1 token ≈ 3~4 bytes for CJK, 4 chars for English.
    We use a blended heuristic: tokens ≈ chars / 2.5 for mixed text.
    Each message adds ~4 tokens of structural overhead (role markers, etc.).
    """
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            # CJK characters: ~1.5 tokens per char; ASCII: ~0.25 tokens per char
            # Blended: len(content) / 2.5 covers most cases
            total += max(1, int(len(content) / 2.5))
        elif isinstance(content, list):
            # Multimodal: estimate text parts only
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += max(1, int(len(part["text"]) / 2.5))
                elif isinstance(part, dict) and isinstance(part.get("content"), str):
                    total += max(1, int(len(part["content"]) / 2.5))
        total += 4  # structural overhead per message
    return total


def _contains_image(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in {"image", "image_url"}
            for part in content
        ):
            return True
    return False
