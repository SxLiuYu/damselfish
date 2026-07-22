from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

from .config import AppConfig, TargetConfig
from .store import TargetStats
from .tokens import estimate_messages_tokens, estimate_text_tokens


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
    # Only join the last 6 messages and only string content for efficiency
    text_parts: list[str] = []
    for message in messages[-6:]:
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content)
    text = "".join(text_parts).lower()
    system_parts: list[str] = []
    for message in messages:
        if message.get("role") == "system" and isinstance(message.get("content"), str):
            system_parts.append(message["content"])
    system_text = "".join(system_parts).lower()

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
        estimated_input_tokens=estimate_messages_tokens(messages),
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
        # Dynamic penalty: when failure rate > 50%, add exponential penalty
        # so chronically failing targets sink to the bottom of the ranking.
        if failure_rate > 0.5 and attempts >= 5:
            score += 10000.0 * (failure_rate - 0.5) * 2
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
        # Load balancing: if the top target has served the vast majority of
        # recent requests, add a small random penalty to avoid pinning all
        # traffic to a single target.  This only applies when the target has
        # enough history to be statistically meaningful.
        if state.successes > 100 and state.failure_rate < 0.1:
            # Target is doing well — still give it a small chance of being
            # skipped so other targets get exercised.
            score += random.uniform(0, 50.0) if state.successes > 500 else 0.0
        ranked.append((score, target))
    ranked.sort(key=lambda item: (item[0], item[1].id))
    return [target for _, target in ranked]


# Re-export for backward compatibility (tests import these directly)
_estimate_messages_tokens = estimate_messages_tokens
_estimate_text_tokens = estimate_text_tokens


def _contains_image(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in {"image", "image_url"}
            for part in content
        ):
            return True
    return False
