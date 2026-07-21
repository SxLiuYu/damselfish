from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class TargetConfig:
    id: str
    label: str
    base_url: str
    model: str
    api_key_env: str | None = None
    enabled: bool = True
    local: bool = False
    free: bool = True
    priority: int = 100
    capabilities: frozenset[str] = frozenset({"chat"})
    scenarios: frozenset[str] = frozenset()
    personas: frozenset[str] = frozenset()
    probe: bool = True
    probe_prompt: str = "Reply OK"
    max_concurrency: int = 4
    api_key_value: str = field(default="", repr=False, compare=False)

    @property
    def chat_url(self) -> str:
        url = self.base_url.rstrip("/")
        return url if url.endswith("/chat/completions") else f"{url}/chat/completions"

    @property
    def api_key(self) -> str:
        if self.api_key_value:
            return self.api_key_value
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""

    @property
    def available(self) -> bool:
        return self.enabled and (self.local or not self.api_key_env or bool(self.api_key))


@dataclass(frozen=True, slots=True)
class RouteRule:
    required: frozenset[str] = frozenset()
    preferred: frozenset[str] = frozenset()
    targets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PersonaRule:
    keywords: tuple[str, ...] = ()
    required: frozenset[str] = frozenset()
    preferred: frozenset[str] = frozenset()
    targets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    request_timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 10.0
    probe_interval_seconds: float = 180.0
    probe_stale_seconds: float = 120.0
    ewma_alpha: float = 0.35
    unknown_latency_ms: float = 1500.0
    failure_penalty_ms: float = 2000.0
    priority_weight_ms: float = 10.0
    circuit_failures: int = 3
    circuit_base_seconds: float = 15.0
    circuit_max_seconds: float = 300.0
    memory_max_messages: int = 40
    memory_ttl_days: int = 30
    project_memory_session_limit: int = 3
    project_memory_message_limit: int = 6
    project_memory_max_chars: int = 12000
    memory_compression_threshold: int = 30
    memory_compression_keep: int = 10
    parallel_fallback_count: int = 3
    parallel_fallback_timeout_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class GitSyncConfig:
    enabled: bool = False
    repository: Path = Path("./data/memory-repo")
    remote_url_env: str = "DAMSELFISH_MEMORY_GIT_URL"
    branch: str = "main"
    pull_interval_seconds: float = 30.0
    push_retries: int = 3
    push_on_write: bool = True
    author_name: str = "Damselfish Memory"
    author_email: str = "damselfish@localhost"
    device_id_env: str = "DAMSELFISH_DEVICE_ID"

    @property
    def remote_url(self) -> str:
        return os.environ.get(self.remote_url_env, "")

    @property
    def device_id(self) -> str:
        return os.environ.get(self.device_id_env, "")


@dataclass(frozen=True, slots=True)
class AppConfig:
    host: str
    port: int
    database: Path
    routing: RoutingConfig
    targets: tuple[TargetConfig, ...]
    git_sync: GitSyncConfig = GitSyncConfig()
    scenarios: dict[str, RouteRule] = field(default_factory=dict)
    personas: dict[str, PersonaRule] = field(default_factory=dict)
    managed_nodes_file: Path | None = None
    managed_target_ids: frozenset[str] = frozenset()


def _set(value: Any, default: tuple[str, ...] = ()) -> frozenset[str]:
    return frozenset(str(item).lower() for item in (value or default))


def _tuple(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in (value or ()))


def _route_rule(raw: dict[str, Any]) -> RouteRule:
    return RouteRule(
        required=_set(raw.get("required")),
        preferred=_set(raw.get("preferred")),
        targets=_tuple(raw.get("targets")),
    )


def target_from_mapping(item: dict[str, Any], *, managed: bool = False) -> TargetConfig:
    return TargetConfig(
        id=str(item["id"]),
        label=str(item.get("label", item["id"])),
        base_url=str(item["base_url"]),
        model=str(item["model"]),
        api_key_env=item.get("api_key_env"),
        enabled=bool(item.get("enabled", True)),
        local=bool(item.get("local", False)),
        free=bool(item.get("free", True)),
        priority=int(item.get("priority", 100)),
        capabilities=_set(item.get("capabilities"), ("chat",)),
        scenarios=_set(item.get("scenarios")),
        personas=_set(item.get("personas")),
        probe=bool(item.get("probe", True)),
        probe_prompt=str(item.get("probe_prompt", "Reply OK")),
        max_concurrency=max(int(item.get("max_concurrency", 4)), 1),
        api_key_value=str(item.get("api_key", "")) if managed else "",
    )


def _load_managed_targets(path: Path) -> tuple[TargetConfig, ...]:
    if not path.exists():
        return ()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    nodes = raw.get("nodes", []) if isinstance(raw, dict) else []
    if not isinstance(nodes, list):
        raise ValueError("managed nodes file must contain a nodes list")
    return tuple(target_from_mapping(item, managed=True) for item in nodes)


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(
        path or os.environ.get("DAMSELFISH_CONFIG", "config.yml")
    ).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    routing_raw = raw.get("routing", {})
    routing = RoutingConfig(
        **{
            key: value
            for key, value in routing_raw.items()
            if key in RoutingConfig.__dataclass_fields__
        }
    )
    base_targets = tuple(target_from_mapping(item) for item in raw.get("targets", []))
    database = Path(raw.get("database", "./data/damselfish.db")).expanduser()
    if not database.is_absolute():
        database = (config_path.parent / database).resolve()
    managed_nodes_raw = raw.get("managed_nodes_file")
    managed_nodes_file = Path(
        managed_nodes_raw or database.parent / "managed-nodes.json"
    ).expanduser()
    if not managed_nodes_file.is_absolute():
        managed_nodes_file = (config_path.parent / managed_nodes_file).resolve()
    managed_targets = _load_managed_targets(managed_nodes_file)
    targets = base_targets + managed_targets
    if not targets:
        raise ValueError("config must define at least one target")
    ids = [target.id for target in targets]
    if len(ids) != len(set(ids)):
        raise ValueError("target ids must be unique")

    scenarios = {
        str(name).lower(): _route_rule(value or {})
        for name, value in raw.get("scenarios", {}).items()
    }
    personas = {
        str(name).lower(): PersonaRule(
            keywords=tuple(
                str(keyword).lower() for keyword in (value or {}).get("keywords", [])
            ),
            required=_set((value or {}).get("required")),
            preferred=_set((value or {}).get("preferred")),
            targets=_tuple((value or {}).get("targets")),
        )
        for name, value in raw.get("personas", {}).items()
    }
    sync_raw = raw.get("git_sync", {}) or {}
    repository = Path(sync_raw.get("repository", "./data/memory-repo")).expanduser()
    if not repository.is_absolute():
        repository = (config_path.parent / repository).resolve()
    git_sync = GitSyncConfig(
        enabled=bool(sync_raw.get("enabled", False)),
        repository=repository,
        remote_url_env=str(
            sync_raw.get("remote_url_env", "DAMSELFISH_MEMORY_GIT_URL")
        ),
        branch=str(sync_raw.get("branch", "main")),
        pull_interval_seconds=float(sync_raw.get("pull_interval_seconds", 30)),
        push_retries=max(int(sync_raw.get("push_retries", 3)), 1),
        push_on_write=bool(sync_raw.get("push_on_write", True)),
        author_name=str(sync_raw.get("author_name", "Damselfish Memory")),
        author_email=str(sync_raw.get("author_email", "damselfish@localhost")),
        device_id_env=str(sync_raw.get("device_id_env", "DAMSELFISH_DEVICE_ID")),
    )
    server = raw.get("server", {})
    return AppConfig(
        host=str(server.get("host", "127.0.0.1")),
        port=int(server.get("port", 8086)),
        database=database,
        routing=routing,
        targets=targets,
        git_sync=git_sync,
        scenarios=scenarios,
        personas=personas,
        managed_nodes_file=managed_nodes_file,
        managed_target_ids=frozenset(target.id for target in managed_targets),
    )
