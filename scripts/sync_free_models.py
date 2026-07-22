#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


SYNC_TAG = "free-model-sync"
DEFAULT_NODES_FILE = Path("/var/lib/damselfish/managed-nodes.json")
DEFAULT_BACKUP_DIR = Path("/var/lib/damselfish/backups/free-model-sync")
DEFAULT_FIXTURE_DIR = None
KILO_MODELS_URL = "https://api.kilo.ai/api/gateway/v1/models"
POLLINATIONS_MODELS_URL = "https://text.pollinations.ai/models"
ZHIPU_MODELS_URL = "https://docs.bigmodel.cn/cn/guide/start/model-overview"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
KILO_BASE_URL = "https://api.kilo.ai/api/gateway/v1"
POLLINATIONS_BASE_URL = "https://text.pollinations.ai/openai/v1"

EXCLUDED_KILO_IDS = {
    "nvidia/nemotron-3.5-content-safety:free",
    "google/lyria-3-pro-preview",
    "google/lyria-3-clip-preview",
}
EXCLUDED_ZHIPU_SLUGS = {"cogview-3-flash", "cogvideox-flash"}
ZHIPU_FREE_LINK_RE = re.compile(r"/cn/guide/models/free/([a-zA-Z0-9._-]+)")

# Known context limits for Zhipu free models (tokens).
# When a model is not listed here, no limit is enforced (assumed large).
ZHIPU_CONTEXT_LIMITS: dict[str, int] = {
    "glm-4v-flash": 16384,
    "glm-4v": 16384,
}

# Known context limits for Kilo free models (tokens).
# Kilo API may include context_length per model; this dict provides fallbacks
# for models where the catalog omits the field.
KILO_CONTEXT_LIMITS: dict[str, int] = {
    "google/gemma-3-1b-it:free": 8192,
    "google/gemma-3-4b-it:free": 8192,
    "meta-llama/llama-3.1-8b-instruct:free": 8192,
    "meta-llama/llama-3.2-3b-instruct:free": 8192,
    "qwen/qwen-2.5-7b-instruct:free": 8192,
    "qwen/qwen-2.5-coder-7b-instruct:free": 8192,
}

# Known context limits for Pollinations free models (tokens).
# Pollinations anonymous tier typically uses 8K context.
POLLINATIONS_CONTEXT_LIMITS: dict[str, int] = {
    "openai": 8192,
    "mistral": 8192,
    "llama": 8192,
}


class DiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class Discovery:
    provider: str
    models: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile verified free Damselfish models")
    parser.add_argument("--nodes-file", type=Path, default=DEFAULT_NODES_FILE)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--restart-service", action="store_true")
    return parser.parse_args()


def fetch_json(client: httpx.Client, url: str, fixture: Path | None) -> Any:
    if fixture:
        return json.loads(fixture.read_text())
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def fetch_text(client: httpx.Client, url: str, fixture: Path | None) -> str:
    if fixture:
        return fixture.read_text()
    response = client.get(url)
    response.raise_for_status()
    return response.text


def fixture_path(fixture_dir: Path | None, name: str) -> Path | None:
    return fixture_dir / name if fixture_dir else None


def is_zero(value: Any) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def is_chat_compatible(model: dict[str, Any]) -> bool:
    architecture = model.get("architecture") or {}
    output_modalities = architecture.get("output_modalities") or []
    return not output_modalities or "text" in output_modalities


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:100]


def _kilo_context(entry: dict[str, Any], model_id: str) -> int | None:
    """Extract context length from Kilo catalog entry, falling back to known limits."""
    raw = entry.get("context_length")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return KILO_CONTEXT_LIMITS.get(model_id)


def _pollinations_context(entry: dict[str, Any], model_name: str) -> int | None:
    """Extract context length from Pollinations catalog entry, falling back to known limits."""
    raw = entry.get("maxLength")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    # Fall back by matching model name prefix
    for prefix, limit in POLLINATIONS_CONTEXT_LIMITS.items():
        if model_name.lower().startswith(prefix):
            return limit
    return None


def common_node(
    *,
    node_id: str,
    label: str,
    base_url: str,
    model: str,
    provider: str,
    priority: int,
    api_key: str = "",
    capabilities: list[str] | None = None,
    scenarios: list[str] | None = None,
    max_context: int | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "label": label,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "enabled": True,
        "local": False,
        "free": True,
        "priority": priority,
        "capabilities": capabilities or ["chat", "multilingual", "coding", "reasoning", "tools", "fast"],
        "scenarios": scenarios or ["default", "tool", "coding", "reasoning"],
        "personas": ["developer", "operator"],
        "probe": False,
        "probe_prompt": "OK",
        "max_concurrency": 1,
        "max_context": max_context,
        "auto_managed": SYNC_TAG,
        "provider": provider,
        "source": "official-catalog",
    }


def discover_kilo(client: httpx.Client, fixture_dir: Path | None) -> Discovery:
    payload = fetch_json(client, KILO_MODELS_URL, fixture_path(fixture_dir, "kilo.json"))
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise DiscoveryError("Kilo catalog has no data list")
    free_models = []
    for entry in entries:
        model_id = entry.get("id")
        pricing = entry.get("pricing") or {}
        if not isinstance(model_id, str) or model_id in EXCLUDED_KILO_IDS:
            continue
        if not is_zero(pricing.get("prompt")) or not is_zero(pricing.get("completion")):
            continue
        if not is_chat_compatible(entry):
            continue
        name = entry.get("name") or model_id
        capabilities = ["chat", "multilingual", "coding", "fast"]
        parameters = set(entry.get("supported_parameters") or [])
        if "tools" in parameters:
            capabilities.append("tools")
        if "reasoning" in parameters or "include_reasoning" in parameters:
            capabilities.append("reasoning")
        free_models.append(
            common_node(
                node_id=f"kilo-{slugify(model_id.removesuffix(':free'))}",
                label=f"Kilo {name}",
                base_url=KILO_BASE_URL,
                model=model_id,
                provider="kilo",
                priority=30 + len(free_models),
                capabilities=capabilities,
                max_context=_kilo_context(entry, model_id),
            )
        )
    if not entries:
        raise DiscoveryError("Kilo catalog returned an empty data list")
    return Discovery("kilo", free_models)


def discover_pollinations(client: httpx.Client, fixture_dir: Path | None) -> Discovery:
    payload = fetch_json(client, POLLINATIONS_MODELS_URL, fixture_path(fixture_dir, "pollinations.json"))
    if not isinstance(payload, list):
        raise DiscoveryError("Pollinations catalog is not a list")
    free_models = []
    for entry in payload:
        name = entry.get("name")
        if not isinstance(name, str) or entry.get("tier") != "anonymous":
            continue
        capabilities = ["chat", "multilingual", "coding", "fast"]
        if entry.get("tools"):
            capabilities.append("tools")
        if entry.get("reasoning"):
            capabilities.append("reasoning")
        label = entry.get("description") or name
        free_models.append(
            common_node(
                node_id=f"pollinations-{slugify(name)}",
                label=f"Pollinations {label}",
                base_url=POLLINATIONS_BASE_URL,
                model=name,
                provider="pollinations",
                priority=50 + len(free_models),
                capabilities=capabilities,
                scenarios=["default", "tool", "coding", "reasoning", "translation"],
                max_context=_pollinations_context(entry, name),
            )
        )
    if not payload:
        raise DiscoveryError("Pollinations catalog returned an empty list")
    return Discovery("pollinations", free_models)


def find_api_key(nodes: list[dict[str, Any]], base_url: str) -> str:
    for node in nodes:
        if node.get("base_url") == base_url and node.get("api_key"):
            return str(node["api_key"])
    return ""


def discover_zhipu(client: httpx.Client, fixture_dir: Path | None, nodes: list[dict[str, Any]]) -> Discovery:
    page = fetch_text(client, ZHIPU_MODELS_URL, fixture_path(fixture_dir, "zhipu.html"))
    free_slugs = set(ZHIPU_FREE_LINK_RE.findall(page))
    if not free_slugs:
        raise DiscoveryError("Zhipu documentation returned no free-model links")
    slugs = sorted(free_slugs - EXCLUDED_ZHIPU_SLUGS)
    slugs = [slug for slug in slugs if slug.startswith("glm-")]
    api_key = find_api_key(nodes, ZHIPU_BASE_URL)
    if slugs and not api_key:
        raise DiscoveryError("Zhipu free models found but no existing API key is available")
    models = []
    for slug in slugs:
        models.append(
            common_node(
                node_id=f"zhipu-{slug}",
                label=f"Zhipu {slug.upper()}",
                base_url=ZHIPU_BASE_URL,
                model=slug,
                provider="zhipu",
                priority=10 + len(models),
                api_key=api_key,
                capabilities=["chat", "chinese", "multilingual", "coding", "reasoning", "tools", "fast"],
                scenarios=["default", "tool", "coding", "reasoning", "translation"],
                max_context=ZHIPU_CONTEXT_LIMITS.get(slug),
            )
        )
    return Discovery("zhipu", models)


def merge_existing_fields(desired: dict[str, Any], existing: dict[str, Any] | None) -> dict[str, Any]:
    if not existing:
        return desired
    result = dict(desired)
    for field in ("id", "api_key", "max_concurrency"):
        if existing.get(field) not in (None, ""):
            result[field] = existing[field]
    return result


def reconcile(
    nodes: list[dict[str, Any]], discoveries: list[Discovery], failed_providers: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    desired_by_provider = {item.provider: item.models for item in discoveries}
    existing_by_key = {(node.get("provider"), node.get("model")): node for node in nodes}
    existing_by_location = {(node.get("base_url"), node.get("model")): node for node in nodes}
    managed_providers = set(desired_by_provider) | failed_providers
    preserved = []
    removed = []
    migrated_locations = {
        (model["base_url"], model["model"])
        for models in desired_by_provider.values()
        for model in models
    }
    for node in nodes:
        provider = node.get("provider")
        if node.get("auto_managed") == SYNC_TAG and provider in managed_providers:
            if provider in failed_providers:
                preserved.append(node)
            else:
                removed.append(node["id"])
            continue
        if (node.get("base_url"), node.get("model")) in migrated_locations:
            continue
        preserved.append(node)

    desired_nodes = []
    for discovery in discoveries:
        for desired in discovery.models:
            existing = existing_by_key.get((discovery.provider, desired["model"]))
            if not existing:
                existing = existing_by_location.get((desired["base_url"], desired["model"]))
            desired_nodes.append(merge_existing_fields(desired, existing))

    result = preserved + desired_nodes
    ids = [node["id"] for node in result]
    if len(ids) != len(set(ids)):
        raise RuntimeError("reconciliation produced duplicate node IDs")
    old_by_id = {node["id"]: node for node in nodes}
    new_by_id = {node["id"]: node for node in result}
    summary = {
        "changed": nodes != result,
        "added": sorted(set(new_by_id) - set(old_by_id)),
        "removed": sorted(set(old_by_id) - set(new_by_id)),
        "updated": sorted(
            node_id for node_id in set(old_by_id) & set(new_by_id) if old_by_id[node_id] != new_by_id[node_id]
        ),
        "total": len(result),
    }
    return result, summary


def backup_file(nodes_file: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = backup_dir / f"managed-nodes.{stamp}.json"
    shutil.copy2(nodes_file, destination)
    destination.chmod(0o600)
    backups = sorted(backup_dir.glob("managed-nodes.*.json"), reverse=True)
    for stale in backups[30:]:
        stale.unlink()
    return destination


def atomic_write(nodes_file: Path, payload: dict[str, Any], original_stat: os.stat_result) -> None:
    nodes_file.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{nodes_file.name}.", dir=nodes_file.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IMODE(original_stat.st_mode))
        os.chown(temporary, original_stat.st_uid, original_stat.st_gid)
        os.replace(temporary, nodes_file)
        directory_fd = os.open(nodes_file.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    lock_file = args.nodes_file.with_suffix(args.nodes_file.suffix + ".free-model-sync.lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        original_stat = args.nodes_file.stat()
        payload = json.loads(args.nodes_file.read_text())
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            raise RuntimeError("managed nodes file has no nodes list")

        discoveries = []
        errors = {}
        timeout = httpx.Timeout(30.0, connect=10.0)
        headers = {"User-Agent": "damselfish-free-model-sync/1.0"}
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            providers = (
                ("kilo", lambda: discover_kilo(client, args.fixture_dir)),
                ("pollinations", lambda: discover_pollinations(client, args.fixture_dir)),
                ("zhipu", lambda: discover_zhipu(client, args.fixture_dir, nodes)),
            )
            for provider, discover in providers:
                try:
                    discoveries.append(discover())
                except Exception as exc:
                    errors[provider] = f"{type(exc).__name__}: {exc}"

        result, summary = reconcile(nodes, discoveries, set(errors))
        summary["dry_run"] = args.dry_run
        summary["discovered"] = {item.provider: len(item.models) for item in discoveries}
        summary["provider_errors"] = errors
        summary["backup"] = None
        summary["restarted"] = False
        if summary["changed"] and not args.dry_run:
            if not args.no_backup:
                summary["backup"] = str(backup_file(args.nodes_file, args.backup_dir))
            atomic_write(args.nodes_file, {**payload, "nodes": result}, original_stat)
            if args.restart_service:
                subprocess.run(
                    ["/usr/bin/systemctl", "try-restart", "damselfish.service"],
                    check=True,
                    timeout=60,
                )
                summary["restarted"] = True
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if discoveries else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"fatal": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
