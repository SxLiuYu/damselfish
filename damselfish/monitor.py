# -*- coding: utf-8 -*-
"""Disk monitoring and auto-cleanup for the Damselfish service.

Provides two capabilities:
  1. Monitoring   – disk usage snapshots (via HTTP or CLI).
  2. Control      – policy-based automatic cleanup when usage crosses thresholds.

CLI usage:
    python -m damselfish.monitor          # Run once, print report
    python -m damselfish.monitor --serve  # Start HTTP server on :9876
    python -m damselfish.monitor --clean  # Force a cleanup run
    python -m damselfish.monitor --json   # Output raw JSON

HTTP endpoints (when running as service):
    GET  /monitor/disk     – Current disk snapshot
    POST /monitor/cleanup  – Trigger cleanup with optional policy
    GET  /health           – Health check
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Cleanup regex: matches common temp/old files
_CLEANUP_RE = re.compile(r".*\.(bak|tmp|old|orig|~|log\.gz|backup)$", re.I)


def get_disk_usage(path: str = "/") -> dict[str, Any]:
    """Return structured disk-usage data for *path* (cross-platform)."""
    try:
        usage = shutil.disk_usage(path)
        total = usage.total
        free  = usage.free
        used  = usage.used
        avail = free  # on Unix avail differs from free; on Windows it matches
        pct   = round(used / total * 100, 2) if total else 0
        result = {
            "path":         path,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "total_gb":     round(total / (1024**3), 2),
            "used_gb":      round(used / (1024**3), 2),
            "free_gb":      round(free / (1024**3), 2),
            "available_gb": round(avail / (1024**3), 2),
            "percent_used": pct,
        }
        # Only include inode info on platforms that support it
        try:
            st = os.statvfs(path)
            result["inodes_total"] = st.f_files
            result["inodes_free"]  = st.f_ffree
            result["inodes_used"]  = st.f_files - st.f_ffree
            result["inodes_percent"] = round(
                (st.f_files - st.f_ffree) / st.f_files * 100, 2
            ) if st.f_files > 0 else 0
        except (OSError, AttributeError):
            pass  # Windows: no statvfs
        return result
    except OSError as e:
        log.error("Disk stat failed for %s: %s", path, e)
        return {"path": path, "error": str(e)}


def get_disk_report(paths: list[str] | None = None) -> dict[str, Any]:
    """Build a complete disk snapshot for all monitored paths."""
    if paths is None:
        paths = ["/"]
    disks: dict[str, Any] = {}
    warnings: list[str] = []
    critical = False
    for p in paths:
        if not p.startswith("/"):
            p = "/" + p
        info = get_disk_usage(p)
        disks[p] = info
        if "percent_used" in info:
            pct = info["percent_used"]
            if pct >= 95:
                critical = True
                msg = f"CRITICAL: {p} is at {pct}% ({info['used_gb']} GB used)"
                warnings.append(msg)
            elif pct >= 85:
                warnings.append(f"WARN: {p} is at {pct}% ({info['used_gb']} GB used)")
            elif pct >= 70:
                warnings.append(f"INFO: {p} is at {pct}% ({info['used_gb']} GB used)")
    return {
        "service":        "damselfish-disk-monitor",
        "disks":          disks,
        "overall_status": "critical" if critical else ("warning" if warnings else "ok"),
        "warnings":       warnings,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }


def _safe_remove(path: Path, dry_run: bool = False) -> bool:
    """Remove *path* unless it looks unsafe; return True if removed."""
    if not path.exists():
        return False
    refuse_patterns = [".ssh/", "authorized_keys", "known_hosts", "id_",
                       "deploy_key", ".config", ".bashrc"]
    if any(ref in str(path) for ref in refuse_patterns):
        return False
    if any(path.name.endswith(ext) for ext in (".py", ".json", ".yml", ".yaml",
                                                ".db", ".sqlite", ".lock")):
        return False
    try:
        if dry_run:
            log.info("[dry-run] would remove: %s (%d bytes)", path, path.stat().st_size)
        else:
            path.unlink()
            log.info("removed: %s (%d bytes)", path, path.stat().st_size)
        return True
    except Exception as exc:
        log.warning("failed to remove %s: %s", path, exc)
        return False


def collect_cleanup_targets(root_dir: str | Path, max_age_days: int = 30,
                            max_size_bytes: int | None = None) -> list[Path]:
    """Walk *root_dir* and return files eligible for cleanup."""
    targets: list[tuple[int, int, Path]] = []
    now_ts = time.time()
    cutoff = now_ts - max_age_days * 86400
    root = Path(root_dir)
    if not root.is_dir():
        return []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.stat().st_mtime > cutoff:
            continue
        age = (now_ts - p.stat().st_mtime) / 86400
        size = p.stat().st_size
        ok = bool(_CLEANUP_RE.search(p.name)) or age > max_age_days
        if max_size_bytes and size > max_size_bytes and not ok:
            ok = True
        if ok:
            targets.append((age, size, p))
    targets.sort(key=lambda t: t[1], reverse=True)
    return [t[2] for t in targets]


def cleanup(dirs: list[str], max_age_days: int = 30,
            max_size_bytes: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Execute a cleanup pass and return summary."""
    started_at = datetime.now(timezone.utc)
    removed: list[dict[str, Any]] = []
    freed_bytes = 0
    for d in dirs:
        candidates = collect_cleanup_targets(d, max_age_days, max_size_bytes)
        for path_obj in candidates[:100]:
            if _safe_remove(path_obj, dry_run=dry_run):
                st = path_obj.stat()
                freed_bytes += st.st_size
                removed.append({
                    "path":     str(path_obj),
                    "size":     st.st_size,
                    "age_days": round((time.time() - st.st_mtime) / 86400, 1),
                })
    return {
        "status":        "cleanup_complete",
        "started_at":    started_at.isoformat(),
        "removed_count": len(removed),
        "freed_bytes":   freed_bytes,
        "freed_mb":      round(freed_bytes / (1024**2), 2),
        "removed":       removed[:20],
    }


def serve(port: int = 9876, monitor_paths: list[str] | None = None) -> None:
    """Start an HTTP server serving disk metrics and cleanup endpoint."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    _paths = monitor_paths or ["/"]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/monitor/disk":
                report = get_disk_report(_paths)
                self._json_reply(200, report)
            elif self.path == "/health":
                self.wfile.write(b"OK\n")
            else:
                self.send_response(404)
                self.end_headers()
        def do_POST(self):
            if self.path == "/monitor/cleanup":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length) if length else b"{}"
                params: dict = {}
                try:
                    params = json.loads(body)
                except json.JSONDecodeError:
                    pass
                dry_run  = params.pop("dry_run", True)
                dirs     = params.pop("dirs", _paths)
                max_age  = params.pop("max_age_days", 30)
                max_sz   = params.pop("max_size_bytes", None)
                result = cleanup(dirs, max_age, max_sz, dry_run=dry_run)
                self._json_reply(200, result)
            else:
                self.send_response(404)
                self.end_headers()
        def _json_reply(self, code: int, data: Any) -> None:
            body = json.dumps(data, indent=2).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, fmt, *args):
            log.debug(fmt, *args)
    srv = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Disk monitor listening on :{port}")
    print("  GET  /monitor/disk   - current disk snapshot (JSON)")
    print("  POST /monitor/cleanup - trigger cleanup (JSON body)")
    print(f"Monitor paths: {_paths}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        srv.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Damselfish disk monitor & control")
    parser.add_argument("--serve", action="store_true", help="Start HTTP server on :9876")
    parser.add_argument("--json", action="store_true", help="Raw JSON output")
    parser.add_argument("--clean", "--cleanup", action="store_true", help="Run cleanup pass")
    parser.add_argument("--dir", nargs="+", help="Dirs to monitor or clean")
    parser.add_argument("--max-age", type=int, default=30, help="Max file age (days)")
    parser.add_argument("--max-size", type=int, default=None, help="Max single-file size (bytes)")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run cleanup")
    args = parser.parse_args()
    if args.serve:
        serve(monitor_paths=args.dir)
    elif args.clean:
        dirs = args.dir or ["/"]
        result = cleanup(dirs, args.max_age, args.max_size, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
    else:
        report = get_disk_report(args.dir)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            lines = ["=== Damselfish Disk Monitor ===",
                     f"Timestamp: {report['timestamp']}",
                     f"Status: {report['overall_status'].upper()}", ""]
            for path, info in report["disks"].items():
                if "error" in info:
                    lines.append(f"X {path}: {info['error']}")
                    continue
                lines.extend([
                    f"- {path}",
                    f"   Total:    {info['total_gb']} GB",
                    f"   Used:     {info['used_gb']} GB ({info['percent_used']}%)",
                    f"   Free:     {info['free_gb']} GB",
                    f"   Available:{info['available_gb']} GB",
                    "",
                ])
            if report.get("warnings"):
                for w in report["warnings"]:
                    prefix = "!" if "WARN" in w else ("#" if "CRITICAL" in w else "i")
                    lines.append(f"{prefix} {w}")
                lines.append("")
            print("\n".join(lines))


if __name__ == "__main__":
    main()