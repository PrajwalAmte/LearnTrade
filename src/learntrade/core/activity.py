"""
Cross-process "an update is running" marker.

The data file allows one writer at a time across ALL processes (web page, its background
jobs, the MCP server, terminal commands). The web page's job runner writes this marker while
a job runs, so other processes (the MCP server) can say "busy, updating" right away instead
of failing on a locked file.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_NAME = ".learntrade_job.json"


def _path(data_dir: Path) -> Path:
    return Path(data_dir) / _NAME


def mark_job(data_dir: Path, title: str) -> None:
    p = _path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"pid": os.getpid(), "title": title,
                             "started": datetime.now(tz=timezone.utc).isoformat(timespec="seconds")}))


def clear_job(data_dir: Path) -> None:
    try:
        _path(data_dir).unlink()
    except FileNotFoundError:
        pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_job(data_dir: Path) -> dict | None:
    """The job marker if its process is still alive; a stale marker (crash) is removed."""
    p = _path(data_dir)
    try:
        info = json.loads(p.read_text())
    except (FileNotFoundError, ValueError):
        return None
    if not _alive(int(info.get("pid", -1))):
        clear_job(data_dir)
        return None
    return info
