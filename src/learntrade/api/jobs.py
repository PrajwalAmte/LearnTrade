"""
Tiny background-job runner for the web UI.

Each job is a list of steps; each step is one ``learntrade`` CLI call run in a
subprocess. Reusing the CLI means the UI exercises exactly the code paths that
are tested from the terminal. Only one job runs at a time (the data files allow a
single writer), and read endpoints report "busy" while it does.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    title: str           # plain-English label shown to the user
    args: list[str]      # CLI arguments after "learntrade"
    optional: bool = False  # failure does not fail the whole job


@dataclass
class Job:
    id: str
    title: str
    steps: list[Step]
    status: str = "running"          # running | done | failed
    current: int = 0
    log: list[str] = field(default_factory=list)
    error: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "status": self.status,
            "current": self.current, "error": self.error,
            "steps": [s.title for s in self.steps],
            "log_tail": self.log[-12:],
        }


_lock = threading.Lock()
_jobs: dict[str, Job] = {}
_active: str | None = None


def is_busy() -> bool:
    with _lock:
        return _active is not None and _jobs[_active].status == "running"


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def active_job() -> Job | None:
    with _lock:
        return _jobs.get(_active) if _active else None


def start(title: str, steps: list[Step]) -> Job:
    global _active
    with _lock:
        if _active and _jobs[_active].status == "running":
            raise RuntimeError("Another task is already running. Please wait for it to finish.")
        job = Job(id=uuid.uuid4().hex[:8], title=title, steps=steps)
        _jobs[job.id] = job
        _active = job.id
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return job


def _data_dir():
    from learntrade.core.config import get_config
    return get_config().workspace.data_dir


def _run(job: Job) -> None:
    from learntrade.core.activity import clear_job, mark_job
    try:
        mark_job(_data_dir(), job.title)
    except Exception:
        pass
    try:
        _run_steps(job)
    finally:
        try:
            clear_job(_data_dir())
        except Exception:
            pass


def _run_steps(job: Job) -> None:
    # A step may have to wait for another process (e.g. an MCP evaluation) to release the
    # data file, so let it wait minutes instead of failing straight away.
    env = {**os.environ, "LEARNTRADE_DB_LOCK_WAIT": os.environ.get("LEARNTRADE_JOB_LOCK_WAIT", "900")}
    for i, step in enumerate(job.steps):
        job.current = i
        job.log.append(f"▶ {step.title}")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "learntrade.cli.main", *step.args],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    job.log.append(line[-300:])
            code = proc.wait()
        except Exception as exc:  # could not even launch
            code = 1
            job.log.append(str(exc))
        if code != 0 and not step.optional:
            job.status = "failed"
            job.error = f"“{step.title}” did not finish. See details below."
            return
    job.current = len(job.steps)
    job.status = "done"
