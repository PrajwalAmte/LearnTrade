"""
Safe-ish model files: every saved model records the library versions it was made with.

Models are pickled scikit-learn / LightGBM objects. A pickle written by one scikit-learn or
LightGBM release can fail to load, or load and behave differently, under another. So the file
starts with a small header (plain dict) naming the versions; on load the header is read first
and a version mismatch raises ModelCompatibilityError, which callers treat as "not trained"
(the page then asks for Update data, which retrains in minutes) instead of crashing or using a
silently broken model.

Files written before this header existed are still loaded.
"""

from __future__ import annotations

import pickle
import platform
from pathlib import Path
from typing import Any

_MAGIC = "learntrade-model"


class ModelCompatibilityError(RuntimeError):
    pass


def _major_minor(v: str | None) -> str | None:
    return ".".join(str(v).split(".")[:2]) if v else None


def library_versions() -> dict[str, str | None]:
    # Read installed versions from package metadata: importing LightGBM here (just to read a
    # version) loads its OpenMP runtime, which can stall interpreter shutdown next to Polars'
    # thread pool.
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {"python": _major_minor(platform.python_version())}
    for name in ("numpy", "scikit-learn", "lightgbm"):
        try:
            out[name] = _major_minor(version(name))
        except PackageNotFoundError:
            out[name] = None
    return out


def dump(path: str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"format": _MAGIC, "versions": library_versions()}, f)
        pickle.dump(payload, f)


def load(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        first = pickle.load(f)
        if not (isinstance(first, dict) and first.get("format") == _MAGIC):
            return first                       # written before version headers existed
        saved, now = first.get("versions") or {}, library_versions()
        changed = [f"{k} {saved[k]} -> {now.get(k)}" for k in saved
                   if saved.get(k) and now.get(k) and saved[k] != now.get(k)]
        if changed:
            raise ModelCompatibilityError(
                f"{Path(path).name} was saved with different library versions ({', '.join(changed)}). "
                "Retrain it (Update data, or `learntrade train`).")
        return pickle.load(f)
