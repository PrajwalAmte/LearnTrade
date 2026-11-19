"""Turn low-level exceptions into one readable line with a fix, where we know it."""

from __future__ import annotations


def friendly_error(exc: BaseException) -> str:
    text = str(exc)
    if "libomp" in text:
        return ("LightGBM needs the OpenMP library on macOS. Fix: run `brew install libomp`, "
                "then try again. (The ridge model works without it.)")
    first = text.strip().splitlines()[0] if text.strip() else type(exc).__name__
    return first[:200]
