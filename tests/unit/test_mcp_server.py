"""MCP server: tools over a real stdio connection, read-only annotations, clean errors."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from learntrade.data.store.pit import PITStore
from learntrade.setup.demo import generate_demo_data


def _call_all(data_dir: Path, calls: list[tuple[str, dict]]):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from learntrade.cli.main import app; app()", "mcp"],
        env={"LEARNTRADE_DATA_DIR": str(data_dir), "PATH": "/usr/bin:/bin"},
    )

    async def run():
        out = {}
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools = (await s.list_tools()).tools
                out["tools"] = {t.name: t for t in tools}
                out["results"] = [await s.call_tool(n, a) for n, a in calls]
        return out

    return asyncio.run(asyncio.wait_for(run(), 120))


def test_tools_work_over_stdio_and_errors_are_readable(tmp_path):
    with PITStore(tmp_path) as store:
        generate_demo_data(store, mic="XNYS", n_days=300, seed=2)
    out = _call_all(tmp_path, [
        ("status", {}),
        ("rank", {"market": "XNYS", "top": 3}),
        ("stock", {"symbol": "NOPE", "market": "XNYS"}),
        ("rank", {"market": "ZZZZ"}),
    ])
    tools = out["tools"]
    assert set(tools) == {"status", "rank", "stock", "reliability", "evaluate"}
    for name in ("status", "rank", "reliability", "evaluate"):
        assert tools[name].annotations.read_only_hint is True
    assert tools["stock"].annotations.read_only_hint is False      # may download one symbol (opt-in)

    status, rank, outside, bad = out["results"]
    st = json.loads(status.content[0].text)
    assert st["markets"][0]["mic"] == "XNYS" and st["markets"][0]["demo"] is True
    rk = json.loads(rank.content[0].text)
    assert len(rk["strongest"]) == 3 and "reliability" in rk and rk["stocks"] > 3
    assert outside.is_error and "download=True" in outside.content[0].text   # nothing fetched silently
    assert bad.is_error and "No ranking for ZZZZ" in bad.content[0].text


def test_data_dir_environment_variable_is_honoured(monkeypatch, tmp_path):
    from learntrade.core.config import load_config
    monkeypatch.setenv("LEARNTRADE_DATA_DIR", str(tmp_path / "elsewhere"))
    assert load_config().workspace.data_dir == tmp_path / "elsewhere"
    monkeypatch.delenv("LEARNTRADE_DATA_DIR")
    assert str(load_config().workspace.data_dir) in ("data", "./data")


def test_ollama_uses_the_configured_address():
    from types import SimpleNamespace
    from learntrade.llm.router import LLMRouter
    r = LLMRouter.__new__(LLMRouter)
    c = r._build_client(SimpleNamespace(provider="ollama", model="llama3.1", base_url="http://gpu-box:11434/"))
    assert c._base_url == "http://gpu-box:11434"
    c = r._build_client(SimpleNamespace(provider="ollama", model="llama3.1", base_url=None))
    assert c._base_url == "http://localhost:11434"
