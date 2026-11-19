"""
Terminal-mode setup wizard (browser wizard runs via the API).
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt

console = Console()


def run_terminal_wizard(config_dir: Path) -> None:
    console.rule("[bold]LearnTrade Setup Wizard[/]")
    console.print(
        "\nThis wizard creates [bold]config/learntrade.toml[/]. "
        "You can re-run it at any time or use [bold]learntrade serve[/] "
        "to open the browser wizard instead.\n"
    )

    # Risk notice
    if not Confirm.ask(
        "I understand that LearnTrade is research software and not financial advice. "
        "I am responsible for every trade I place."
    ):
        console.print("[red]Setup cancelled.[/]")
        return

    # Base currency
    currency = Prompt.ask("Your home currency (ISO 4217)", default="USD").upper()

    # Markets
    markets_input = Prompt.ask(
        "Exchange MIC codes to enable (comma-separated)", default="XNYS,XNAS"
    )
    markets = [m.strip().upper() for m in markets_input.split(",") if m.strip()]

    # LLM
    llm_choice = Prompt.ask(
        "AI assistant",
        choices=["none", "local", "cloud"],
        default="none",
    )

    # Write config
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "learntrade.toml"

    llm_section = '[llm]\nenabled = false\n'
    if llm_choice == "cloud":
        llm_section = '[llm]\nenabled = true\nbudget_usd_per_day = 20.0\n'
    elif llm_choice == "local":
        llm_section = (
            '[llm]\nenabled = true\nbudget_usd_per_day = 0.0\n\n'
            '[llm.models.extract]\nprovider = "ollama"\nmodel = "qwen2.5:14b-instruct"\n'
        )

    config_content = f"""[workspace]
base_currency = "{currency}"
data_dir = "./data"
timezone = "UTC"

[markets]
enabled = {markets!r}

[horizons]
enabled = ["swing_5d", "swing_20d", "positional_60d"]

[quant]
models = ["ridge", "lgbm"]
calibration = "isotonic"
pooled_across_markets = true

{llm_section}
[portfolio]
allocator = "hrp"
max_position = 0.05
long_only = true
rebalance = "weekly"

[risk]
max_drawdown_stop = 0.20
kill_switch = true

[execution]
mode = "paper"
"""

    config_path.write_text(config_content)
    console.print(f"\n[green]Config written to {config_path}[/]")
    console.print("\nNext steps:")
    console.print("  [bold]learntrade data pull[/]  — download market data")
    console.print("  [bold]learntrade train[/]       — train quant models")
    console.print("  [bold]learntrade serve[/]        — open the web UI")
