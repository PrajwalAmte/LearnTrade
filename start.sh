#!/bin/bash
# Linux launcher — also called by Start LeanTrade.command on macOS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launcher.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "Starting LearnTrade..."

if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Current uv installers put the binary in ~/.local/bin; older ones used
    # ~/.cargo/bin. Add both so the rest of this script finds it either way.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv &>/dev/null; then
    log "ERROR: uv install finished but 'uv' is still not on PATH."
    log "Open a new terminal (so PATH changes take effect) and re-run this script,"
    log "or install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

log "Checking dependencies..."
uv sync --quiet

PORT="${LEANTRADE_PORT:-8787}"
URL="http://localhost:${PORT}"

# Open the browser only once the server actually answers, instead of guessing
# a fixed delay. Runs in the background so it never blocks the server itself.
(
    for _ in $(seq 1 60); do
        if curl -fs "${URL}/api/health" >/dev/null 2>&1; then
            if command -v xdg-open &>/dev/null; then
                xdg-open "${URL}" &>/dev/null &
            elif command -v open &>/dev/null; then
                open "${URL}" &>/dev/null &
            fi
            break
        fi
        sleep 0.5
    done
) &

log "Starting server at ${URL}"
exec uv run leantrade serve --port "${PORT}" 2>&1 | tee -a "$LOG_FILE"
