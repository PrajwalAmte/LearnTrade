#!/bin/bash
# macOS double-click launcher — right-click → Open the first time to bypass Gatekeeper

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launcher.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "Starting LearnTrade..."

# Install uv if missing
if ! command -v uv &>/dev/null; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Current uv installers put the binary in ~/.local/bin; older ones used
    # ~/.cargo/bin. Add both so the rest of this script finds it either way.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv &>/dev/null; then
    log "ERROR: uv install finished but 'uv' is still not on PATH."
    log "Open a new Terminal window (so PATH changes take effect) and double-click"
    log "this launcher again, or install uv manually:"
    log "https://docs.astral.sh/uv/getting-started/installation/"
    read -n 1 -s -r -p "Press any key to close..."
    exit 1
fi

# Sync dependencies (fast when already installed)
log "Checking dependencies..."
uv sync --quiet

PORT="${LEANTRADE_PORT:-8787}"
URL="http://localhost:${PORT}"

# Open the browser only once the server actually answers, instead of guessing
# a fixed delay.
(
    for _ in $(seq 1 60); do
        if curl -fs "${URL}/api/health" >/dev/null 2>&1; then
            open "${URL}" &>/dev/null &
            break
        fi
        sleep 0.5
    done
) &

log "Starting server at ${URL}"
exec uv run leantrade serve --port "${PORT}" 2>&1 | tee -a "$LOG_FILE"
