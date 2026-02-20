#!/usr/bin/env bash
set -euo pipefail

# ── Local Miner ──────────────────────────────────────────────────
# Starts the miner in dev mode (no Bittensor wallet required).
#
# Override any variable on the command line:
#   MINER_MODEL_ID=gpt2 ./run_local_miner.sh
# ─────────────────────────────────────────────────────────────────

export MINER_DEV_MODE="${MINER_DEV_MODE:-1}"
export MINER_DEVICE="${MINER_DEVICE:-cpu}"
export MINER_MODEL_ID="${MINER_MODEL_ID:-gpt2}"
export MINER_AXON_PORT="${MINER_AXON_PORT:-8092}"

# Kill any leftover process on the target port
if fuser "${MINER_AXON_PORT}/tcp" >/dev/null 2>&1; then
    echo "Port ${MINER_AXON_PORT} in use -- killing previous process..."
    fuser -k "${MINER_AXON_PORT}/tcp" >/dev/null 2>&1 || true
    sleep 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Babelbit Local Miner"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model:   ${MINER_MODEL_ID}"
echo "  Device:  ${MINER_DEVICE}"
echo "  Port:    ${MINER_AXON_PORT}"
echo "  DevMode: ${MINER_DEV_MODE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exec uv run python babelbit/miner/serve_miner.py
