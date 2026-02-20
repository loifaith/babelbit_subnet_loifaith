#!/usr/bin/env bash
set -euo pipefail

# ── Local Validator (multi-miner) ────────────────────────────────
# Runs the local validator against one or more local miners.
# By default picks a random challenge from miner-test-data/.
#
# Set MINER_URLS (comma-separated) to test multiple miners:
#   MINER_URLS=http://localhost:8091,http://localhost:8092 ./run_local_validator.sh
#
# Or pass --miner-url flags directly:
#   ./run_local_validator.sh --miner-url http://localhost:8091 --miner-url http://localhost:8092
#
# Other flags are forwarded to `bb local-validate`:
#   ./run_local_validator.sh --max-challenges 3 --max-dialogues 2
# ─────────────────────────────────────────────────────────────────

OUTPUT_DIR="${OUTPUT_DIR:-local_test_output}"

# Check if the user already passed --miner-url in the extra args.
# If so, skip the MINER_URLS env var to avoid duplicates.
HAS_MINER_URL_FLAG=false
for arg in "$@"; do
    if [[ "$arg" == "--miner-url" ]]; then
        HAS_MINER_URL_FLAG=true
        break
    fi
done

MINER_ARGS=()
if [[ "$HAS_MINER_URL_FLAG" == false ]]; then
    MINER_URLS="${MINER_URLS:-http://localhost:8091}"
    IFS=',' read -ra URL_ARRAY <<< "${MINER_URLS}"
    for url in "${URL_ARRAY[@]}"; do
        url="$(echo -n "$url" | xargs)"
        [ -n "$url" ] && MINER_ARGS+=(--miner-url "$url")
    done
fi

# Resolve display string for banner
if [[ "${#MINER_ARGS[@]}" -gt 0 ]]; then
    DISPLAY_MINERS="${MINER_ARGS[*]}"
else
    DISPLAY_MINERS="(from command-line flags)"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Babelbit Local Validator"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Miners:     ${DISPLAY_MINERS}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Extra args: $*"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exec uv run python -c "from babelbit import cli; cli()" local-validate \
    "${MINER_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
