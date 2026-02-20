#!/usr/bin/env bash
set -euo pipefail

# ── Local Validator ──────────────────────────────────────────────
# Runs the local validator against a local miner.
# By default picks a random challenge from miner-test-data/.
#
# All flags are forwarded to `bb local-validate`, e.g.:
#   ./run_local_validator.sh --max-challenges 3 --max-dialogues 2
#   ./run_local_validator.sh --challenge path/to/file.json
# ─────────────────────────────────────────────────────────────────

MINER_URL="${MINER_URL:-http://localhost:8091}"
OUTPUT_DIR="${OUTPUT_DIR:-local_test_output}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Babelbit Local Validator"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Miner URL:  ${MINER_URL}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Extra args: $*"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exec uv run python -c "from babelbit import cli; cli()" local-validate \
    --miner-url "${MINER_URL}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
