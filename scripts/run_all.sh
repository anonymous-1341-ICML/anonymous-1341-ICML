#!/bin/bash
# ==========================================================================
# Run all TSCD experiments (Tables 1, 2, 3, 8)
# ==========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${1:-42}"

echo "============================================================"
echo "  TSCD — Full Experiment Suite"
echo "============================================================"

bash "${SCRIPT_DIR}/run_table1.sh" ./outputs/table1 "${SEED}"
bash "${SCRIPT_DIR}/run_table3.sh" ./outputs/table3 "${SEED}"
bash "${SCRIPT_DIR}/run_table2.sh" ./outputs/table2 "${SEED}"
bash "${SCRIPT_DIR}/run_table8.sh" ./outputs/table8 "${SEED}"

echo ""
echo "============================================================"
echo "  All experiments completed successfully."
echo "============================================================"
