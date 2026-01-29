#!/bin/bash
# ==========================================================================
# Table 8 — Cross-Domain Generalization
# Evaluates TSCD (10-layer CNN) on 7 domain-specific datasets.
# ==========================================================================
set -e

OUTPUT_DIR="${1:-./outputs/table8}"
SEED="${2:-42}"

echo "============================================================"
echo "  TSCD — Table 8: Cross-Domain Generalization"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"

python -m experiments.train_domain \
    --dataset all \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED}"

echo ""
echo "============================================================"
echo "  All Table 8 experiments completed."
echo "  Results saved to: ${OUTPUT_DIR}/results/"
echo "============================================================"
