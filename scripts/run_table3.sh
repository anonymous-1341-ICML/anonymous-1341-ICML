#!/bin/bash
# ==========================================================================
# Table 3 — Ablation Study
# Evaluates each TSCD component on CIFAR-10 and CIFAR-100.
# ==========================================================================
set -e

OUTPUT_DIR="${1:-./outputs/table3}"
SEED="${2:-42}"

echo "============================================================"
echo "  TSCD — Table 3: Ablation Study"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"

python -m experiments.ablation \
    --dataset all \
    --config all \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED}"

echo ""
echo "============================================================"
echo "  All Table 3 experiments completed."
echo "  Results saved to: ${OUTPUT_DIR}/results/"
echo "============================================================"
