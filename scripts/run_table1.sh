#!/bin/bash
# ==========================================================================
# Table 1 — Standard Benchmark Experiments
# Trains TSCD (10-layer CNN) on all 8 standard datasets.
# ==========================================================================
set -e

OUTPUT_DIR="${1:-./outputs/table1}"
SEED="${2:-42}"

echo "============================================================"
echo "  TSCD — Table 1: Standard Benchmarks"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"

DATASETS=("mnist" "fashionmnist" "svhn" "cifar10" "cifar100" "stl10" "tinyimagenet" "imagenette")

for DS in "${DATASETS[@]}"; do
    echo ""
    echo ">>> Training on ${DS} ..."
    python -m experiments.train_standard \
        --dataset "${DS}" \
        --output_dir "${OUTPUT_DIR}" \
        --seed "${SEED}"
done

echo ""
echo "============================================================"
echo "  All Table 1 experiments completed."
echo "  Results saved to: ${OUTPUT_DIR}/results/"
echo "============================================================"
