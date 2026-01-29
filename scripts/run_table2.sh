#!/bin/bash
# ==========================================================================
# Table 2 — Extended Architecture Evaluation
# Trains TSCD with 9 architectures on CIFAR-10 and CIFAR-100.
# ==========================================================================
set -e

OUTPUT_DIR="${1:-./outputs/table2}"
SEED="${2:-42}"

echo "============================================================"
echo "  TSCD — Table 2: Extended Architecture Evaluation"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"

ARCHITECTURES=("resnet50" "resnext50" "regnety_3.2gf" "convnext_tiny" "efficientnetv2_s" "shufflenetv2_2.0x" "vit_s_16" "deit_s" "vim_s")
DATASETS=("cifar10" "cifar100")

for ARCH in "${ARCHITECTURES[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo ""
        echo ">>> Training ${ARCH} on ${DS} ..."
        python -m experiments.train_extended \
            --arch "${ARCH}" \
            --dataset "${DS}" \
            --output_dir "${OUTPUT_DIR}" \
            --seed "${SEED}"
    done
done

echo ""
echo "============================================================"
echo "  All Table 2 experiments completed."
echo "  Results saved to: ${OUTPUT_DIR}/results/"
echo "============================================================"
