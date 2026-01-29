# TSCD: Tri-Stream Coupled Dynamics for Forward Learning

Official implementation of **"Overcoming Information and Approximation Errors in Forward Learning via Tri-Stream Coupled Dynamics"** (ICML 2026).

## Overview

TSCD is a forward-learning framework that eliminates backpropagation while addressing two fundamental limitations of existing forward-learning methods:

1. **Information loss** — resolved via Tri-Stream Coupled Dynamics (cross-fusion of positive/negative streams)
2. **Gradient approximation error** — resolved via MP-GBS (bias suppression) and TF-GVS (variance suppression)

### Key Components

| Component | Section | Description |
|-----------|---------|-------------|
| Dyadic Neurons | Sec 3.1, Eq 1-8 | Dual-state neurons (excited/relaxed) for gradient-free learning |
| Tri-Stream Architecture | Sec 4.1, Eq 6-11 | Positive, Negative, Cross-Fusion streams with periodic coupling |
| MP-GBS | Sec 4.2, Eq 13-16 | Multi-Plane Forward Gradient Bias Suppression |
| TF-GVS | Sec 4.2, Eq 17-18 | Training-Free Gradient Variance Suppression |

## Repository Structure

```
icml26/
├── tscd/                        # Core library
│   ├── models/
│   │   ├── dyadic_neuron.py     # Dyadic neuron (Eq 1, 8)
│   │   ├── backbones.py         # 10-layer CNN + timm architectures
│   │   └── tscd_network.py      # TSCD framework (Eq 6-11)
│   ├── optimizers/
│   │   ├── mp_gbs.py            # MP-GBS (Eq 13-16)
│   │   └── tf_gvs.py            # TF-GVS (Eq 17-18)
│   ├── data/
│   │   ├── datasets.py          # 13 dataset loaders
│   │   └── negative_sampling.py # Negative sample generation
│   └── train.py                 # Training pipeline (Algorithm 1)
├── experiments/
│   ├── train_standard.py        # Table 1: Standard benchmarks
│   ├── train_extended.py        # Table 2: Architecture evaluation
│   ├── ablation.py              # Table 3: Ablation study
│   └── train_domain.py          # Table 8: Cross-domain generalization
├── configs/
│   ├── default.yaml             # Default hyperparameters
│   ├── cifar10.yaml
│   └── cifar100.yaml
├── scripts/
│   ├── run_all.sh               # Run all experiments
│   ├── run_table1.sh            # Table 1 experiments
│   ├── run_table2.sh            # Table 2 experiments
│   ├── run_table3.sh            # Table 3 experiments
│   └── run_table8.sh            # Table 8 experiments
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## Quick Start

### Train on CIFAR-10 (Table 1)

```bash
cd icml26
python -m experiments.train_standard --dataset cifar10
```

### Train with a specific architecture (Table 2)

```bash
python -m experiments.train_extended --arch resnet50 --dataset cifar10
```

### Run ablation study (Table 3)

```bash
python -m experiments.ablation --dataset cifar10 --config all
```

### Cross-domain evaluation (Table 8)

```bash
python -m experiments.train_domain --dataset eurosat
```

### Run all experiments

```bash
bash scripts/run_all.sh
```

## Datasets

### Standard Benchmarks (Table 1)
MNIST, Fashion-MNIST, SVHN, CIFAR-10, CIFAR-100, STL-10, Tiny ImageNet, ImageNette

### Cross-Domain (Table 8)
Food-101, DTD, NEU Surface Defect, EuroSAT, PlantVillage, Galaxy10, BreakHis

Standard datasets are downloaded automatically via torchvision. For custom datasets (Tiny ImageNet, ImageNette, NEU Surface, PlantVillage, Galaxy10, BreakHis), organize as ImageFolder:
```
data/{dataset_name}/train/{class_name}/
data/{dataset_name}/test/{class_name}/
```

## Architectures (Table 2)

| Type | Architectures |
|------|--------------|
| CNN | ResNet-50, ResNeXt-50, RegNetY-3.2GF, ConvNeXt-Tiny, EfficientNetV2-S, ShuffleNetV2 2.0x |
| Transformer/SSM | ViT-S/16, DeiT-S, Vim-S, CKAN-S |

## Hyperparameters

Key hyperparameters (Appendix C):

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Nudging factor | gamma | 0.1 | Controls dyadic relaxation |
| Fusion interval | T | 100 | Cross-fusion every T epochs |
| Fine-tune steps | T_fine | 10 | Cross-stream fine-tuning steps |
| Perturbation radius | rho | 0.05 | MP-GBS perturbation scale |
| Window size | M | 3 | TF-GVS temporal window |
| Goodness threshold | theta | 2.0 | Energy objective threshold |
| Learning rate | eta | 1e-3 | AdamW learning rate |
| Epochs | E | 500 | Total training epochs |

## Algorithm

The training procedure follows Algorithm 1 from the paper:

```
for epoch e = 1 to E:
    for each mini-batch B:
        1. Compute dyadic states via positive and negative streams
        2. TF-GVS: accumulate gradient proxies over M steps
        3. MP-GBS: compute consensus direction from 6 norm pairs
        4. Update streams with look-ahead gradients
    if e mod T == 0:
        5. Cross-fusion: transplant states, fine-tune, transfer weights
```

## Citation

```bibtex
@inproceedings{tscd2026,
  title={Overcoming Information and Approximation Errors in Forward Learning via Tri-Stream Coupled Dynamics},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
