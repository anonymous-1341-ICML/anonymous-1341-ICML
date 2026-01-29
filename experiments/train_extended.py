"""
Table 2 — Extended Evaluation across Architectures.

Trains TSCD on CIFAR-10 and CIFAR-100 using 10 different backbone
architectures to demonstrate architecture-agnostic generalization.

Architectures (Table 2):
  CNN: ResNet-50, ResNeXt-50, RegNetY-3.2GF, ConvNeXt-Tiny,
       EfficientNetV2-S, ShuffleNetV2 2.0x
  Transformer/SSM: ViT-S/16, DeiT-S, Vim-S, CKAN-S

Usage:
    python -m experiments.train_extended --arch resnet50 --dataset cifar10
    python -m experiments.train_extended --arch all --dataset all
"""

import argparse
import os
import json
import torch

from tscd import TSCDFramework, train_tscd
from tscd.data.datasets import get_dataloader, get_dataset_config

ARCHITECTURES = [
    "resnet50",
    "resnext50",
    "regnety_3.2gf",
    "convnext_tiny",
    "efficientnetv2_s",
    "shufflenetv2_2.0x",
    "vit_s_16",
    "deit_s",
    "vim_s",
    # "ckan_s",  # requires custom implementation
]

DATASETS = ["cifar10", "cifar100"]

# Paper hyper-parameters for extended evaluation (Appendix C)
DEFAULT_HP = dict(
    epochs=500,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-4,
    gamma=0.1,
    fusion_interval=100,
    fusion_steps=10,
    goodness_threshold=2.0,
    use_mp_gbs=True,
    mp_gbs_rho=0.05,
    use_tf_gvs=True,
    tf_gvs_window=3,
    log_interval=10,
    img_size=224,  # timm models expect 224x224
)


def run_single(arch: str, dataset_name: str, output_dir: str,
               hp: dict, seed: int = 42):
    """Train one (arch, dataset) combination."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes, _, _, _ = get_dataset_config(dataset_name)
    img_size = hp["img_size"]

    print(f"\n{'='*60}")
    print(f"  Arch: {arch}  |  Dataset: {dataset_name}  |  "
          f"Image: {img_size}x{img_size}")
    print(f"{'='*60}\n")

    train_loader, test_loader, num_classes, img_size = get_dataloader(
        dataset_name, batch_size=hp["batch_size"],
        root=os.path.join(output_dir, "data"), img_size=img_size,
    )

    model = TSCDFramework(
        backbone_name=arch,
        num_classes=num_classes,
        in_channels=3,
        img_size=img_size,
        gamma=hp["gamma"],
        pretrained=False,
    )

    tag = f"{arch}_{dataset_name}"
    save_path = os.path.join(output_dir, "checkpoints", f"tscd_{tag}.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    history = train_tscd(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        epochs=hp["epochs"],
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
        device=device,
        fusion_interval=hp["fusion_interval"],
        fusion_steps=hp["fusion_steps"],
        gamma=hp["gamma"],
        use_mp_gbs=hp["use_mp_gbs"],
        mp_gbs_rho=hp["mp_gbs_rho"],
        use_tf_gvs=hp["use_tf_gvs"],
        tf_gvs_window=hp["tf_gvs_window"],
        log_interval=hp["log_interval"],
        save_path=save_path,
        goodness_threshold=hp["goodness_threshold"],
    )

    results_path = os.path.join(output_dir, "results", f"table2_{tag}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "architecture": arch,
            "dataset": dataset_name,
            "final_test_acc": history["test_acc"][-1] if history["test_acc"] else None,
            "best_test_acc": max(history["test_acc"]) if history["test_acc"] else None,
            "hyperparameters": hp,
            "history": {k: [float(v) for v in vs] for k, vs in history.items()},
        }, f, indent=2)

    best = max(history["test_acc"]) if history["test_acc"] else 0.0
    print(f"\n  {tag}: best acc = {best:.2f}%\n")
    return history


def main():
    parser = argparse.ArgumentParser(
        description="TSCD — Table 2: Extended Architecture Evaluation"
    )
    parser.add_argument("--arch", type=str, default="resnet50",
                        help="Architecture name or 'all'.")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="'cifar10', 'cifar100', or 'all'.")
    parser.add_argument("--output_dir", type=str, default="./outputs/table2")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_mp_gbs", action="store_true")
    parser.add_argument("--no_tf_gvs", action="store_true")
    args = parser.parse_args()

    hp = DEFAULT_HP.copy()
    if args.epochs is not None:
        hp["epochs"] = args.epochs
    if args.batch_size is not None:
        hp["batch_size"] = args.batch_size
    if args.lr is not None:
        hp["lr"] = args.lr
    if args.img_size is not None:
        hp["img_size"] = args.img_size
    if args.no_mp_gbs:
        hp["use_mp_gbs"] = False
    if args.no_tf_gvs:
        hp["use_tf_gvs"] = False

    archs = ARCHITECTURES if args.arch == "all" else [args.arch]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    results_table = {}
    for arch in archs:
        for ds in datasets:
            history = run_single(arch, ds, args.output_dir, hp, seed=args.seed)
            best = max(history["test_acc"]) if history["test_acc"] else 0.0
            results_table[(arch, ds)] = best

    print("\n" + "=" * 70)
    print("  Table 2 — Extended Architecture Evaluation Results")
    print("=" * 70)
    header = f"  {'Architecture':25s}"
    for ds in datasets:
        header += f" | {ds:>12s}"
    print(header)
    print("-" * 70)
    for arch in archs:
        row = f"  {arch:25s}"
        for ds in datasets:
            acc = results_table.get((arch, ds), 0.0)
            row += f" | {acc:11.2f}%"
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    main()
