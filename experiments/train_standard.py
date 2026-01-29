"""
Table 1 — Standard Benchmarks.

Trains the TSCD framework with a 10-layer CNN backbone on each of the 8
standard datasets: MNIST, Fashion-MNIST, SVHN, CIFAR-10, CIFAR-100,
STL-10, Tiny ImageNet, ImageNette.

Usage:
    python -m experiments.train_standard --dataset cifar10
    python -m experiments.train_standard --dataset all
"""

import argparse
import os
import json
import torch

from tscd import TSCDFramework, train_tscd
from tscd.data.datasets import get_dataloader, get_dataset_config

STANDARD_DATASETS = [
    "mnist", "fashionmnist", "svhn", "cifar10",
    "cifar100", "stl10", "tinyimagenet", "imagenette",
]

# Paper hyper-parameters (Section 5.1, Appendix C)
DEFAULT_HP = dict(
    backbone="10layer_cnn",
    epochs=500,
    batch_size=256,
    lr=1e-3,
    weight_decay=0.0,
    gamma=0.1,
    fusion_interval=100,
    fusion_steps=10,
    goodness_threshold=2.0,
    use_mp_gbs=True,
    mp_gbs_rho=0.05,
    use_tf_gvs=True,
    tf_gvs_window=3,
    log_interval=10,
)


def run_single(dataset_name: str, output_dir: str, hp: dict, seed: int = 42):
    """Train TSCD on one dataset and save results."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes, img_size, _, _ = get_dataset_config(dataset_name)

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}  |  Classes: {num_classes}  |  "
          f"Image: {img_size}x{img_size}")
    print(f"{'='*60}\n")

    train_loader, test_loader, num_classes, img_size = get_dataloader(
        dataset_name, batch_size=hp["batch_size"], root=os.path.join(output_dir, "data"),
    )

    model = TSCDFramework(
        backbone_name=hp["backbone"],
        num_classes=num_classes,
        in_channels=3,
        img_size=img_size,
        gamma=hp["gamma"],
    )

    save_path = os.path.join(output_dir, "checkpoints",
                             f"tscd_{dataset_name}.pt")
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

    # Save results
    results_path = os.path.join(output_dir, "results",
                                f"table1_{dataset_name}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "final_test_acc": history["test_acc"][-1] if history["test_acc"] else None,
            "best_test_acc": max(history["test_acc"]) if history["test_acc"] else None,
            "hyperparameters": hp,
            "history": {k: [float(v) for v in vs] for k, vs in history.items()},
        }, f, indent=2)

    print(f"\n  {dataset_name}: best acc = "
          f"{max(history['test_acc']):.2f}%\n")
    return history


def main():
    parser = argparse.ArgumentParser(description="TSCD — Table 1: Standard Benchmarks")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Dataset name or 'all' for all standard benchmarks.")
    parser.add_argument("--output_dir", type=str, default="./outputs/table1",
                        help="Output directory for checkpoints and results.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
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
    if args.no_mp_gbs:
        hp["use_mp_gbs"] = False
    if args.no_tf_gvs:
        hp["use_tf_gvs"] = False

    datasets = STANDARD_DATASETS if args.dataset == "all" else [args.dataset]

    results_summary = {}
    for ds in datasets:
        history = run_single(ds, args.output_dir, hp, seed=args.seed)
        if history["test_acc"]:
            results_summary[ds] = max(history["test_acc"])

    print("\n" + "=" * 60)
    print("  Table 1 — Standard Benchmark Results (TSCD + 10-Layer CNN)")
    print("=" * 60)
    for ds, acc in results_summary.items():
        print(f"  {ds:20s}: {acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
