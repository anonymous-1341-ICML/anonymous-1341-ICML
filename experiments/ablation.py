"""
Table 3 — Ablation Study.

Evaluates the contribution of each TSCD component:
  (a) Baseline FF (no TSCD components)
  (b) + Tri-Stream (TS) only
  (c) + Tri-Stream + TF-GVS
  (d) + Tri-Stream + MP-GBS
  (e) Full TSCD (TS + TF-GVS + MP-GBS)

Run on CIFAR-10 and CIFAR-100 with 10-layer CNN.

Usage:
    python -m experiments.ablation --dataset cifar10
    python -m experiments.ablation --dataset all
"""

import argparse
import os
import json
import torch

from tscd import TSCDFramework, train_tscd
from tscd.data.datasets import get_dataloader, get_dataset_config

DATASETS = ["cifar10", "cifar100"]

BASE_HP = dict(
    backbone="10layer_cnn",
    epochs=500,
    batch_size=256,
    lr=1e-3,
    weight_decay=0.0,
    gamma=0.1,
    goodness_threshold=2.0,
    mp_gbs_rho=0.05,
    tf_gvs_window=3,
    log_interval=10,
)

# Ablation configurations (Table 3)
ABLATION_CONFIGS = {
    "baseline_ff": {
        "fusion_interval": 0,     # no cross-fusion
        "fusion_steps": 0,
        "use_mp_gbs": False,
        "use_tf_gvs": False,
    },
    "ts_only": {
        "fusion_interval": 100,
        "fusion_steps": 10,
        "use_mp_gbs": False,
        "use_tf_gvs": False,
    },
    "ts_tfgvs": {
        "fusion_interval": 100,
        "fusion_steps": 10,
        "use_mp_gbs": False,
        "use_tf_gvs": True,
    },
    "ts_mpgbs": {
        "fusion_interval": 100,
        "fusion_steps": 10,
        "use_mp_gbs": True,
        "use_tf_gvs": False,
    },
    "full_tscd": {
        "fusion_interval": 100,
        "fusion_steps": 10,
        "use_mp_gbs": True,
        "use_tf_gvs": True,
    },
}

ABLATION_LABELS = {
    "baseline_ff": "Baseline FF",
    "ts_only": "+ Tri-Stream (TS)",
    "ts_tfgvs": "+ TS + TF-GVS",
    "ts_mpgbs": "+ TS + MP-GBS",
    "full_tscd": "Full TSCD",
}


def run_ablation(config_name: str, dataset_name: str, output_dir: str,
                 base_hp: dict, seed: int = 42):
    """Run one ablation configuration."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hp = {**base_hp, **ABLATION_CONFIGS[config_name]}

    num_classes, img_size, _, _ = get_dataset_config(dataset_name)

    label = ABLATION_LABELS[config_name]
    print(f"\n{'='*60}")
    print(f"  Ablation: {label}  |  Dataset: {dataset_name}")
    print(f"  MP-GBS: {hp['use_mp_gbs']}  |  TF-GVS: {hp['use_tf_gvs']}  |  "
          f"Fusion: {'ON' if hp['fusion_interval'] > 0 else 'OFF'}")
    print(f"{'='*60}\n")

    train_loader, test_loader, num_classes, img_size = get_dataloader(
        dataset_name, batch_size=hp["batch_size"],
        root=os.path.join(output_dir, "data"),
    )

    model = TSCDFramework(
        backbone_name=hp["backbone"],
        num_classes=num_classes,
        in_channels=3,
        img_size=img_size,
        gamma=hp["gamma"],
    )

    tag = f"{config_name}_{dataset_name}"
    save_path = os.path.join(output_dir, "checkpoints", f"ablation_{tag}.pt")
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

    results_path = os.path.join(output_dir, "results", f"table3_{tag}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "config": config_name,
            "label": label,
            "dataset": dataset_name,
            "final_test_acc": history["test_acc"][-1] if history["test_acc"] else None,
            "best_test_acc": max(history["test_acc"]) if history["test_acc"] else None,
            "hyperparameters": hp,
            "history": {k: [float(v) for v in vs] for k, vs in history.items()},
        }, f, indent=2)

    best = max(history["test_acc"]) if history["test_acc"] else 0.0
    print(f"\n  {label} ({dataset_name}): best acc = {best:.2f}%\n")
    return best


def main():
    parser = argparse.ArgumentParser(description="TSCD — Table 3: Ablation Study")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="'cifar10', 'cifar100', or 'all'.")
    parser.add_argument("--config", type=str, default="all",
                        help="Ablation config name or 'all'.")
    parser.add_argument("--output_dir", type=str, default="./outputs/table3")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hp = BASE_HP.copy()
    if args.epochs is not None:
        hp["epochs"] = args.epochs
    if args.batch_size is not None:
        hp["batch_size"] = args.batch_size

    configs = list(ABLATION_CONFIGS.keys()) if args.config == "all" else [args.config]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    results_table = {}
    for cfg in configs:
        for ds in datasets:
            best_acc = run_ablation(cfg, ds, args.output_dir, hp, seed=args.seed)
            results_table[(cfg, ds)] = best_acc

    print("\n" + "=" * 70)
    print("  Table 3 — Ablation Study Results")
    print("=" * 70)
    header = f"  {'Configuration':25s}"
    for ds in datasets:
        header += f" | {ds:>12s}"
    print(header)
    print("-" * 70)
    for cfg in configs:
        label = ABLATION_LABELS[cfg]
        row = f"  {label:25s}"
        for ds in datasets:
            acc = results_table.get((cfg, ds), 0.0)
            row += f" | {acc:11.2f}%"
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    main()
