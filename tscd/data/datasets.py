"""
Dataset loaders for all 13 datasets evaluated in the paper (Table 8).

Standard benchmarks:
  MNIST, Fashion-MNIST, SVHN, CIFAR-10, CIFAR-100, STL-10,
  Tiny ImageNet, ImageNette

Extended evaluation:
  Food-101, DTD, NEU Surface Defect, EuroSAT, PlantVillage,
  Galaxy10, BreakHis (40x/100x/200x/400x)
"""

import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

# -----------------------------------------------------------------------
# Dataset configs: (num_classes, image_size, mean, std)
# -----------------------------------------------------------------------
DATASET_CONFIGS = {
    "mnist": (10, 32, (0.1307,), (0.3081,)),
    "fashionmnist": (10, 32, (0.2860,), (0.3530,)),
    "svhn": (10, 32, (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "cifar10": (10, 32, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": (100, 32, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "stl10": (10, 96, (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    "tinyimagenet": (200, 64, (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
    "imagenette": (10, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "food101": (101, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "dtd": (47, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "neusurface": (6, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "eurosat": (10, 64, (0.3444, 0.3803, 0.4078), (0.2035, 0.1367, 0.1152)),
    "plantvillage": (39, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "galaxy10": (10, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "breakhis": (2, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def get_dataset_config(name: str):
    """Return (num_classes, img_size, mean, std) for a dataset."""
    key = name.lower().replace("-", "").replace("_", "")
    if key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {name}. "
                         f"Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[key]


def _build_transforms(img_size, mean, std, is_train=True, in_channels=3):
    """Build standard augmentation pipeline."""
    if is_train:
        if img_size <= 32:
            tfm = T.Compose([
                T.Resize(img_size) if in_channels == 1 else T.RandomCrop(img_size, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
        else:
            tfm = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomCrop(img_size, padding=img_size // 8),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
    else:
        tfm = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    # For grayscale datasets, prepend conversion to 3-channel
    if in_channels == 1:
        tfm = T.Compose([
            T.Grayscale(num_output_channels=3),
            tfm,
        ])

    return tfm


def get_dataloader(name: str, batch_size: int = 256, root: str = "./data",
                   num_workers: int = 4, img_size: int = None):
    """
    Build train and test DataLoaders for a given dataset.

    Args:
        name: Dataset name (case-insensitive).
        batch_size: Batch size.
        root: Data root directory.
        num_workers: DataLoader workers.
        img_size: Override image size (None = use default).

    Returns:
        train_loader, test_loader, num_classes, actual_img_size
    """
    key = name.lower().replace("-", "").replace("_", "")
    num_classes, default_size, mean, std = get_dataset_config(name)
    img_size = img_size or default_size
    in_channels = 1 if key in ("mnist", "fashionmnist") else 3

    tfm_train = _build_transforms(img_size, mean, std, is_train=True,
                                  in_channels=in_channels)
    tfm_test = _build_transforms(img_size, mean, std, is_train=False,
                                 in_channels=in_channels)

    # ----- torchvision built-ins -----
    if key == "cifar10":
        ds_train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tfm_train)
        ds_test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tfm_test)
    elif key == "cifar100":
        ds_train = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=tfm_train)
        ds_test = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=tfm_test)
    elif key == "mnist":
        ds_train = torchvision.datasets.MNIST(root, train=True, download=True, transform=tfm_train)
        ds_test = torchvision.datasets.MNIST(root, train=False, download=True, transform=tfm_test)
    elif key == "fashionmnist":
        ds_train = torchvision.datasets.FashionMNIST(root, train=True, download=True, transform=tfm_train)
        ds_test = torchvision.datasets.FashionMNIST(root, train=False, download=True, transform=tfm_test)
    elif key == "svhn":
        ds_train = torchvision.datasets.SVHN(root, split="train", download=True, transform=tfm_train)
        ds_test = torchvision.datasets.SVHN(root, split="test", download=True, transform=tfm_test)
    elif key == "stl10":
        ds_train = torchvision.datasets.STL10(root, split="train", download=True, transform=tfm_train)
        ds_test = torchvision.datasets.STL10(root, split="test", download=True, transform=tfm_test)
    elif key == "food101":
        ds_train = torchvision.datasets.Food101(root, split="train", download=True, transform=tfm_train)
        ds_test = torchvision.datasets.Food101(root, split="test", download=True, transform=tfm_test)
    elif key == "dtd":
        ds_train = torchvision.datasets.DTD(root, split="train", download=True, transform=tfm_train)
        ds_test = torchvision.datasets.DTD(root, split="test", download=True, transform=tfm_test)
    elif key == "eurosat":
        ds_full = torchvision.datasets.EuroSAT(root, download=True, transform=tfm_train)
        n = len(ds_full)
        n_train = int(0.7 * n)
        ds_train, ds_test = torch.utils.data.random_split(ds_full, [n_train, n - n_train])
    else:
        # For datasets not in torchvision, expect ImageFolder layout:
        # {root}/{name}/train/  and  {root}/{name}/test/
        ds_dir = os.path.join(root, name)
        train_dir = os.path.join(ds_dir, "train")
        test_dir = os.path.join(ds_dir, "test")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"Dataset {name} not found at {ds_dir}. "
                f"Please download it and organize as ImageFolder: "
                f"{train_dir}/ and {test_dir}/"
            )
        ds_train = ImageFolder(train_dir, transform=tfm_train)
        ds_test = ImageFolder(test_dir, transform=tfm_test)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, num_classes, img_size
