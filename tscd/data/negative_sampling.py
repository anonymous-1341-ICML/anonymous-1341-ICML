"""
Negative sample generation for Forward-Forward learning.

Following Hinton (2022), positive samples embed the correct label,
while negative samples embed a wrong label.  The network must learn
to distinguish the two via the goodness function.
"""

import torch
import torch.nn.functional as F


def embed_label(images: torch.Tensor, labels: torch.Tensor,
                num_classes: int) -> torch.Tensor:
    """
    Embed one-hot label into the first `num_classes` pixels of an image.

    This is the standard label overlay used in FF literature:
    the first `num_classes` entries of the flattened image are replaced
    with the one-hot encoded label.

    Args:
        images: (B, C, H, W) or (B, D) tensor.
        labels: (B,) integer labels.
        num_classes: total number of classes.

    Returns:
        images_with_label: same shape as input, with label embedded.
    """
    is_image = images.dim() == 4
    if is_image:
        B, C, H, W = images.shape
        flat = images.reshape(B, -1)
    else:
        flat = images.clone()
        B = flat.size(0)

    one_hot = F.one_hot(labels, num_classes).float().to(flat.device)

    # Replace first num_classes entries
    flat = flat.clone()
    flat[:, :num_classes] = one_hot

    if is_image:
        return flat.reshape(B, C, H, W)
    return flat


def create_positive_samples(images: torch.Tensor, labels: torch.Tensor,
                            num_classes: int) -> torch.Tensor:
    """
    Create positive samples by embedding the CORRECT label (Hinton 2022).

    Positive data: original image with its true label embedded in the
    first `num_classes` pixel positions.

    Args:
        images: (B, C, H, W) input images.
        labels: (B,) true labels.
        num_classes: total number of classes.

    Returns:
        pos_images: (B, C, H, W) images with correct label embedded.
    """
    return embed_label(images, labels, num_classes)


def create_negative_samples(images: torch.Tensor, labels: torch.Tensor,
                            num_classes: int,
                            method: str = "label_flip") -> tuple:
    """
    Create negative samples for contrastive forward learning.

    Following Hinton (2022), negative samples have a WRONG label embedded
    in the first `num_classes` pixel positions.

    Methods:
      - "label_flip": Keep images, assign random wrong labels, embed them.
      - "hybrid": Mixup between different-class images with wrong labels.

    Args:
        images: (B, C, H, W) input images.
        labels: (B,) true labels.
        num_classes: total classes.
        method: sampling strategy.

    Returns:
        neg_images: (B, C, H, W) negative images with wrong label embedded.
        neg_labels: (B,) wrong labels.
    """
    B = images.size(0)
    device = labels.device

    if method == "label_flip":
        neg_labels = torch.randint(0, num_classes, (B,), device=device)
        # Ensure different from true labels
        same = neg_labels == labels
        while same.any():
            neg_labels[same] = torch.randint(
                0, num_classes, (same.sum().item(),), device=device
            )
            same = neg_labels == labels
        # Embed wrong label into the image
        neg_images = embed_label(images, neg_labels, num_classes)

    elif method == "hybrid":
        # Mixup between different-class images
        perm = torch.randperm(B, device=device)
        alpha = 0.5
        neg_images = alpha * images + (1.0 - alpha) * images[perm]
        neg_labels = labels[perm]
        # Embed wrong label into mixed image
        neg_images = embed_label(neg_images, neg_labels, num_classes)
    else:
        raise ValueError(f"Unknown negative sampling method: {method}")

    return neg_images, neg_labels
