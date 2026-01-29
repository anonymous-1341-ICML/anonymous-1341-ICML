"""
Backbone architectures for TSCD.

Supports the 10 architectures evaluated in the paper (Table 2, Appendix B):
  CNN: ResNet-50, ResNeXt-50, RegNetY-3.2GF, ConvNeXt-Tiny,
       EfficientNetV2-S, ShuffleNetV2 2.0x
  Transformer / SSM: ViT-S/16, DeiT-S, Vim-S, CKAN-S

For standard benchmarks (Table 1), a custom 10-layer CNN is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


# ---------------------------------------------------------------------------
# 10-layer CNN used for standard benchmarks (following Wu et al. 2024)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU, with separate pre_activation for dyadic computation."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def pre_activation(self, x):
        """Conv + BN without ReLU (needed for dyadic state computation Eq 8)."""
        return self.bn(self.conv(x))

    def forward(self, x):
        return F.relu(self.pre_activation(x))

    def compute_transpose(self, delta, target_h=None):
        """
        Compute W^T(delta) via transposed convolution for top-down feedback.
        Used in Eq 8: feedback = W_l^T (u_{l+1} - v_{l+1}).
        """
        s = self.conv.stride[0]
        p = self.conv.padding[0]
        k = self.conv.kernel_size[0]
        output_padding = 0
        if s > 1 and target_h is not None:
            expected_h = (delta.size(2) - 1) * s - 2 * p + k
            output_padding = target_h - expected_h
        return F.conv_transpose2d(
            delta, self.conv.weight,
            stride=self.conv.stride, padding=self.conv.padding,
            output_padding=output_padding,
        )


class TenLayerCNN(nn.Module):
    """
    10-layer CNN following prior FF literature (Wu et al. 2024; Chen et al. 2025).
    Returns feature blocks as a ModuleList for dyadic wrapping.
    """
    def __init__(self, in_channels=3, num_classes=10, base_ch=128):
        super().__init__()
        # 10 convolutional blocks, grouped into stages
        self.blocks = nn.ModuleList([
            # Stage 1: 32x32
            ConvBlock(in_channels, base_ch),
            ConvBlock(base_ch, base_ch),
            ConvBlock(base_ch, base_ch * 2, stride=2),  # -> 16x16
            # Stage 2: 16x16
            ConvBlock(base_ch * 2, base_ch * 2),
            ConvBlock(base_ch * 2, base_ch * 2),
            ConvBlock(base_ch * 2, base_ch * 4, stride=2),  # -> 8x8
            # Stage 3: 8x8
            ConvBlock(base_ch * 4, base_ch * 4),
            ConvBlock(base_ch * 4, base_ch * 4),
            ConvBlock(base_ch * 4, base_ch * 4, stride=2),  # -> 4x4
            # Stage 4: 4x4
            ConvBlock(base_ch * 4, base_ch * 4),
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base_ch * 4, num_classes)
        self._feat_dim = base_ch * 4

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    def get_blocks(self):
        """Return blocks for dyadic wrapping."""
        return self.blocks

    @property
    def feat_dim(self):
        return self._feat_dim


# ---------------------------------------------------------------------------
# timm-based backbones for extended evaluation (Table 2)
# ---------------------------------------------------------------------------
TIMM_MODEL_MAP = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "regnety_3.2gf": "regnety_032",
    "convnext_tiny": "convnext_tiny",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "shufflenetv2_2.0x": "shufflenetv2_x2_0",
    "vit_s_16": "vit_small_patch16_224",
    "deit_s": "deit_small_patch16_224",
    "vim_s": "vim_small_patch16_224",
    "ckan_s": None,  # not in timm; placeholder
}


class TimmBackbone(nn.Module):
    """Wrapper around timm models that exposes sequential blocks."""

    def __init__(self, model_name: str, num_classes: int = 10,
                 pretrained: bool = False, img_size: int = 224):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm is required for extended architectures. "
                              "Install via: pip install timm")
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes,
        )
        self._feat_dim = self.model.num_features

    def forward(self, x):
        return self.model(x)

    def get_blocks(self):
        """
        Extract sequential blocks from the timm model.
        This varies by architecture; we return the children modules.
        """
        children = list(self.model.children())
        # Typically: [stem, layer1, layer2, layer3, layer4, pool, fc]
        # We return everything except the final classifier.
        feature_blocks = nn.ModuleList(children[:-1])
        return feature_blocks

    @property
    def feat_dim(self):
        return self._feat_dim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_backbone(name: str, num_classes: int = 10, in_channels: int = 3,
                 img_size: int = 32, pretrained: bool = False):
    """
    Get a backbone network.

    Args:
        name: Architecture name. "10layer_cnn" for standard benchmarks,
              or one of the TIMM_MODEL_MAP keys for extended evaluation.
        num_classes: Number of output classes.
        in_channels: Number of input channels.
        img_size: Input image size (used for timm models).
        pretrained: Whether to use pretrained weights.

    Returns:
        backbone: nn.Module with .get_blocks() and .feat_dim
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    if name_lower in ("10layer_cnn", "10_layer_cnn", "cnn10"):
        return TenLayerCNN(in_channels=in_channels, num_classes=num_classes)

    if name_lower in TIMM_MODEL_MAP:
        timm_name = TIMM_MODEL_MAP[name_lower]
        if timm_name is None:
            raise NotImplementedError(f"Architecture {name} has no timm model; "
                                      "provide a custom implementation.")
        return TimmBackbone(timm_name, num_classes=num_classes,
                            pretrained=pretrained, img_size=img_size)

    # Fallback: try timm directly
    if HAS_TIMM:
        return TimmBackbone(name_lower, num_classes=num_classes,
                            pretrained=pretrained, img_size=img_size)

    raise ValueError(f"Unknown backbone: {name}")
