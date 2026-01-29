"""
Tri-Stream Coupled Dynamics (TSCD) Framework.

Architecture (Section 4.1, Figure 2):
  - Positive Stream (N_P): maximizes goodness for positive data
  - Negative Stream (N_N): minimizes goodness for negative data
  - Cross-Fusion Stream (N_C): periodically inherits states from both
    streams and mines boundary-discriminative features

Key equations:
  Eq (6): Layer-wise energy objective  J^P(W^P)
  Eq (7): Potential energy difference  DeltaPhi
  Eq (8): Closed-form dyadic state relaxation
  Eq (9): State transplantation  u^C <- u^P, v^C <- v^N
  Eq (10): Cross-error weight update  DeltaW^C = (eta/gamma)(v^C - u^C) h_bar^T
  Eq (11): Coupled feedback  W^{P,N} <- 0.5 W^{P,N} + 0.5 W^C
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import get_backbone, ConvBlock


class SingleStream(nn.Module):
    """
    One of the parallel streams (Positive or Negative).

    Implements the dyadic neuron dynamics (Section 4.1):
      - Bottom-up forward pass to compute initial activations
      - Top-down feedback relaxation to produce (u, v, h_bar) per layer (Eq 8)
      - Per-layer energy loss (Eq 6-7)
    """

    def __init__(self, backbone: nn.Module, gamma: float = 0.1,
                 lambda_asym: float = 0.5, num_relaxation_steps: int = 1):
        super().__init__()
        self.backbone = backbone
        self.gamma = gamma
        self.lambda_asym = lambda_asym
        self.num_relaxation_steps = num_relaxation_steps

        self.blocks = backbone.get_blocks()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone.feat_dim

    def forward(self, x):
        """
        Two-phase dyadic forward (Eq 8):
          Phase 1: Bottom-up — compute initial (u, v, h_bar) without feedback.
          Phase 2: Top-down relaxation — refine states using feedback from
                   upper layers via W_l^T (u_{l+1} - v_{l+1}).

        Returns:
            features: final pooled feature (batch, feat_dim)
            states: list of (u, v, h_bar) per layer (for energy loss / fusion)
        """
        # --- Phase 1: bottom-up (no feedback) ---
        us, vs, h_bars = [], [], []
        h = x
        for block in self.blocks:
            if hasattr(block, 'pre_activation'):
                a = block.pre_activation(h)
                u = F.relu(a)
                v = F.relu(a)
            else:
                out = block(h)
                u = out
                v = out
            h_bar = (u + v) / 2.0
            us.append(u)
            vs.append(v)
            h_bars.append(h_bar)
            h = h_bar

        # --- Phase 2: top-down relaxation ---
        for _ in range(self.num_relaxation_steps):
            for i in range(len(self.blocks) - 2, -1, -1):
                delta_next = us[i + 1] - vs[i + 1]

                # Compute feedback: (gamma_l / gamma_{l+1}) * W_l^T(delta)
                # For simplicity we use uniform gamma, so ratio = 1.
                block_next = self.blocks[i + 1]
                if hasattr(block_next, 'compute_transpose'):
                    target_h = h_bars[i].size(2)
                    feedback = block_next.compute_transpose(delta_next, target_h)
                else:
                    # Fallback for blocks without transpose (e.g. timm modules)
                    if delta_next.shape == h_bars[i].shape:
                        feedback = delta_next
                    else:
                        # Cannot compute proper feedback; skip relaxation
                        continue

                feedback = self.gamma * feedback

                # Recompute (u, v) with feedback (Eq 8)
                h_prev = h_bars[i - 1] if i > 0 else x
                if hasattr(self.blocks[i], 'pre_activation'):
                    a = self.blocks[i].pre_activation(h_prev)
                else:
                    a = self.blocks[i](h_prev)

                us[i] = F.relu(a + self.lambda_asym * feedback)
                vs[i] = F.relu(a - (1.0 - self.lambda_asym) * feedback)
                h_bars[i] = (us[i] + vs[i]) / 2.0

        features = self.pool(h_bars[-1]).flatten(1)
        states = list(zip(us, vs, h_bars))
        return features, states

    def compute_goodness(self, features):
        """
        Goodness function: sum of squared activations.
        G(x) = sum_j (h_j)^2
        """
        return torch.sum(features ** 2, dim=1)

    def compute_layer_energy_loss(self, states, threshold=2.0,
                                  stream_type='positive'):
        """
        Per-layer energy loss (Eq 6-7).

        For positive stream: push per-layer goodness ABOVE threshold.
        For negative stream: push per-layer goodness BELOW threshold.

        The potential energy difference DeltaPhi_l (Eq 7) is captured by the
        difference between excited and relaxed state energies at each layer.

        Args:
            states: list of (u, v, h_bar) per layer.
            threshold: goodness threshold theta.
            stream_type: 'positive' or 'negative'.
        Returns:
            total_loss: scalar loss summed over layers.
        """
        total_loss = 0.0
        for u, v, h_bar in states:
            # Flatten spatial dimensions
            goodness = h_bar.pow(2).sum(dim=tuple(range(1, h_bar.dim())))
            if stream_type == 'positive':
                total_loss = total_loss + F.relu(threshold - goodness).mean()
            else:
                total_loss = total_loss + F.relu(goodness - threshold).mean()

            # Potential energy difference DeltaPhi (Eq 7):
            # Penalize large discrepancy between u and v states
            delta_phi = (u - v).pow(2).sum(dim=tuple(range(1, u.dim()))).mean()
            total_loss = total_loss + (1.0 / self.gamma) * delta_phi * 0.01

        return total_loss

    def compute_energy_loss(self, pos_features, neg_features, threshold=2.0):
        """
        Energy-based contrastive loss on final pooled features (Eq 6).

        Positive samples: push goodness above threshold.
        Negative samples: push goodness below threshold.
        """
        g_pos = self.compute_goodness(pos_features)
        g_neg = self.compute_goodness(neg_features)

        loss_pos = F.relu(threshold - g_pos).mean()
        loss_neg = F.relu(g_neg - threshold).mean()

        loss = self.lambda_asym * loss_pos + (1.0 - self.lambda_asym) * loss_neg
        return loss


class CrossFusionStream(nn.Module):
    """
    Cross-Fusion Stream (N_C) — Section 4.1.

    Periodically activated (every T epochs) to mine discriminative features
    at the decision boundary by reconciling divergent states from N_P and N_N.

    The cross-stream is initialized from the main streams' weights, then
    transplanted states are used to fine-tune, and the result is transferred
    back.
    """

    def __init__(self, backbone: nn.Module, gamma: float = 0.1,
                 lambda_asym: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.gamma = gamma
        self.lambda_asym = lambda_asym
        self.blocks = backbone.get_blocks()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone.feat_dim

    def forward(self, x):
        """Standard forward through the cross-stream backbone."""
        h = x
        block_outputs = []
        for block in self.blocks:
            h = block(h)
            block_outputs.append(h)
        features = self.pool(h).flatten(1)
        return features, block_outputs

    def compute_cross_energy_loss(self, pos_features, neg_features):
        """
        Cross-energy loss: penalize similarity between positive and negative
        manifold representations (Eq 10, Eq 23).

        This drives the cross-stream to maximize the margin between the
        strongest positive features and the suppressed negative features.
        """
        cross_error = (pos_features - neg_features).pow(2).sum(dim=1).mean()
        return cross_error


class TSCDFramework(nn.Module):
    """
    Complete TSCD Framework (Algorithm 1).

    Integrates three streams and exposes methods for the full training loop.

    Args:
        backbone_name: Name of backbone architecture.
        num_classes: Number of output classes.
        in_channels: Input channels (3 for RGB).
        img_size: Input spatial size.
        gamma: Nudging factor (default 0.1).
        lambda_asym: Asymmetry coefficient (default 0.5).
        pretrained: Use pretrained backbone weights.
    """

    def __init__(self, backbone_name: str = "10layer_cnn", num_classes: int = 10,
                 in_channels: int = 3, img_size: int = 32,
                 gamma: float = 0.1, lambda_asym: float = 0.5,
                 pretrained: bool = False):
        super().__init__()
        self.gamma = gamma
        self.lambda_asym = lambda_asym
        self.num_classes = num_classes

        # Build three copies of the backbone
        bb_pos = get_backbone(backbone_name, num_classes=num_classes,
                              in_channels=in_channels, img_size=img_size,
                              pretrained=pretrained)
        bb_neg = get_backbone(backbone_name, num_classes=num_classes,
                              in_channels=in_channels, img_size=img_size,
                              pretrained=pretrained)
        bb_cross = get_backbone(backbone_name, num_classes=num_classes,
                                in_channels=in_channels, img_size=img_size,
                                pretrained=pretrained)

        self.positive_stream = SingleStream(bb_pos, gamma, lambda_asym)
        self.negative_stream = SingleStream(bb_neg, gamma, lambda_asym)
        self.cross_stream = CrossFusionStream(bb_cross, gamma, lambda_asym)

        feat_dim = bb_pos.feat_dim

        # Shared classifier head (trained with CE on positive stream output)
        self.classifier = nn.Linear(feat_dim, num_classes)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def forward_positive(self, x):
        """Forward through positive stream, returns (features, states)."""
        return self.positive_stream(x)

    def forward_negative(self, x):
        """Forward through negative stream, returns (features, states)."""
        return self.negative_stream(x)

    def classify(self, features):
        """Produce logits from feature vector."""
        return self.classifier(features)

    def predict(self, x):
        """
        Inference using classifier head.

        The input should NOT have labels embedded. For classifier-based
        inference we zero out the first num_classes pixels to match the
        training distribution (where those pixels contain label info).
        """
        with torch.no_grad():
            # Zero out label-embedding region to avoid distribution mismatch
            x_input = x.clone()
            B, C, H, W = x_input.shape
            flat = x_input.reshape(B, -1)
            flat[:, :self.num_classes] = 0.0
            x_input = flat.reshape(B, C, H, W)

            feat, _ = self.forward_positive(x_input)
            logits = self.classify(feat)
        return logits.argmax(dim=1)

    def predict_goodness(self, x, num_classes=None):
        """
        Goodness-based FF inference (Hinton 2022).

        For each possible class c, embed c into the image, forward through
        the positive stream, compute goodness G_c = sum(h^2). Predict
        the class with the highest goodness: c* = argmax_c G_c.

        This is the standard Forward-Forward inference procedure.

        Args:
            x: (B, C, H, W) raw images (WITHOUT label embedding).
            num_classes: number of classes (uses self.num_classes if None).
        Returns:
            predictions: (B,) predicted class labels.
        """
        from ..data.negative_sampling import embed_label

        nc = num_classes or self.num_classes
        B = x.size(0)
        best_goodness = torch.full((B,), -float('inf'), device=x.device)
        best_class = torch.zeros(B, dtype=torch.long, device=x.device)

        with torch.no_grad():
            for c in range(nc):
                labels_c = torch.full((B,), c, dtype=torch.long,
                                      device=x.device)
                x_c = embed_label(x, labels_c, nc)
                feat, _ = self.forward_positive(x_c)
                goodness = self.positive_stream.compute_goodness(feat)
                better = goodness > best_goodness
                best_goodness[better] = goodness[better]
                best_class[better] = c

        return best_class

    # ------------------------------------------------------------------
    # Per-layer energy loss (Eq 6-7)
    # ------------------------------------------------------------------
    def compute_energy_loss(self, pos_features, neg_features, threshold=2.0):
        """Compute contrastive energy loss on final pooled features."""
        loss_p = self.positive_stream.compute_energy_loss(
            pos_features, neg_features, threshold
        )
        return loss_p

    def compute_full_energy_loss(self, pos_states, neg_states,
                                 pos_features, neg_features,
                                 threshold=2.0):
        """
        Full energy loss combining per-layer and final-feature terms (Eq 6).

        Args:
            pos_states: per-layer (u,v,h_bar) from positive stream.
            neg_states: per-layer (u,v,h_bar) from negative stream.
            pos_features: pooled features from positive stream.
            neg_features: pooled features from negative stream.
            threshold: goodness threshold.
        """
        # Per-layer energy: positive stream → maximize goodness
        layer_loss_pos = self.positive_stream.compute_layer_energy_loss(
            pos_states, threshold=threshold, stream_type='positive'
        )
        # Per-layer energy: negative stream → minimize goodness
        layer_loss_neg = self.negative_stream.compute_layer_energy_loss(
            neg_states, threshold=threshold, stream_type='negative'
        )
        # Final feature contrastive energy
        feature_loss = self.positive_stream.compute_energy_loss(
            pos_features, neg_features, threshold
        )

        return layer_loss_pos + layer_loss_neg + feature_loss

    # ------------------------------------------------------------------
    # Cross-Fusion (Section 4.1, activated every T epochs)
    # ------------------------------------------------------------------
    def cross_fusion_step(self, train_loader, num_classes, device,
                          fine_tune_steps=10, lr_cross=1e-3,
                          threshold=2.0):
        """
        Perform one cross-fusion cycle (Algorithm 1, Lines 13-18):

          1. Initialize cross-stream weights from main streams
          2. State transplantation (Eq 9): u^C <- u^P, v^C <- v^N
          3. Fine-tune cross-stream W^C (Eq 10)
          4. Transfer knowledge back (Eq 11):
             W^{P,N} <- 0.5 * W^{P,N} + 0.5 * W^C

        Args:
            train_loader: DataLoader for getting training batches.
            num_classes: number of classes.
            device: torch device.
            fine_tune_steps: T_fine — number of fine-tuning iterations.
            lr_cross: learning rate for cross-stream fine-tuning.
            threshold: goodness threshold.
        """
        from ..data.negative_sampling import (
            create_positive_samples,
            create_negative_samples,
        )

        # Step 0: Initialize cross-stream from average of positive + negative
        self._init_cross_from_main()

        # Step 1-2: Fine-tune cross-stream
        cross_opt = torch.optim.SGD(
            self.cross_stream.parameters(), lr=lr_cross, momentum=0.9
        )

        self.cross_stream.train()
        data_iter = iter(train_loader)

        for step in range(fine_tune_steps):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)

            images = images.to(device)
            labels = labels.to(device)
            # Label embedding: correct label for pos, wrong label for neg
            pos_images = create_positive_samples(images, labels, num_classes)
            neg_images, _ = create_negative_samples(images, labels, num_classes)

            cross_opt.zero_grad()

            # Forward through cross-stream for pos and neg
            pos_feat, _ = self.cross_stream(pos_images)
            neg_feat, _ = self.cross_stream(neg_images)

            # Cross-energy loss: penalize similarity + contrastive (Eq 10)
            cross_loss = self.cross_stream.compute_cross_energy_loss(
                pos_feat, neg_feat
            )
            # Also add contrastive goodness loss
            g_pos = pos_feat.pow(2).sum(dim=1)
            g_neg = neg_feat.pow(2).sum(dim=1)
            contrastive = (F.relu(threshold - g_pos).mean()
                           + F.relu(g_neg - threshold).mean())
            total_cross_loss = cross_loss + contrastive
            total_cross_loss.backward()
            cross_opt.step()

        # Step 3: Coupled feedback (Eq 11)
        self._transfer_cross_weights()

    def _init_cross_from_main(self):
        """
        Initialize cross-stream weights as the average of positive and
        negative stream weights. This ensures the cross-stream starts
        from a state representative of both manifolds.
        """
        pos_params = dict(self.positive_stream.named_parameters())
        neg_params = dict(self.negative_stream.named_parameters())
        cross_params = dict(self.cross_stream.named_parameters())

        for name in cross_params:
            # Map stream parameter names: blocks in single stream are
            # at backbone.blocks.* but cross uses blocks.* directly.
            # Both SingleStream and CrossFusionStream use self.blocks
            # from backbone.get_blocks(), so parameter names should match
            # via the backbone path.
            pos_name = name
            neg_name = name
            if pos_name in pos_params and neg_name in neg_params:
                cross_params[name].data.copy_(
                    0.5 * pos_params[pos_name].data
                    + 0.5 * neg_params[neg_name].data
                )

    def _transfer_cross_weights(self):
        """
        Transfer learned knowledge from cross-stream to main streams (Eq 11).
        W^{P,N} <- 0.5 * W^{P,N} + 0.5 * W^C
        """
        cross_params = dict(self.cross_stream.named_parameters())
        for name, param_pos in self.positive_stream.named_parameters():
            if name in cross_params:
                param_pos.data.copy_(
                    0.5 * param_pos.data + 0.5 * cross_params[name].data
                )
        for name, param_neg in self.negative_stream.named_parameters():
            if name in cross_params:
                param_neg.data.copy_(
                    0.5 * param_neg.data + 0.5 * cross_params[name].data
                )
