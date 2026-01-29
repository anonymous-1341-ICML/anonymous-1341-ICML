"""
Dyadic Neuron Implementation for TSCD.

Each neuron maintains two intrinsic states:
  - excited state (u): positive activation
  - relaxed state (v): negative activation

Key equations from the paper:
  Eq (1): Gradient proxy  g_hat = (1/gamma)(u - v) . h^T
  Eq (8): Closed-form relaxation
    u_l <- f(W_{l-1} h_bar_{l-1} + (lambda * gamma_l / gamma_{l+1}) W_l^T (u_{l+1} - v_{l+1}))
    v_l <- f(W_{l-1} h_bar_{l-1} - ((1-lambda) * gamma_l / gamma_{l+1}) W_l^T (u_{l+1} - v_{l+1}))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DyadicLayer(nn.Module):
    """
    A layer that wraps an arbitrary nn.Module to support dyadic (u, v) states.

    Given a sub-module (e.g., nn.Linear, nn.Conv2d, or a residual block),
    this layer computes dual states via the closed-form relaxation (Eq 8).
    """

    def __init__(self, module: nn.Module, gamma: float = 0.1, lambda_asym: float = 0.5):
        super().__init__()
        self.module = module
        self.gamma = gamma
        self.lambda_asym = lambda_asym

    def forward_base(self, x):
        """Forward pass through the wrapped module."""
        return self.module(x)

    def forward(self, h_bar_prev, feedback=None):
        """
        Compute dyadic states via closed-form relaxation (Eq 8).

        Args:
            h_bar_prev: Mean activity from previous layer.
            feedback: gamma_l * W_l^T (u_{l+1} - v_{l+1}) / gamma_{l+1},
                      pre-computed feedback signal from the layer above.
                      None for the topmost layer.
        Returns:
            u, v, h_bar
        """
        a = self.forward_base(h_bar_prev)

        if feedback is not None:
            u = F.relu(a + self.lambda_asym * feedback)
            v = F.relu(a - (1.0 - self.lambda_asym) * feedback)
        else:
            u = F.relu(a)
            v = F.relu(a)

        h_bar = (u + v) / 2.0
        return u, v, h_bar

    def compute_dyadic_gradient(self, u, v, h_bar_prev):
        """
        Dyadic gradient proxy (Eq 1 / Eq 13).
        nabla_D L_l = (1/gamma)(u_l - v_l) . h_bar_{l-1}^T

        For convolutional layers this returns a scalar proxy (norm).
        """
        state_diff = (u - v) / self.gamma
        return state_diff


class DyadicNetwork(nn.Module):
    """
    Multi-layer dyadic network that wraps an arbitrary backbone.

    The backbone is decomposed into sequential blocks; each block is
    wrapped by a DyadicLayer. A top-down feedback relaxation pass refines
    the (u, v) states.
    """

    def __init__(self, blocks: nn.ModuleList, gamma: float = 0.1,
                 lambda_asym: float = 0.5, num_relaxation_steps: int = 1):
        super().__init__()
        self.gamma = gamma
        self.lambda_asym = lambda_asym
        self.num_relaxation_steps = num_relaxation_steps

        self.dyadic_layers = nn.ModuleList([
            DyadicLayer(block, gamma=gamma, lambda_asym=lambda_asym)
            for block in blocks
        ])
        self.num_layers = len(self.dyadic_layers)

    def forward(self, x):
        """
        Two-phase forward:
          1. Bottom-up: compute initial (u, v, h_bar) without feedback.
          2. Top-down relaxation: refine states using feedback from upper layers.

        Returns:
            states: list of (u, v, h_bar) per layer
            h_bars: list of h_bar per layer (index 0 = input x)
        """
        # Phase 1: bottom-up (no feedback)
        us, vs, h_bars = [], [], [x]
        h = x
        for layer in self.dyadic_layers:
            u, v, h_bar = layer(h, feedback=None)
            us.append(u)
            vs.append(v)
            h_bars.append(h_bar)
            h = h_bar

        # Phase 2: top-down relaxation
        for _ in range(self.num_relaxation_steps):
            for i in range(self.num_layers - 2, -1, -1):
                # feedback from layer i+1
                delta_next = us[i + 1] - vs[i + 1]  # (u_{l+1} - v_{l+1})
                # For simplicity, the feedback is passed as the scaled delta.
                # In a full implementation with heterogeneous gamma per layer,
                # we would multiply by gamma_l / gamma_{l+1}.
                feedback = self.gamma * delta_next
                u, v, h_bar = self.dyadic_layers[i](h_bars[i], feedback=feedback)
                us[i] = u
                vs[i] = v
                h_bars[i + 1] = (u + v) / 2.0

        states = list(zip(us, vs, h_bars[1:]))
        return states, h_bars
