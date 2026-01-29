"""
Multi-Plane Forward Gradient Bias Suppression (MP-GBS).

Mitigates estimation bias by filtering out non-robust gradient components
via the intersection of multiple optimization manifolds (Section 4.2).

Key equations:
  Eq (13): Dyadic gradient proxy  nabla_D L(W)
  Eq (14): Anisotropic perturbation  epsilon^(k)
  Eq (15): Consensus direction  d* = argmax ...
  Eq (16): Look-ahead weight update  W_{t+1} <- W_t - eta nabla_D L|_{W_t + d*}
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class MPGBS:
    """
    MP-GBS optimizer wrapper.

    Wraps a base optimizer and augments each step with a multi-plane
    perturbation to suppress gradient bias.

    Args:
        model: The neural network module whose parameters are optimized.
        base_optimizer: A standard optimizer (e.g. AdamW).
        rho: Perturbation radius (default 0.05).
        norm_pairs: Conjugate (p, q) pairs satisfying 1/p + 1/q = 1.
                    Default: 6 pairs organized into 3 geometric planes.
        scaler: Optional torch.amp.GradScaler for FP16 mixed precision.
    """

    DEFAULT_NORM_PAIRS = [
        (2.0, 2.0),
        (1.0, float("inf")),
        (float("inf"), 1.0),
        (1.5, 3.0),
        (3.0, 1.5),
        (4.0, 4.0 / 3.0),
    ]

    def __init__(self, model: nn.Module, base_optimizer, rho: float = 0.05,
                 norm_pairs=None, scaler=None):
        self.model = model
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.norm_pairs = norm_pairs or self.DEFAULT_NORM_PAIRS
        self.scaler = scaler

    # ------------------------------------------------------------------
    # Core perturbation logic (Eq 14)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_perturbation(grad_flat: torch.Tensor, p: float, q: float,
                              rho: float) -> torch.Tensor:
        """
        Compute a single anisotropic perturbation (Eq 14).

        epsilon^(k) = rho * sign(grad) * |grad|^{q-1} / ||grad||_q^{q/p}
        """
        sign_g = torch.sign(grad_flat)
        abs_g = grad_flat.abs().clamp(min=1e-12)

        # |grad|^{q-1}
        if q == float("inf"):
            w = torch.ones_like(abs_g)
        else:
            w = abs_g.pow(q - 1.0)

        # ||grad||_q
        if q == float("inf"):
            norm_q = abs_g.max()
        elif q == 1.0:
            norm_q = abs_g.sum()
        else:
            norm_q = abs_g.pow(q).sum().pow(1.0 / q)

        # normalization factor ||grad||_q^{q/p}
        if p == float("inf") or q == float("inf"):
            denom = norm_q.clamp(min=1e-12)
        else:
            denom = norm_q.pow(q / p).clamp(min=1e-12)

        return rho * sign_g * w / denom

    # ------------------------------------------------------------------
    # Consensus direction (Eq 15)
    # ------------------------------------------------------------------
    def _compute_consensus(self, grad_flat: torch.Tensor) -> torch.Tensor:
        """
        Build 6 perturbation vectors from the norm pairs, organize them
        into 3 conjugate geometric planes, and compute the consensus
        direction d* as the projection-weighted average.
        """
        perturbations = [
            self._compute_perturbation(grad_flat, p, q, self.rho)
            for p, q in self.norm_pairs
        ]

        # Weighted average by alignment with the gradient proxy
        d_star = torch.zeros_like(grad_flat)
        total_w = 0.0
        for eps in perturbations:
            w = torch.dot(eps, grad_flat).abs().item()
            d_star += w * eps
            total_w += w

        if total_w > 0:
            d_star /= total_w

        # Project onto L2 ball of radius rho
        d_norm = d_star.norm(p=2)
        if d_norm > self.rho:
            d_star = d_star * (self.rho / d_norm)

        return d_star

    # ------------------------------------------------------------------
    # Step (Eq 16)
    # ------------------------------------------------------------------
    def step(self, closure=None):
        """
        Perform one MP-GBS optimization step.

        1. Collect current gradients g = nabla_D L(W_t).
        2. Compute consensus direction d*.
        3. Perturb parameters: W_t -> W_t + d*.
        4. Recompute gradients at perturbed point (via closure).
        5. Restore parameters and apply update with perturbed gradients.

        Args:
            closure: A callable that clears gradients, does forward + backward,
                     and returns the loss.  Required for the look-ahead step.
        """
        # Unscale gradients if using mixed precision (before gathering)
        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.unscale_(self.base_optimizer)

        if closure is None:
            # Fall back to standard step without look-ahead
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.step(self.base_optimizer)
            else:
                self.base_optimizer.step()
            return

        # Step 1: collect current (unscaled) gradient vector
        grad_vec = self._gather_grad_flat()

        if grad_vec is None or grad_vec.norm() < 1e-12:
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.step(self.base_optimizer)
            else:
                self.base_optimizer.step()
            return

        # Step 2: consensus direction
        d_star = self._compute_consensus(grad_vec)

        # Step 3: perturb parameters
        self._perturb_params(d_star, sign=+1)

        # Step 4: recompute loss & gradients at W + d*
        closure()

        # Unscale the new gradients from the closure
        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.unscale_(self.base_optimizer)

        # Step 5: restore original parameters, then base optimizer step
        self._perturb_params(d_star, sign=-1)
        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.step(self.base_optimizer)
        else:
            self.base_optimizer.step()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _gather_grad_flat(self):
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.view(-1))
        if len(grads) == 0:
            return None
        return torch.cat(grads)

    def _perturb_params(self, d_flat: torch.Tensor, sign: int = 1):
        offset = 0
        with torch.no_grad():
            for p in self.model.parameters():
                numel = p.numel()
                if p.grad is not None:
                    p.data.add_(sign * d_flat[offset:offset + numel].view_as(p))
                offset += numel

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
