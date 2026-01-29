"""
Training-Free Gradient Variance Suppression (TF-GVS).

Reduces gradient variance by averaging dyadic gradient proxies across
a temporal window of M consecutive mini-batches (Section 4.2).

Key equations:
  Eq (17): Instantaneous dyadic proxy  G_hat^{(i)}_l
  Eq (18): Variance-suppressed proxy  nabla_bar_D L = (1/M) sum G_hat^{(i)}

Theoretical result (Appendix A.9):
  Var(nabla_bar_D L) = sigma^2 / M
  With M=3, ~66.7% theoretical (70.2% empirical) variance reduction.
"""

import torch
from collections import deque


class TFGVS:
    """
    TF-GVS: temporal averaging of gradient proxies.

    Args:
        window_size: Temporal window M (default 3).
    """

    def __init__(self, window_size: int = 3):
        self.M = window_size
        self._window = deque(maxlen=window_size)

    def add(self, grad_proxy: torch.Tensor):
        """
        Add an instantaneous dyadic gradient proxy G_hat^{(i)} (Eq 17).
        """
        self._window.append(grad_proxy.detach().clone())

    @property
    def ready(self) -> bool:
        """True when the window is full."""
        return len(self._window) >= self.M

    def compute(self) -> torch.Tensor:
        """
        Variance-suppressed proxy (Eq 18):
          nabla_bar_D L = (1/M) sum_{i=1}^{M} G_hat^{(i)}

        Returns the averaged gradient and resets the window.
        """
        assert len(self._window) > 0, "Window is empty"
        avg = torch.stack(list(self._window)).mean(dim=0)
        self._window.clear()
        return avg

    def peek(self) -> torch.Tensor:
        """Return averaged gradient without clearing."""
        assert len(self._window) > 0
        return torch.stack(list(self._window)).mean(dim=0)

    def reset(self):
        self._window.clear()


class BatchGroupTFGVS:
    """
    Batch-grouped TF-GVS that stores full model gradients (all parameters)
    and averages them after M steps.

    Usage in training loop:
        tf_gvs = BatchGroupTFGVS(window_size=3)
        for batch in loader:
            loss.backward()
            tf_gvs.accumulate(model)
            if tf_gvs.ready:
                tf_gvs.apply_averaged_grad(model)
                optimizer.step()
                optimizer.zero_grad()
                tf_gvs.reset()
    """

    def __init__(self, window_size: int = 3):
        self.M = window_size
        self._count = 0
        self._grad_accum = {}  # param_name -> accumulated gradient

    def accumulate(self, model):
        """
        Accumulate gradients from the current backward pass.
        """
        for name, p in model.named_parameters():
            if p.grad is not None:
                if name not in self._grad_accum:
                    self._grad_accum[name] = p.grad.data.clone()
                else:
                    self._grad_accum[name].add_(p.grad.data)
        self._count += 1

    @property
    def ready(self) -> bool:
        return self._count >= self.M

    def apply_averaged_grad(self, model):
        """
        Replace model gradients with the temporal average (Eq 18).
        """
        for name, p in model.named_parameters():
            if name in self._grad_accum and p.grad is not None:
                p.grad.data.copy_(self._grad_accum[name] / self._count)

    def reset(self):
        self._grad_accum.clear()
        self._count = 0
