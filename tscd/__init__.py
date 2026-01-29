"""
TSCD: Tri-Stream Coupled Dynamics for Forward Learning

Official implementation of "Overcoming Information and Approximation Errors
in Forward Learning via Tri-Stream Coupled Dynamics" (ICML 2026).
"""

from .models.tscd_network import TSCDFramework
from .models.dyadic_neuron import DyadicLayer, DyadicNetwork
from .models.backbones import get_backbone
from .optimizers.mp_gbs import MPGBS
from .optimizers.tf_gvs import TFGVS, BatchGroupTFGVS
from .train import train_tscd

__version__ = "1.0.0"
__all__ = [
    "TSCDFramework",
    "DyadicLayer",
    "DyadicNetwork",
    "get_backbone",
    "MPGBS",
    "TFGVS",
    "BatchGroupTFGVS",
    "train_tscd",
]
