"""
Gradient Outer Product Analysis Framework

A unified framework for computing and analyzing gradient outer products
during neural network training to understand grokking phenomena.
"""

__version__ = "0.1.0"

from .gop_tracker import GOPTracker
from .gop_metrics import GOPMetrics
from .storage import HDF5Storage
from .wrapper import TrainingWrapper
from .config import ExperimentConfig

__all__ = [
    "GOPTracker",
    "GOPMetrics",
    "HDF5Storage",
    "TrainingWrapper",
    "ExperimentConfig",
]

