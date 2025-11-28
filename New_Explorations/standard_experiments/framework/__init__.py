"""
Framework for Grokking Experiments with Optimizer Comparison
"""

from .spectral_metrics import SpectralMetricsComputer
from .muon_official import Muon, MuonW  # Official Muon from modded-nanogpt
from .trainer import GrokkingTrainer

__all__ = [
    'SpectralMetricsComputer',
    'Muon',
    'MuonW',
    'GrokkingTrainer',
]

