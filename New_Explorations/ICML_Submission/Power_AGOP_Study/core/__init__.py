"""
Power AGOP Study Core Components

Provides:
- PowerTransformer: Decoder-only transformer (Power et al. 2022 style)
- GrokkingMLP: 3-layer MLP baseline
- ModularArithmeticDataset: Dataset with discrete and one-hot formats
- InputGradientAGOPTracker: AGOP metrics computation
"""

from .power_transformer import PowerTransformer
from .grokking_mlp import GrokkingMLP
from .datasets import (
    ModularArithmeticDataset,
    create_modular_dataset_discrete,
    create_modular_dataset_onehot,
    discrete_to_onehot,
    create_transformer_tokens,
)
from .agop_utils import InputGradientAGOPTracker

__all__ = [
    'PowerTransformer',
    'GrokkingMLP',
    'ModularArithmeticDataset',
    'create_modular_dataset_discrete',
    'create_modular_dataset_onehot',
    'discrete_to_onehot',
    'create_transformer_tokens',
    'InputGradientAGOPTracker',
]

