from dataclasses import dataclass

import torch


@dataclass
class SparseAutoencoderLoss:
    total: torch.Tensor
    """The combined loss"""

    reconstruction: torch.Tensor
    """The reconstruction loss"""

    sparsity: torch.Tensor
    """The sparsity loss"""


@dataclass
class SparseAutoencoderOutput:
    pre_activation: torch.Tensor
    """Pre-activation from the hidden layer"""

    activation: torch.Tensor
    """The activation from the hidden layer"""

    reconstruction: torch.Tensor
    """The feature reconstructed by the SAE"""

    loss: SparseAutoencoderLoss
    """The losses related to the SAE"""
