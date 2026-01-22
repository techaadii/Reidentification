import torch
from typing_extensions import override


class JumpRELU(torch.nn.Module):
    """
    The JumpRELU [1] activation function module.

    ### References
    - [1] Rajamanoharan, Senthooran, et al. "Jumping ahead: Improving reconstruction fidelity with jumprelu sparse autoencoders." arXiv preprint arXiv:2407.14435 (2024).
    """

    def __init__(self, theta: torch.nn.Parameter) -> None:
        super().__init__()
        self._theta: torch.nn.Parameter = theta

    @override
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mask: torch.Tensor = (z > self._theta).float()
        return mask * z
