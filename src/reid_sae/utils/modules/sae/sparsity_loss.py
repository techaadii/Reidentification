import torch
from typing_extensions import override
from abc import abstractmethod, ABC


class SAE_SparsityLoss(torch.nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self, activation: torch.Tensor, pre_activation: torch.Tensor
    ) -> torch.Tensor:
        pass


class KL_SparsityLoss(SAE_SparsityLoss):
    """
    The SAE sparsity loss discussed in [1].

    ### References
    - [1] Ng, Andrew. "Sparse autoencoder." CS294A Lecture notes 72.2011 (2011): 1-19.
    """

    _EPSILON: float = 1e-8

    def __init__(self, rho: float) -> None:
        super().__init__()

        self._rho: float = rho

    @override
    def forward(
        self, activation: torch.Tensor, pre_activation: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the sparsity loss for the hidden layer activations
        of a sparse autoencoder.

        :param activation: The activations from the hidden layer of the SAE.
            Shape: `(m , s2)`
        :type activation: torch.Tensor
        :param pre_activation: The pre activations, not used here.
        :type pre_activation: torch.Tensor
        :return: The sparsity loss
        :rtype: torch.Tensor
        """
        # Mean along dimension 0 (batch size)
        rho_hat: torch.Tensor = torch.mean(activation, dim=0, keepdim=True).clamp(
            min=KL_SparsityLoss._EPSILON, max=1 - KL_SparsityLoss._EPSILON
        )  # Shape: (1, s2)

        # Summation along dimension 1 (neurons)
        loss: torch.Tensor = torch.sum(
            self._rho * torch.log(self._rho / rho_hat)
            + (1 - self._rho) * torch.log((1 - self._rho) / (1 - rho_hat)),
            dim=1,
        )  # Shape: (1, 1)

        return loss


class L0_SparsityLoss(SAE_SparsityLoss):
    """
    L0 sparsity loss for a JumpRELU SAE.
    """

    def __init__(self, dim_activation: int, epsilon: float = 0.1) -> None:
        super().__init__()

        self._theta: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand(dim_activation) * 0.1
        )
        self._epsilon: float = epsilon

    @property
    def theta(self) -> torch.Tensor:
        return self._theta

    @override
    def forward(
        self, activation: torch.Tensor, pre_activation: torch.Tensor
    ) -> torch.Tensor:
        sparsity_loss: torch.Tensor = (
            torch.sigmoid((pre_activation - self._theta) / self._epsilon)
            .sum(dim=-1)
            .mean()
        )

        return sparsity_loss


class L1_SparsityLoss(SAE_SparsityLoss):
    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(
        self, activation: torch.Tensor, pre_activation: torch.Tensor
    ) -> torch.Tensor:
        # L1 sparsity - calculate L1 loss with target = 0
        sparsity_loss: torch.Tensor = torch.sum(torch.abs(activation), dim=-1).mean()
        return sparsity_loss
