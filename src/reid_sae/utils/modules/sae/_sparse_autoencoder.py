import torch
from typing_extensions import override
from reid_sae.utils.modules.sae._typing import (
    SparseAutoencoderLoss,
    SparseAutoencoderOutput,
)
from reid_sae.utils.modules.sae.sparsity_loss import SAE_SparsityLoss, L1_SparsityLoss


class SparseAutoencoder(torch.nn.Module):
    def __init__(
        self,
        dim_features: int,
        exp_factor: int,
        lambda_: float,
        hidden_activation: torch.nn.Module = torch.nn.ReLU(),
        sparsity_loss_fn: SAE_SparsityLoss = L1_SparsityLoss(),
        reconst_loss_fn: torch.nn.Module = torch.nn.MSELoss(),
    ) -> None:
        super().__init__()

        self._dim_features: int = dim_features
        self._expansion_factor: int = exp_factor
        self._hidden_activation: torch.nn.Module = hidden_activation
        self._lambda: float = lambda_

        self._sparse_module: torch.nn.Module = torch.nn.Linear(
            in_features=self._dim_features,
            out_features=self._expansion_factor * self._dim_features,
        )

        self._reconstruction_module: torch.nn.Module = torch.nn.Linear(
            in_features=self._expansion_factor * self._dim_features,
            out_features=self._dim_features,
        )

        self._reconst_loss_fn: torch.nn.Module = reconst_loss_fn
        self._sparsity_loss_fn: torch.nn.Module = sparsity_loss_fn

    def _calculate_loss(
        self,
        feature: torch.Tensor,
        reconstruction: torch.Tensor,
        hidden_pre_activation: torch.Tensor,
        hidden_activation: torch.Tensor,
    ) -> SparseAutoencoderLoss:
        reconstruction_loss: torch.Tensor = self._reconst_loss_fn(
            feature, reconstruction
        )
        sparsity_loss: torch.Tensor = self._sparsity_loss_fn(
            activation=hidden_activation, pre_activation=hidden_pre_activation
        )
        loss: torch.Tensor = reconstruction_loss + self._lambda * sparsity_loss
        return SparseAutoencoderLoss(
            total=loss, reconstruction=reconstruction_loss, sparsity=sparsity_loss
        )

    @override
    def forward(self, feature: torch.Tensor) -> SparseAutoencoderOutput:
        # Pass through the hidden layer
        pre_activation: torch.Tensor = self._sparse_module(feature)

        # Apply activation to hidden layer pre-activation
        activation: torch.Tensor = self._hidden_activation(pre_activation)

        # Reconstruct the features
        reconstruction: torch.Tensor = self._reconstruction_module(activation)

        return SparseAutoencoderOutput(
            pre_activation=pre_activation,
            activation=activation,
            reconstruction=reconstruction,
            loss=self._calculate_loss(
                feature=feature,
                reconstruction=reconstruction,
                hidden_pre_activation=pre_activation,
                hidden_activation=activation,
            ),
        )
