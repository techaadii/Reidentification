import torch
from typing_extensions import override


class SAESparsityLoss(torch.nn.Module):
    def __init__(self, rho: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._rho: float = rho

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rho_hat: torch.Tensor = x.mean(dim=0, keepdim=True)  # Shape: (1, d)
        rho: torch.Tensor = torch.as_tensor(self._rho).expand_as(rho_hat)
        loss_sparsity: torch.Tensor = torch.sum(
            rho * torch.log(rho / rho_hat)
            + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        )

        return loss_sparsity
