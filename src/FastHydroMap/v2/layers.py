import torch
from torch import nn


def scatter_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    output = values.new_zeros((size, *values.shape[1:]))
    output.index_add_(0, index, values)
    return output


def scatter_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    output = scatter_sum(values, index, size)
    counts = values.new_zeros(size)
    counts.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    shape = (size,) + (1,) * (values.ndim - 1)
    return output / counts.clamp_min(1.0).reshape(shape)


def radial_basis(distance: torch.Tensor, bins: int) -> torch.Tensor:
    centers = torch.linspace(
        2.0, 14.0, bins, device=distance.device, dtype=distance.dtype
    )
    sigma = 2.0 if bins <= 8 else 1.0
    return torch.exp(-0.5 * ((distance[:, None] - centers[None]) / sigma) ** 2)


class ContinuousLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, dropout: float):
        super().__init__()
        self.source = nn.Linear(hidden, hidden, bias=False)
        self.filter = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge):
        source, target = edge_index
        message = self.source(h[source]) * self.filter(edge)
        aggregate = scatter_mean(message, target, len(h))
        return self.norm(h + self.drop(self.update(torch.cat((h, aggregate), dim=-1))))
