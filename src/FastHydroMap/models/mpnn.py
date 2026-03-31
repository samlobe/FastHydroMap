"""
FastHydroMap message-passing models.

FdewetMPNN predicts direct per-residue Fdewet as
mu_fixed + intrinsic(local) + context(message-passed).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..tensor_graph import TensorGraph

# ─── default hyper-params for the locked production model ──────────────
NODE_IN = 32
EDGE_DIM = 84
HIDDEN = 24
DEPTH = 2
DROPOUT = 0.1
EDGE_DROP = 0.1
HEAD_HIDDEN = 20
DEFAULT_FDEWET_MU = 0.0


class EvoLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, dropout: float = DROPOUT, edge_drop: float = EDGE_DROP):
        super().__init__()
        self.dropout = dropout
        self.edge_drop = edge_drop
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.phi_v = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.norm_e = nn.LayerNorm(edge_dim)
        self.norm_v = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor):
        src, dst = edge_index
        keep_expand = 1.0
        if self.training and self.edge_drop > 0.0:
            keep = (torch.rand(src.size(0), device=h.device) >= self.edge_drop).float()
            keep_expand = keep.unsqueeze(-1)

        e_msg = self.phi_e(torch.cat([h[src], h[dst], e], -1)) * keep_expand
        e = self.norm_e(e + self.drop(e_msg))

        m = self.phi_v(torch.cat([h[src], h[dst], e], -1)) * keep_expand
        agg = _scatter_mean(m, dst, dim_size=h.size(0))
        h = self.norm_v(h + self.drop(agg))
        return h, e


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, *, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size, src.shape[1]), dtype=src.dtype, device=src.device)
    counts = torch.zeros((dim_size, 1), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    counts.index_add_(0, index, torch.ones((src.shape[0], 1), dtype=src.dtype, device=src.device))
    return out / counts.clamp_min_(1.0)


class FdewetMPNN(nn.Module):
    """
    Direct Fdewet model with a fixed global mean, an intrinsic head before
    message passing, and a context head after message passing.
    """

    def __init__(
        self,
        node_in: int = NODE_IN,
        edge_dim: int = EDGE_DIM,
        hidden: int = HIDDEN,
        depth: int = DEPTH,
        mu: float = DEFAULT_FDEWET_MU,
        dropout: float = DROPOUT,
        edge_drop: float = EDGE_DROP,
        head_hidden: int = HEAD_HIDDEN,
    ):
        super().__init__()
        self.node_in = nn.Linear(node_in, hidden)
        self.intrinsic_head = self._make_head(hidden, head_hidden, dropout)
        self.layers = nn.ModuleList([EvoLayer(hidden, edge_dim, dropout=dropout, edge_drop=edge_drop) for _ in range(depth)])
        self.context_head = self._make_head(hidden, head_hidden, dropout)
        self.register_buffer("mu", torch.tensor(float(mu), dtype=torch.float32))

    @staticmethod
    def _make_head(hidden: int, head_hidden: int, dropout: float) -> nn.Module:
        if head_hidden <= 0:
            return nn.Linear(hidden, 1)
        return nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, d: TensorGraph, return_parts: bool = False):
        h0 = self.node_in(d.x)
        intrinsic = self.intrinsic_head(h0).squeeze(-1)

        h = h0
        e = d.edge_attr
        for layer in self.layers:
            h, e = layer(h, e, d.edge_index)

        context = self.context_head(h).squeeze(-1)
        intrinsic_total = self.mu + intrinsic
        total = intrinsic_total + context

        if return_parts:
            return {
                "total": total,
                "intrinsic": intrinsic_total,
                "context": context,
            }
        return total

__all__ = [
    "DEFAULT_FDEWET_MU",
    "EvoLayer",
    "FdewetMPNN",
    "NODE_IN",
    "EDGE_DIM",
    "HIDDEN",
    "DEPTH",
    "DROPOUT",
    "EDGE_DROP",
    "HEAD_HIDDEN",
]
