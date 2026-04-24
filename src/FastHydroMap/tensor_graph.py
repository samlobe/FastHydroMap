from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TensorGraph:
    """Minimal tensor container for FastHydroMap inference graphs."""

    x: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor
    resid: torch.Tensor
    pdb_id: torch.Tensor

    def to(self, device: str | torch.device) -> "TensorGraph":
        return TensorGraph(
            x=self.x.to(device),
            pos=self.pos.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            y=self.y.to(device),
            resid=self.resid.to(device),
            pdb_id=self.pdb_id.to(device),
        )
