"""Selected v2 architecture, with checkpoint-compatible parameter names."""

import torch
from torch import nn
from .layers import PredictionBase, ContinuousLayer, selected_graph


class SurfaceModel(PredictionBase):
    def __init__(self):
        super().__init__(32, 0.0, 0.1, 24)
        self.node_embedding = nn.Linear(141, 32)
        self.layers = nn.ModuleList([ContinuousLayer(32, 20, 0.1) for _ in range(2)])
        self.encoder = nn.Sequential(
            nn.Linear(32, 24), nn.SiLU(), nn.Linear(24, 24), nn.SiLU()
        )
        self.fusion = nn.Linear(48, 32, bias=False)
        self.atom_encoder = nn.Sequential(
            nn.Linear(21, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU()
        )
        self.atom_fusion = nn.Sequential(nn.Linear(32, 24), nn.SiLU())
        # Retained for exact checkpoint loading; these auxiliary heads are not outputs.
        self.component_head = nn.Linear(32, 3)
        self.register_buffer("component_mean", torch.zeros(3))
        self.register_buffer("component_std", torch.ones(3))

    def forward(self, g, return_parts=False):
        x = torch.cat(
            [g.x]
            + [
                getattr(g, "rich_" + k)
                for k in ("cavity", "angular", "field", "atom_env")
            ],
            -1,
        )
        h0 = self.node_embedding(x)
        mask = g.point_mask
        p = g.point.clone()
        p[:, :, 14:] = 0
        p *= mask[:, :, None]
        a = self.atom_encoder(g.probe_atom)
        chemical = self.atom_fusion(torch.cat((a.mean(2), a.amax(2)), -1))
        ph = (self.encoder(p) + chemical) * mask[:, :, None]
        mean = (ph * mask[:, :, None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        maximum = ph.masked_fill(~mask[:, :, None], -1e9).amax(1) * mask.any(1)[:, None]
        h0 = h0 + self.fusion(torch.cat((mean, maximum), -1))
        h = h0
        index, edge = selected_graph(g, 12, "functional", 8)
        for layer in self.layers:
            h = layer(h, index, edge)
        return self.finish(h0, h, g, return_parts)
