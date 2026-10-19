"""Shared residue encoder with checkpoint-compatible parameter names."""

import torch
from torch import nn
from .layers import ContinuousLayer


class SharedWaterModel(nn.Module):
    """One local 16D residue encoder and four prediction heads (8,740 weights).

    Embeddings belong to a particular checkpoint. Never average their coordinates
    across independently trained ensemble members.
    """
    def __init__(self):
        super().__init__()
        self.width = 16
        self.node_embedding = nn.Linear(35, 16)
        self.encoder = nn.Sequential(nn.Linear(11, 24), nn.SiLU(), nn.Linear(24, 24), nn.SiLU())
        self.atom_encoder = nn.Sequential(nn.Linear(19, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU())
        self.atom_fusion = nn.Sequential(nn.Linear(32, 24), nn.SiLU())
        self.fusion = nn.Linear(48, 16, bias=False)
        self.layers = nn.ModuleList([ContinuousLayer(16, 20, .1) for _ in range(2)])
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(16, 24), nn.GELU(),
            nn.Dropout(.1), nn.Linear(24, 1)) for _ in range(4)])

    @staticmethod
    def inputs(g):
        x = torch.cat([g.x] + [getattr(g, 'rich_' + k)
            for k in ('cavity', 'angular', 'field', 'atom_env')], -1)
        return dict(node=x[:, [*range(34), 56]], point=g.point[..., :11],
            point_mask=g.point_mask, atom=g.probe_atom[..., [i for i in range(21) if i not in (2, 3)]],
            edge_index=g.graph_edge_index, edge=g.graph_edge_attr)

    def encode(self, g):
        mask = g['point_mask']
        a = self.atom_encoder(g['atom'])
        chemistry = self.atom_fusion(torch.cat((a.mean(2), a.amax(2)), -1))
        ph = (self.encoder(g['point'] * mask[..., None]) + chemistry) * mask[..., None]
        mean = ph.sum(1) / mask.sum(1).clamp_min(1)[:, None]
        maximum = ph.masked_fill(~mask[..., None], -1e9).amax(1) * mask.any(1)[:, None]
        h = self.node_embedding(g['node']) + self.fusion(torch.cat((mean, maximum), -1))
        for layer in self.layers:
            h = layer(h, g['edge_index'], g['edge'])
        return h

    def readout(self, embedding):
        return torch.cat([head(embedding) for head in self.heads], -1)

    def forward(self, g, return_embeddings=False):
        h = self.encode(g)
        prediction = self.readout(h)
        return (prediction, h) if return_embeddings else prediction
