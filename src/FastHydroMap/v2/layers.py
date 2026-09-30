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


def graph_membership(graph, node_count: int) -> tuple[torch.Tensor, int]:
    if hasattr(graph, "batch") and graph.batch is not None:
        return graph.batch, int(graph.num_graphs)
    return torch.zeros(node_count, dtype=torch.long, device=graph.x.device), 1


def radial_basis(distance: torch.Tensor, bins: int) -> torch.Tensor:
    centers = torch.linspace(
        2.0, 14.0, bins, device=distance.device, dtype=distance.dtype
    )
    sigma = 2.0 if bins <= 8 else 1.0
    return torch.exp(-0.5 * ((distance[:, None] - centers[None]) / sigma) ** 2)


def selected_graph(
    graph,
    neighbors: int,
    edge_kind: str | None,
    rbf_bins: int,
    chemistry_gate: torch.Tensor | None = None,
):
    keep = graph.pf_edge_rank <= neighbors
    edge_index = graph.pf_edge_index[:, keep]
    edge = radial_basis(graph.pf_edge_distance[keep], rbf_bins)
    if edge_kind == "centroid_functional":
        edge = torch.cat(
            (
                edge,
                graph.pf_centroid_edge[keep],
                graph.pf_functional_edge[keep],
            ),
            dim=-1,
        )
    elif edge_kind == "chemistry_gated":
        if chemistry_gate is None:
            raise ValueError("chemistry-gated edge requested without learned gates")
        target = edge_index[1]
        category = graph.aa_code.long().clone()
        category[graph.histidine_class == 2] = 20
        weight = torch.sigmoid(chemistry_gate[category[target]])[:, None]
        sidechain = (1.0 - weight) * graph.pf_centroid_edge[
            keep
        ] + weight * graph.pf_functional_edge[keep]
        edge = torch.cat((edge, sidechain), dim=-1)
    elif edge_kind in {"centroid_planar", "functional_planar"}:
        base_kind = edge_kind.split("_", 1)[0]
        edge = torch.cat(
            (
                edge,
                getattr(graph, f"pf_{base_kind}_edge")[keep],
                graph.pf_planar_edge[keep],
            ),
            dim=-1,
        )
    elif edge_kind in {"centroid_polar", "functional_polar"}:
        base_kind = edge_kind.split("_", 1)[0]
        edge = torch.cat(
            (
                edge,
                getattr(graph, f"pf_{base_kind}_edge")[keep],
                graph.pf_polar_edge[keep],
            ),
            dim=-1,
        )
    elif edge_kind is not None:
        edge = torch.cat((edge, getattr(graph, f"pf_{edge_kind}_edge")[keep]), dim=-1)
    return edge_index, edge


def scalar_head(input_dim: int, hidden: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


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


class PredictionBase(nn.Module):
    def __init__(self, hidden: int, mu: float, dropout: float, head_hidden: int):
        super().__init__()
        self.register_buffer("mu", torch.tensor(float(mu), dtype=torch.float32))
        self.intrinsic_head = scalar_head(hidden, head_hidden, dropout)
        self.context_head = scalar_head(2 * hidden, head_hidden, dropout)

    def finish(self, h0, h, graph, return_parts: bool):
        membership, graph_count = graph_membership(graph, len(h))
        pooled = scatter_mean(h, membership, graph_count)
        intrinsic = self.mu + self.intrinsic_head(h0).squeeze(-1)
        context = self.context_head(torch.cat((h, pooled[membership]), dim=-1)).squeeze(
            -1
        )
        total = intrinsic + context
        if return_parts:
            return {"total": total, "intrinsic": intrinsic, "context": context}
        return total
