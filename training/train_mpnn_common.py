#!/usr/bin/env python3
"""Shared utilities for the canonical FastHydroMap MPNN training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import pearsonr, spearmanr
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from FastHydroMap.models.mpnn import FdewetMPNN, NODE_IN
from residue_keys import ensure_residue_key_columns

TRAINING_DIR = Path(__file__).resolve().parent
ROOT = TRAINING_DIR / "data"
GRAPH_DIR = ROOT / "graphs"
META_CSV = ROOT / "all_residue_results.csv"
SPLIT_YML = ROOT / "splits.yaml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class TrainConfig:
    k_nn: int = 12
    n_rbf: int = 3
    rbf_min: float = 2.0
    rbf_max: float = 14.0
    rbf_sigma: float = 4.0
    pair_set: str = "full"
    hidden: int = 24
    depth: int = 2
    head_hidden: int = 20
    dropout: float = 0.1
    edge_drop: float = 0.1
    weight_decay: float = 1e-3
    warmup: int = 300
    batch_size: int = 32
    epochs: int = 50
    patience: int = 20
    clip: float = 2.0
    factor: float = 1.0
    seed: int = 48


def _flt_tag(x: float) -> str:
    return str(x).replace("-", "m").replace(".", "p")


def graph_cache_paths(cfg: TrainConfig) -> tuple[Path, Path]:
    suffix = ""
    if cfg.pair_set != "full":
        suffix += f"_{cfg.pair_set}"
    if cfg.n_rbf != 16:
        suffix += f"_rbf{cfg.n_rbf}"
    if cfg.rbf_min != 2.0 or cfg.rbf_max != 20.0 or cfg.rbf_sigma != 2.0:
        suffix += f"_r{_flt_tag(cfg.rbf_min)}to{_flt_tag(cfg.rbf_max)}_s{_flt_tag(cfg.rbf_sigma)}"
    return (
        GRAPH_DIR / f"graphs_k{cfg.k_nn}{suffix}.pt",
        GRAPH_DIR / f"pdb_ids_k{cfg.k_nn}{suffix}.pt",
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def masked_mean_target(meta: pd.DataFrame, pdb_ids: list[str]) -> float:
    subset = meta[meta["pdb_id"].isin(pdb_ids)]
    trusted = (
        (subset.avg_n_waters.values > 7.0)
        & (subset.Fdewet_pred.values >= 3.8)
        & (subset.Fdewet_pred.values <= 8.7)
    )
    return float(subset.loc[trusted, "Fdewet_pred"].mean())


class GraphDSDirect(InMemoryDataset):
    def __init__(self, graphs_pt: Path, meta_df: pd.DataFrame):
        super().__init__("")
        self.data, self.slices = torch.load(graphs_pt)
        self._augment_with_meta(meta_df)

    def _augment_with_meta(self, meta: pd.DataFrame):
        meta = ensure_residue_key_columns(meta)
        mi_uid = meta.set_index(["pdb_id", "res_uid"])
        mi_resid = meta.set_index(["pdb_id", "resid"])
        new_graphs = []
        for i in range(self.len()):
            g = self.get(i)
            if hasattr(g, "res_uid"):
                rows = mi_uid.loc[g.pdb_id].reindex(list(g.res_uid))
            else:
                rows = mi_resid.loc[g.pdb_id].reindex(g.resid.tolist())

            if rows.isnull().any().any():
                raise ValueError(f"missing metadata for graph {g.pdb_id}")

            target = rows.Fdewet_pred.values.astype(np.float32)
            trusted = (
                (rows.avg_n_waters.values > 7.0)
                & (rows.Fdewet_pred.values >= 3.8)
                & (rows.Fdewet_pred.values <= 8.7)
            ).astype(np.bool_)

            g.target = torch.from_numpy(target)
            g.mask = torch.from_numpy(trusted)
            new_graphs.append(g)

        self.data, self.slices = InMemoryDataset.collate(new_graphs)


class NoamOpt:
    def __init__(self, model_size: int, factor: float, warmup: int, optimizer):
        self.optimizer = optimizer
        self.factor = factor
        self.warmup = warmup
        self.model_size = model_size
        self._step = 0

    def step(self):
        self._step += 1
        lr = self.rate()
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.optimizer.step()

    def rate(self, step: int | None = None):
        step = self._step if step is None else step
        return self.factor * (self.model_size ** -0.5) * min(step ** -0.5, step * self.warmup ** -1.5)

    def zero_grad(self):
        self.optimizer.zero_grad()


def load_meta_and_splits():
    meta = pd.read_csv(META_CSV)
    splits = yaml.safe_load(open(SPLIT_YML))
    return meta, splits


def split_indices(pids_pt: Path, split_ids: dict[str, list[str]]):
    pids = torch.load(pids_pt)
    id2ix = {p: i for i, p in enumerate(pids)}
    out = {k: [id2ix[p] for p in ids if p in id2ix] for k, ids in split_ids.items()}
    kept_ids = {k: [p for p in ids if p in id2ix] for k, ids in split_ids.items()}
    return out, kept_ids


def make_model(cfg: TrainConfig, mu: float, edge_dim: int):
    return FdewetMPNN(
        node_in=NODE_IN,
        edge_dim=edge_dim,
        hidden=cfg.hidden,
        depth=cfg.depth,
        mu=mu,
        dropout=cfg.dropout,
        edge_drop=cfg.edge_drop,
        head_hidden=cfg.head_hidden,
    ).to(DEVICE)


def edge_dim_from_cfg(cfg: TrainConfig) -> int:
    pair_count = 25 if cfg.pair_set == "full" else (4 if cfg.pair_set == "ca_cb" else 1)
    return pair_count * cfg.n_rbf + 9


def make_optimizer(model, cfg: TrainConfig):
    base_opt = torch.optim.AdamW(
        model.parameters(),
        lr=0,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=cfg.weight_decay,
    )
    return NoamOpt(cfg.hidden, cfg.factor, cfg.warmup, base_opt)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, truths = [], []
    intrinsic_vals, context_vals = [], []
    for b in loader:
        b = b.to(DEVICE)
        parts = model(b, return_parts=True)
        m = b.mask.cpu().bool()
        preds.append(parts["total"].cpu()[m])
        truths.append(b.target.cpu()[m])
        intrinsic_vals.append(parts["intrinsic"].cpu()[m])
        context_vals.append(parts["context"].cpu()[m])

    p = torch.cat(preds)
    t = torch.cat(truths)
    intrinsic = torch.cat(intrinsic_vals)
    context = torch.cat(context_vals)

    rmse = F.mse_loss(p, t).sqrt().item()
    r = pearsonr(t, p)[0]
    rho = spearmanr(t, p).correlation
    stats = {
        "intrinsic_mean": intrinsic.mean().item(),
        "intrinsic_std": intrinsic.std(unbiased=False).item(),
        "context_mean": context.mean().item(),
        "context_std": context.std(unbiased=False).item(),
        "intrinsic_abs_mean": intrinsic.abs().mean().item(),
        "context_abs_mean": context.abs().mean().item(),
    }
    return rmse, r, rho, stats


def train_one_epoch(model, loader, opt: NoamOpt, cfg: TrainConfig):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch)
        loss = F.mse_loss(pred[batch.mask], batch.target[batch.mask])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
