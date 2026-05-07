#!/usr/bin/env python3
"""
Train FastHydroMap direct MPNN on train+val for production weights.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from train_mpnn_common import (
    GraphDSDirect,
    TrainConfig,
    edge_dim_from_cfg,
    evaluate,
    graph_cache_paths,
    load_meta_and_splits,
    make_model,
    make_optimizer,
    masked_mean_target,
    set_seed,
    split_indices,
    train_one_epoch,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
PKG_WEIGHT_DIR = ROOT.parent / "src" / "FastHydroMap" / "weights"
MODEL_DIR.mkdir(exist_ok=True)
PKG_WEIGHT_DIR.mkdir(exist_ok=True)
VAL_SUMMARY = MODEL_DIR / "03_train_mpnn_val_k12_rbf3_r2to14_s4_h24_d2_head20_seed48.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=48)
    p.add_argument("--epochs", type=int, default=None, help="if omitted, use best_epoch from val summary")
    p.add_argument("--report-test", action="store_true", help="evaluate held-out test split at end")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(seed=args.seed)
    set_seed(cfg.seed)

    graph_pt, pid_pt = graph_cache_paths(cfg)
    if not graph_pt.exists() or not pid_pt.exists():
        raise FileNotFoundError(
            f"missing graph cache: {graph_pt.name} / {pid_pt.name}. "
            "build it first with 02_build_mpnn_graphs.py using matching settings."
        )

    epochs = args.epochs
    if epochs is None:
        if not VAL_SUMMARY.exists():
            raise FileNotFoundError(f"missing val summary for default epochs: {VAL_SUMMARY}")
        epochs = int(json.loads(VAL_SUMMARY.read_text())["best_epoch"])

    meta, splits = load_meta_and_splits()
    split_ids = {
        "trainval": [*splits["train"], *splits["val"]],
        "test": splits["test"],
    }
    split_ix, split_ids_kept = split_indices(pid_pt, split_ids)
    mu = masked_mean_target(meta, split_ids_kept["trainval"])
    edge_dim = edge_dim_from_cfg(cfg)

    ds = GraphDSDirect(graph_pt, meta)
    trL = DataLoader(ds[split_ix["trainval"]], cfg.batch_size, shuffle=True, num_workers=4)
    teL = DataLoader(ds[split_ix["test"]], cfg.batch_size, shuffle=False, num_workers=4)

    model = make_model(cfg, mu=mu, edge_dim=edge_dim)
    opt = make_optimizer(model, cfg)

    tag = "k12_rbf3_r2to14_s4_h24_d2_head20"
    local_pt = MODEL_DIR / f"mpnn_direct_prod_{tag}.pt"
    pkg_pt = PKG_WEIGHT_DIR / f"mpnn_direct_prod_{tag}.pt"
    summary_json = MODEL_DIR / f"04_train_mpnn_prod_{tag}_seed{cfg.seed}.json"

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, trL, opt, cfg)
        print(f"[prod] epoch {epoch:02d}/{epochs}", flush=True)

    torch.save(model.state_dict(), local_pt)
    shutil.copy2(local_pt, pkg_pt)

    summary = {
        "seed": cfg.seed,
        "epochs": epochs,
        "mu": float(mu),
        "edge_dim": edge_dim,
        "local_weight": str(local_pt),
        "package_weight": str(pkg_pt),
        "graph_pt": str(graph_pt),
    }

    if args.report_test:
        test_rmse, test_r, test_rho, test_stats = evaluate(model, teL)
        summary.update(
            {
                "test_rmse": float(test_rmse),
                "test_r": float(test_r),
                "test_rho": float(test_rho),
                "test_stats": {k: float(v) for k, v in test_stats.items()},
            }
        )
        print(f"[test] rmse {test_rmse:.4f} | r {test_r:.3f} | rho {test_rho:.3f}", flush=True)

    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] local weight -> {local_pt}", flush=True)
    print(f"[done] package weight -> {pkg_pt}", flush=True)
    print(f"[done] summary -> {summary_json}", flush=True)


if __name__ == "__main__":
    main()
