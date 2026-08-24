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
    SUPPORTED_MASK_SOURCES,
    SUPPORTED_TARGETS,
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
BASE_TAG = "k12_rbf3_r2to14_s4_h24_d2_head20"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=48)
    p.add_argument("--target", choices=SUPPORTED_TARGETS, default="Fdewet_pred")
    p.add_argument("--mask-source", choices=SUPPORTED_MASK_SOURCES, default="auto")
    p.add_argument("--k-nn", type=int, default=TrainConfig.k_nn)
    p.add_argument("--n-rbf", type=int, default=TrainConfig.n_rbf)
    p.add_argument("--rbf-min", type=float, default=TrainConfig.rbf_min)
    p.add_argument("--rbf-max", type=float, default=TrainConfig.rbf_max)
    p.add_argument("--rbf-sigma", type=float, default=TrainConfig.rbf_sigma)
    p.add_argument("--hidden", type=int, default=TrainConfig.hidden)
    p.add_argument("--depth", type=int, default=TrainConfig.depth)
    p.add_argument("--head-hidden", type=int, default=TrainConfig.head_hidden)
    p.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    p.add_argument("--edge-drop", type=float, default=TrainConfig.edge_drop)
    p.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--epochs", type=int, default=None, help="if omitted, use best_epoch from val summary")
    p.add_argument("--report-test", action="store_true", help="evaluate held-out test split at end")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        seed=args.seed,
        target=args.target,
        mask_source=args.mask_source,
        k_nn=args.k_nn,
        n_rbf=args.n_rbf,
        rbf_min=args.rbf_min,
        rbf_max=args.rbf_max,
        rbf_sigma=args.rbf_sigma,
        hidden=args.hidden,
        depth=args.depth,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        edge_drop=args.edge_drop,
        weight_decay=args.weight_decay,
    )
    set_seed(cfg.seed)

    graph_pt, pid_pt = graph_cache_paths(cfg)
    if not graph_pt.exists() or not pid_pt.exists():
        raise FileNotFoundError(
            f"missing graph cache: {graph_pt.name} / {pid_pt.name}. "
            "build it first with 02_build_mpnn_graphs.py using matching settings."
        )

    graph_tag = graph_pt.stem.removeprefix("graphs_")
    model_tag = f"{graph_tag}_h{cfg.hidden}_d{cfg.depth}_head{cfg.head_hidden}"
    effective_mask = cfg.mask_source if cfg.mask_source != "auto" else ("fdewet" if cfg.target == "Fdewet_pred" else "trusted")
    if cfg.target == "Fdewet_pred" and effective_mask == "fdewet" and model_tag == BASE_TAG:
        tag = BASE_TAG
    else:
        tag = f"{cfg.target.lower()}_{effective_mask}_{model_tag}"

    epochs = args.epochs
    if epochs is None:
        val_summary = MODEL_DIR / f"03_train_mpnn_val_{tag}_seed{cfg.seed}.json"
        if not val_summary.exists():
            raise FileNotFoundError(f"missing val summary for default epochs: {val_summary}")
        epochs = int(json.loads(val_summary.read_text())["best_epoch"])

    meta, splits = load_meta_and_splits()
    split_ids = {
        "trainval": [*splits["train"], *splits["val"]],
        "test": splits["test"],
    }
    split_ix, split_ids_kept = split_indices(pid_pt, split_ids)
    mu = masked_mean_target(meta, split_ids_kept["trainval"], cfg.target, cfg.mask_source)
    edge_dim = edge_dim_from_cfg(cfg)

    ds = GraphDSDirect(graph_pt, meta, target=cfg.target, mask_source=cfg.mask_source)
    trL = DataLoader(ds[split_ix["trainval"]], cfg.batch_size, shuffle=True, num_workers=4)
    teL = DataLoader(ds[split_ix["test"]], cfg.batch_size, shuffle=False, num_workers=4)

    model = make_model(cfg, mu=mu, edge_dim=edge_dim)
    opt = make_optimizer(model, cfg)

    local_pt = MODEL_DIR / f"mpnn_direct_prod_{tag}.pt"
    pkg_pt = PKG_WEIGHT_DIR / f"mpnn_direct_prod_{tag}.pt" if tag == BASE_TAG else None
    summary_json = MODEL_DIR / f"04_train_mpnn_prod_{tag}_seed{cfg.seed}.json"

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, trL, opt, cfg)
        print(f"[prod] epoch {epoch:02d}/{epochs}", flush=True)

    torch.save(model.state_dict(), local_pt)
    if pkg_pt is not None:
        shutil.copy2(local_pt, pkg_pt)

    summary = {
        "seed": cfg.seed,
        "target": cfg.target,
        "mask_source": cfg.mask_source,
        "epochs": epochs,
        "mu": float(mu),
        "edge_dim": edge_dim,
        "config": {
            "k_nn": cfg.k_nn,
            "n_rbf": cfg.n_rbf,
            "rbf_min": cfg.rbf_min,
            "rbf_max": cfg.rbf_max,
            "rbf_sigma": cfg.rbf_sigma,
            "hidden": cfg.hidden,
            "depth": cfg.depth,
            "head_hidden": cfg.head_hidden,
            "dropout": cfg.dropout,
            "edge_drop": cfg.edge_drop,
            "weight_decay": cfg.weight_decay,
        },
        "local_weight": str(local_pt),
        "package_weight": str(pkg_pt) if pkg_pt is not None else None,
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
    if pkg_pt is not None:
        print(f"[done] package weight -> {pkg_pt}", flush=True)
    print(f"[done] summary -> {summary_json}", flush=True)


if __name__ == "__main__":
    main()
