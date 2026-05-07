#!/usr/bin/env python3
"""
Train FastHydroMap direct MPNN on train split, early-stop on val split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from train_mpnn_common import (
    DEVICE,
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
CKPT_DIR = ROOT / "checkpoints_direct_feat"
MODEL_DIR = ROOT / "models"
CKPT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=48)
    p.add_argument("--report-test", action="store_true", help="evaluate held-out test split after training")
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

    meta, splits = load_meta_and_splits()
    split_ids = {"train": splits["train"], "val": splits["val"], "test": splits["test"]}
    split_ix, split_ids_kept = split_indices(pid_pt, split_ids)
    mu = masked_mean_target(meta, split_ids_kept["train"])
    edge_dim = edge_dim_from_cfg(cfg)

    ds = GraphDSDirect(graph_pt, meta)
    trL = DataLoader(ds[split_ix["train"]], cfg.batch_size, shuffle=True, num_workers=4)
    vaL = DataLoader(ds[split_ix["val"]], cfg.batch_size, shuffle=False, num_workers=4)
    teL = DataLoader(ds[split_ix["test"]], cfg.batch_size, shuffle=False, num_workers=4)

    model = make_model(cfg, mu=mu, edge_dim=edge_dim)
    opt = make_optimizer(model, cfg)

    tag = "k12_rbf3_r2to14_s4_h24_d2_head20"
    best_pt = CKPT_DIR / f"best_{tag}_seed{cfg.seed}.pt"
    summary_json = MODEL_DIR / f"03_train_mpnn_val_{tag}_seed{cfg.seed}.json"

    best_val = float("inf")
    best_epoch = 0
    wait = 0
    for epoch in range(1, cfg.epochs + 1):
        train_one_epoch(model, trL, opt, cfg)
        val_rmse, val_r, val_rho, _ = evaluate(model, vaL)
        print(f"[val] epoch {epoch:02d} rmse {val_rmse:.4f} | r {val_r:.3f} | rho {val_rho:.3f}", flush=True)
        if val_rmse + 1e-4 < best_val:
            best_val = val_rmse
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), best_pt)
            print(f"[val] new best -> {best_pt.name}", flush=True)
        else:
            wait += 1
            if wait >= cfg.patience:
                print("[val] early stop", flush=True)
                break

    model.load_state_dict(torch.load(best_pt, map_location=DEVICE))
    val_rmse, val_r, val_rho, val_stats = evaluate(model, vaL)
    summary = {
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "best_val_rmse": float(best_val),
        "val_rmse_reloaded": float(val_rmse),
        "val_r": float(val_r),
        "val_rho": float(val_rho),
        "mu": float(mu),
        "edge_dim": edge_dim,
        "graph_pt": str(graph_pt),
        "best_ckpt": str(best_pt),
        "val_stats": {k: float(v) for k, v in val_stats.items()},
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
    print(f"[done] best val rmse {best_val:.4f} at epoch {best_epoch}", flush=True)
    print(f"[done] summary -> {summary_json}", flush=True)


if __name__ == "__main__":
    main()
