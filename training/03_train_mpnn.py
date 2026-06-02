#!/usr/bin/env python3
"""Train FastHydroMap direct MPNN regressors."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from train_mpnn_common import (
    DEVICE,
    GraphDSDirect,
    SUPPORTED_MASK_SOURCES,
    SUPPORTED_TARGETS,
    TrainConfig,
    compute_winsor_bounds,
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
PKG_WEIGHT_DIR = ROOT.parent / "src" / "FastHydroMap" / "weights"
CKPT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
PKG_WEIGHT_DIR.mkdir(exist_ok=True)

TARGET_PACKAGE_WEIGHTS = {
    "Fdewet_pred": "mpnn_latest.pt",
    "PC1": "mpnn_pc1_latest.pt",
    "PC2": "mpnn_pc2_latest.pt",
    "PC3": "mpnn_pc3_latest.pt",
}


def _tag_float(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return str(x).replace("-", "m").replace(".", "p")


def model_tag(cfg: TrainConfig) -> str:
    effective_mask = cfg.mask_source
    if effective_mask == "auto":
        effective_mask = "fdewet" if cfg.target == "Fdewet_pred" else "trusted"

    graph = (
        f"k{cfg.k_nn}_rbf{cfg.n_rbf}_"
        f"r{_tag_float(cfg.rbf_min)}to{_tag_float(cfg.rbf_max)}_"
        f"s{_tag_float(cfg.rbf_sigma)}"
    )
    model = f"{graph}_h{cfg.hidden}_d{cfg.depth}_head{cfg.head_hidden}"

    if cfg.target == "Fdewet_pred" and effective_mask == "fdewet":
        tag = model
    else:
        tag = f"{cfg.target.lower()}_{effective_mask}_{model}"

    if cfg.winsor_lower is not None:
        tag = f"{tag}_winsor_p{int(cfg.winsor_lower * 100)}_p{int(cfg.winsor_upper * 100)}"
    return tag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("val", "prod"), default="val")
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
    p.add_argument("--winsor-lower", type=float, default=None)
    p.add_argument("--winsor-upper", type=float, default=None)
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="production epochs; if omitted, use best_epoch from the validation summary",
    )
    p.add_argument("--report-test", action="store_true", help="evaluate the held-out test split")
    p.add_argument(
        "--copy-to-package",
        action="store_true",
        help="for production training, copy the resulting weight to src/FastHydroMap/weights",
    )
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
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
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
    )


def config_summary(cfg: TrainConfig) -> dict[str, float | int | str | None]:
    return {
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
        "winsor_lower": cfg.winsor_lower,
        "winsor_upper": cfg.winsor_upper,
    }


def load_training_inputs(cfg: TrainConfig):
    graph_pt, pid_pt = graph_cache_paths(cfg)
    if not graph_pt.exists() or not pid_pt.exists():
        raise FileNotFoundError(
            f"missing graph cache: {graph_pt.name} / {pid_pt.name}. "
            "Build it first with 02_build_mpnn_graphs.py using matching graph settings."
        )
    meta, splits = load_meta_and_splits()
    return graph_pt, pid_pt, meta, splits


def make_dataset_and_loaders(
    cfg: TrainConfig,
    split_ids: dict[str, list[str]],
    *,
    clip_from_split: str,
    shuffle_split: str,
):
    graph_pt, pid_pt, meta, _ = load_training_inputs(cfg)
    split_ix, split_ids_kept = split_indices(pid_pt, split_ids)
    target_clip_bounds = compute_winsor_bounds(
        meta,
        split_ids_kept[clip_from_split],
        cfg.target,
        cfg.mask_source,
        cfg.winsor_lower,
        cfg.winsor_upper,
    )
    mu = masked_mean_target(
        meta,
        split_ids_kept[clip_from_split],
        cfg.target,
        cfg.mask_source,
        target_clip_bounds,
    )
    ds = GraphDSDirect(
        graph_pt,
        meta,
        target=cfg.target,
        mask_source=cfg.mask_source,
        target_clip_bounds=target_clip_bounds,
    )
    loaders = {}
    for name, indices in split_ix.items():
        loaders[name] = DataLoader(
            ds[indices],
            cfg.batch_size,
            shuffle=(name == shuffle_split),
            num_workers=4,
        )
    return loaders, graph_pt, mu, target_clip_bounds


def base_summary(
    cfg: TrainConfig,
    *,
    stage: str,
    tag: str,
    graph_pt: Path,
    mu: float,
    target_clip_bounds: tuple[float, float] | None,
) -> dict:
    return {
        "stage": stage,
        "seed": cfg.seed,
        "target": cfg.target,
        "mask_source": cfg.mask_source,
        "tag": tag,
        "mu": float(mu),
        "edge_dim": edge_dim_from_cfg(cfg),
        "target_clip_bounds": list(target_clip_bounds) if target_clip_bounds is not None else None,
        "config": config_summary(cfg),
        "graph_pt": str(graph_pt),
    }


def run_validation(cfg: TrainConfig, *, report_test: bool) -> None:
    _, _, _, splits = load_training_inputs(cfg)
    split_ids = {"train": splits["train"], "val": splits["val"], "test": splits["test"]}
    loaders, graph_pt, mu, target_clip_bounds = make_dataset_and_loaders(
        cfg,
        split_ids,
        clip_from_split="train",
        shuffle_split="train",
    )

    tag = model_tag(cfg)
    best_pt = CKPT_DIR / f"best_{tag}_seed{cfg.seed}.pt"
    summary_json = MODEL_DIR / f"03_train_mpnn_val_{tag}_seed{cfg.seed}.json"

    model = make_model(cfg, mu=mu, edge_dim=edge_dim_from_cfg(cfg))
    opt = make_optimizer(model, cfg)

    best_val = float("inf")
    best_epoch = 0
    wait = 0
    for epoch in range(1, cfg.epochs + 1):
        train_one_epoch(model, loaders["train"], opt, cfg)
        val_rmse, val_r, val_rho, _ = evaluate(model, loaders["val"])
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
    val_rmse, val_r, val_rho, val_stats = evaluate(model, loaders["val"])
    summary = base_summary(
        cfg,
        stage="val",
        tag=tag,
        graph_pt=graph_pt,
        mu=mu,
        target_clip_bounds=target_clip_bounds,
    )
    summary.update(
        {
            "best_epoch": best_epoch,
            "best_val_rmse": float(best_val),
            "val_rmse_reloaded": float(val_rmse),
            "val_r": float(val_r),
            "val_rho": float(val_rho),
            "best_ckpt": str(best_pt),
            "val_stats": {k: float(v) for k, v in val_stats.items()},
        }
    )

    if report_test:
        test_rmse, test_r, test_rho, test_stats = evaluate(model, loaders["test"])
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


def run_production(
    cfg: TrainConfig,
    *,
    epochs: int | None,
    report_test: bool,
    copy_to_package: bool,
) -> None:
    _, _, _, splits = load_training_inputs(cfg)
    split_ids = {
        "trainval": [*splits["train"], *splits["val"]],
        "test": splits["test"],
    }
    loaders, graph_pt, mu, target_clip_bounds = make_dataset_and_loaders(
        cfg,
        split_ids,
        clip_from_split="trainval",
        shuffle_split="trainval",
    )

    tag = model_tag(cfg)
    if epochs is None:
        val_summary = MODEL_DIR / f"03_train_mpnn_val_{tag}_seed{cfg.seed}.json"
        if not val_summary.exists():
            raise FileNotFoundError(f"missing validation summary for default epochs: {val_summary}")
        epochs = int(json.loads(val_summary.read_text())["best_epoch"])

    model = make_model(cfg, mu=mu, edge_dim=edge_dim_from_cfg(cfg))
    opt = make_optimizer(model, cfg)
    local_pt = MODEL_DIR / f"mpnn_direct_prod_{tag}.pt"
    summary_json = MODEL_DIR / f"03_train_mpnn_prod_{tag}_seed{cfg.seed}.json"

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, loaders["trainval"], opt, cfg)
        print(f"[prod] epoch {epoch:02d}/{epochs}", flush=True)

    torch.save(model.state_dict(), local_pt)

    package_weight = None
    if copy_to_package:
        package_weight = PKG_WEIGHT_DIR / TARGET_PACKAGE_WEIGHTS[cfg.target]
        shutil.copy2(local_pt, package_weight)

    summary = base_summary(
        cfg,
        stage="prod",
        tag=tag,
        graph_pt=graph_pt,
        mu=mu,
        target_clip_bounds=target_clip_bounds,
    )
    summary.update(
        {
            "epochs": epochs,
            "local_weight": str(local_pt),
            "package_weight": str(package_weight) if package_weight is not None else None,
        }
    )

    if report_test:
        test_rmse, test_r, test_rho, test_stats = evaluate(model, loaders["test"])
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
    if package_weight is not None:
        print(f"[done] package weight -> {package_weight}", flush=True)
    print(f"[done] summary -> {summary_json}", flush=True)


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    set_seed(cfg.seed)
    if args.stage == "val":
        run_validation(cfg, report_test=args.report_test)
    else:
        run_production(
            cfg,
            epochs=args.epochs,
            report_test=args.report_test,
            copy_to_package=args.copy_to_package,
        )


if __name__ == "__main__":
    main()
