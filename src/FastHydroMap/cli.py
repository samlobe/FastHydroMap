# FastHydroMap/cli.py
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .io.pdb            import write_bfactor
from .install_torch import install_torch, torch_install_command

QUANTITIES = ("fdewet", "pc1", "pc2", "pc3", "all")
QUANTITY_LABELS = {
    "fdewet": "Fdewet",
    "pc1": "PC1",
    "pc2": "PC2",
    "pc3": "PC3",
}


def _model_name(value):
    # Accept the unpublished preview spellings without advertising them.
    return {"v1": "v0", "v2": "latest"}.get(value, value)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fasthydromap",
                                 description="FastHydroMap – infer per-residue Fdewet or PC maps")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # -------- predict -------------------------------------------------
    p = sub.add_parser("predict", help="run inference on PDB (± DCD trajectory)")
    p.add_argument("pdb", type=Path, help="input PDB topology")
    p.add_argument("dcd", nargs="?", type=Path, default=None,
                help="optional trajectory (DCD/XTC); if omitted runs single-structure mode")
    p.add_argument("-o", "--outroot", type=Path,
                help="basename for outputs (latest: <pdb>_maps or <trajectory>_maps_traj; v0: <pdb>_<quantity>)",
                default=None)
    p.add_argument(
        "--parts",
        action="store_true",
        help="with --model v0, include intrinsic and context columns",
    )
    p.add_argument(
        "--quantity",
        choices=QUANTITIES,
        default=None,
        help="legacy v0 map to export (default: fdewet); latest always exports all four",
    )

    # -------- predict-trajectory --------------------------------------
    pt = sub.add_parser(
        "predict-trajectory",
        help="run trajectory inference",
    )
    pt.add_argument("pdb", type=Path, help="input PDB topology")
    pt.add_argument("dcd", type=Path, help="input trajectory (DCD/XTC)")
    pt.add_argument(
        "-o",
        "--outroot",
        type=Path,
        default=None,
        help="basename for outputs (latest: <trajectory>_maps_traj; v0: <trajectory>_<quantity>_traj)",
    )
    pt.add_argument(
        "--parts",
        action="store_true",
        help="with --model v0, also write intrinsic/context and per-frame summaries",
    )
    pt.add_argument(
        "--quantity",
        choices=QUANTITIES,
        default=None,
        help="legacy v0 map to export (default: fdewet); latest always exports all four",
    )

    for command in (p, pt):
        command.add_argument("--model", type=_model_name, choices=("latest", "v0"), default="latest",
                             help="bundled model (default: latest; v0 reproduces the previous model)")
        command.add_argument("--threads", type=int, default=4, help="CPU workers for latest (default: 4)")
        command.add_argument("--histidine-states", type=Path, help="CSV: res_uid,histidine_state")
        command.add_argument("--prepare", action="store_true",
                             help="add missing heavy atoms (single structure only; rejected with trajectories)"
                             if command is p else argparse.SUPPRESS)
        command.add_argument("--overwrite", action="store_true", help="allow replacing existing latest-model outputs")
        command.add_argument("--embeddings", action="store_true", help="include per-frame embeddings in the trajectory archive (automatic for structures)")
        command.add_argument("--csv", action="store_true", help="also export a wide trajectory CSV (latest only)")

    # -------- install-torch ------------------------------------------
    it = sub.add_parser(
        "install-torch",
        help="install or replace PyTorch in the current environment",
        description="Install or replace PyTorch in the current environment.",
    )
    it.add_argument(
        "--variant",
        choices=("cpu", "cu118", "cu121"),
        default="cpu",
        help="PyTorch wheel channel to install (default: cpu)",
    )
    it.add_argument(
        "--torch-spec",
        default="torch>=2.2,<2.12",
        help="torch requirement specifier to install",
    )
    it.add_argument(
        "--dry-run",
        action="store_true",
        help="print the pip command without executing it",
    )
    it.add_argument(
        "--no-upgrade",
        action="store_true",
        help="do not pass --upgrade to pip",
    )

    return ap


def _load_predictor_or_exit():
    try:
        from .predictors.fdewet import FdewetPredictor
    except ModuleNotFoundError as err:
        if err.name == "torch":
            print(
                "FastHydroMap inference requires PyTorch, but torch is not installed in this environment.\n"
                "Recommended CPU install:\n"
                "  fasthydromap install-torch\n\n"
                "Optional GPU install:\n"
                "  fasthydromap install-torch --variant cu121",
                file=sys.stderr,
            )
            sys.exit(2)
        raise
    return FdewetPredictor


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd not in {"predict", "predict-trajectory", "install-torch"}:
        print("Unknown sub-command", file=sys.stderr)
        sys.exit(1)

    if args.cmd == "install-torch":
        try:
            command = torch_install_command(
                args.variant,
                torch_spec=args.torch_spec,
                upgrade=not args.no_upgrade,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print("Torch install command:")
        print("  " + " ".join(command))
        if args.dry_run:
            return

        try:
            install_torch(
                args.variant,
                torch_spec=args.torch_spec,
                upgrade=not args.no_upgrade,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            parser.exit(2, f"error: PyTorch installation failed: {exc}\n")
        print("✓ torch installation completed")
        print("  Default is CPU; CUDA wheel variants are available on Linux and Windows.")
        return

    # Fail before importing/loading a predictor or touching any output paths.
    # Applies to both trajectory spellings and both model versions.
    if args.dcd is not None and args.prepare:
        parser.error("--prepare is only available for a single structure, not trajectories. "
                     "Use a complete topology whose atom count and order match the trajectory.")
    if args.csv and args.dcd is None:
        parser.error("--csv is a trajectory option; structures always include a CSV.")
    if args.quantity is None:
        args.quantity = "all" if args.model == "latest" else "fdewet"

    if args.model == "latest":
        if args.threads < 1:
            parser.error("--threads must be a positive integer")
        if args.parts:
            parser.error("The latest model has no intrinsic/context decomposition; omit --parts or use --model v0.")
        if args.quantity != "all":
            parser.error("The latest model always exports all four targets; omit --quantity. "
                         "Single-target selection is available with --model v0.")
        _load_predictor_or_exit()  # shared friendly missing-PyTorch message
        from .v2.command import run
        try:
            run(args)
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)
        return
    if args.quantity == "all" or args.histidine_states or args.prepare or args.embeddings or args.csv:
        print("error: --quantity all, --histidine-states, --prepare, --embeddings and --csv require --model latest", file=sys.stderr)
        sys.exit(2)
    FdewetPredictor = _load_predictor_or_exit()
    predictor = FdewetPredictor(quantity=args.quantity)
    quantity_label = QUANTITY_LABELS[args.quantity]
    quantity_suffix = args.quantity

    if args.cmd == "predict":
        if args.parts and args.dcd is not None:
            print(
                "error: --parts is only supported for single-structure 'predict'. "
                "Use 'predict-trajectory' for trajectory decomposition outputs.",
                file=sys.stderr,
            )
            sys.exit(2)

        # -------------------------------------------------------------
        # 1.  Run model
        # -------------------------------------------------------------
        if args.parts:
            parts, res_ids = predictor(args.pdb, return_parts=True)
            scores = parts["total"]
        else:
            scores, res_ids = predictor(args.pdb, args.dcd)     # (L,) or (F,L)

        # ensure 2-D shape so the CSV writer is uniform
        scores_2d: np.ndarray
        if scores.ndim == 1:          # single structure → make it (1,L)
            scores_2d = scores[None, :]
        else:                         # trajectory already (F,L)
            scores_2d = scores

        # -------------------------------------------------------------
        # 2.  Outputs
        # -------------------------------------------------------------
        outroot = args.outroot or args.pdb.with_suffix("")
        if args.outroot is None:
            outroot = outroot.with_name(outroot.name + f"_{quantity_suffix}")

        csv_path = Path(f"{outroot}.csv")
        pdb_path = Path(f"{outroot}.pdb")

        # --- CSV ------------------------------------------------------
        if args.dcd is None:
            data = {
                "residue": [str(r) for r in res_ids],
                quantity_label: scores.round(2),
            }
            if args.parts:
                data[f"{quantity_label}_intrinsic"] = parts["intrinsic"].round(2)
                data[f"{quantity_label}_context"] = parts["context"].round(2)
            df = pd.DataFrame(data)
        else:
            col_names = [str(r) for r in res_ids]               # columns = residue numbers
            df = pd.DataFrame(scores_2d.round(2), columns=col_names)
            df.insert(0, "frame", np.arange(len(scores_2d)))    # helpful index
        df.to_csv(csv_path, index=False)

        # --- PDB with B-factors (first frame only) --------------------
        write_bfactor(args.pdb, scores_2d[0], pdb_path)

        print("✓ results written to:")
        print("   •", csv_path)
        print("   •", pdb_path)
        return

    # -----------------------------------------------------------------
    # predict-trajectory
    # -----------------------------------------------------------------
    if args.parts:
        parts, res_ids = predictor(args.pdb, args.dcd, return_parts=True)
        total = parts["total"]
        intrinsic = parts["intrinsic"]
        context = parts["context"]
    else:
        total, res_ids = predictor(args.pdb, args.dcd)

    outroot = args.outroot or args.dcd.with_suffix("")
    if args.outroot is None:
        outroot = outroot.with_name(outroot.name + f"_{quantity_suffix}_traj")

    total_csv = Path(f"{outroot}_total.csv")
    intrinsic_csv = Path(f"{outroot}_intrinsic.csv")
    context_csv = Path(f"{outroot}_context.csv")
    summary_csv = Path(f"{outroot}_parts_summary.csv")
    pdb_path = Path(f"{outroot}.pdb")

    col_names = [str(r) for r in res_ids]
    df = pd.DataFrame(total, columns=col_names)
    df.insert(0, "frame", np.arange(total.shape[0], dtype=np.int64))
    df.to_csv(total_csv, index=False)

    if args.parts:
        for arr, path in (
            (intrinsic, intrinsic_csv),
            (context, context_csv),
        ):
            df = pd.DataFrame(arr, columns=col_names)
            df.insert(0, "frame", np.arange(arr.shape[0], dtype=np.int64))
            df.to_csv(path, index=False)

        summary = pd.DataFrame(
            {
                "frame": np.arange(total.shape[0], dtype=np.int64),
                "total_mean": total.mean(axis=1),
                "total_std": total.std(axis=1),
                "intrinsic_mean": intrinsic.mean(axis=1),
                "intrinsic_std": intrinsic.std(axis=1),
                "context_mean": context.mean(axis=1),
                "context_std": context.std(axis=1),
            }
        )
        summary.to_csv(summary_csv, index=False)

    write_bfactor(args.pdb, total[0], pdb_path)

    print("✓ trajectory results written to:")
    print("   •", total_csv)
    if args.parts:
        print("   •", intrinsic_csv)
        print("   •", context_csv)
        print("   •", summary_csv)
    print("   •", pdb_path)


if __name__ == "__main__":
    main()
