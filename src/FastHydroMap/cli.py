# FastHydroMap/cli.py
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

from .io.pdb            import write_bfactor
from .install_torch import install_torch, torch_install_command


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fasthydromap",
                                 description="FastHydroMap – infer per-residue Fdewet")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # -------- predict -------------------------------------------------
    p = sub.add_parser("predict", help="run inference on PDB (± DCD trajectory)")
    p.add_argument("pdb", type=Path, help="input PDB topology")
    p.add_argument("dcd", nargs="?", type=Path, default=None,
                help="optional trajectory (DCD/XTC); if omitted runs single-structure mode")
    p.add_argument("-o", "--outroot", type=Path,
                help="basename for outputs (default: <pdb>_fdewet)",
                default=None)

    # -------- predict-trajectory --------------------------------------
    pt = sub.add_parser(
        "predict-trajectory",
        help="run trajectory inference and write total/intrinsic/context CSV outputs",
    )
    pt.add_argument("pdb", type=Path, help="input PDB topology")
    pt.add_argument("dcd", type=Path, help="input trajectory (DCD/XTC)")
    pt.add_argument(
        "-o",
        "--outroot",
        type=Path,
        default=None,
        help="basename for outputs (default: <dcd>_fdewet_traj)",
    )

    # -------- install-torch ------------------------------------------
    it = sub.add_parser(
        "install-torch",
        help="install or replace PyTorch in the current environment (advanced/manual helper)",
        description="Install or replace PyTorch in the current environment (advanced/manual helper).",
    )
    it.add_argument(
        "--variant",
        choices=("cpu", "cu118", "cu121"),
        default="cpu",
        help="PyTorch wheel channel to install (default: cpu)",
    )
    it.add_argument(
        "--torch-spec",
        default="torch>=2.2,<2.8",
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
    args = _build_parser().parse_args()

    if args.cmd not in {"predict", "predict-trajectory", "install-torch"}:
        print("Unknown sub-command", file=sys.stderr)
        sys.exit(1)

    if args.cmd == "install-torch":
        command = torch_install_command(
            args.variant,
            torch_spec=args.torch_spec,
            upgrade=not args.no_upgrade,
        )
        print("Torch install command:")
        print("  " + " ".join(command))
        if args.dry_run:
            return

        install_torch(
            args.variant,
            torch_spec=args.torch_spec,
            upgrade=not args.no_upgrade,
        )
        print("✓ torch installation completed")
        print("  This helper is mainly for advanced/manual CPU or GPU torch setup.")
        return

    FdewetPredictor = _load_predictor_or_exit()
    predictor = FdewetPredictor()

    if args.cmd == "predict":
        # -------------------------------------------------------------
        # 1.  Run model
        # -------------------------------------------------------------
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
            outroot = outroot.with_name(outroot.name + "_fdewet")

        csv_path = Path(f"{outroot}.csv")
        pdb_path = Path(f"{outroot}.pdb")

        # --- CSV ------------------------------------------------------
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
    # predict-trajectory: write decomposition tables
    # -----------------------------------------------------------------
    parts, res_ids = predictor(args.pdb, args.dcd, return_parts=True)
    total = parts["total"]
    intrinsic = parts["intrinsic"]
    context = parts["context"]

    outroot = args.outroot or args.dcd.with_suffix("")
    if args.outroot is None:
        outroot = outroot.with_name(outroot.name + "_fdewet_traj")

    total_csv = Path(f"{outroot}_total.csv")
    intrinsic_csv = Path(f"{outroot}_intrinsic.csv")
    context_csv = Path(f"{outroot}_context.csv")
    summary_csv = Path(f"{outroot}_parts_summary.csv")
    pdb_path = Path(f"{outroot}.pdb")

    col_names = [str(r) for r in res_ids]
    for arr, path in (
        (total, total_csv),
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
    print("   •", intrinsic_csv)
    print("   •", context_csv)
    print("   •", summary_csv)
    print("   •", pdb_path)


if __name__ == "__main__":
    main()
