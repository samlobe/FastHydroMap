#!/usr/bin/env python3
"""Build residue-level SASA features for FastHydroMap training."""

import os, warnings
os.environ["OPENMM_DEFAULT_PLATFORM"] = "CPU"

from pathlib import Path
import numpy as np, pandas as pd, mdtraj as md
from pdbfixer import PDBFixer
from openmm.app import Modeller, PDBFile, forcefield as _ff
from openmm import unit as u
from tqdm import tqdm
from residue_keys import ensure_residue_key_columns, residue_uid, sort_residue_rows
from FastHydroMap.featurize.sasa import (
    SASA_COMPONENT_NAMES,
    compute_relative_sasa_features,
    residue_sasa_components,
    save_sasa_feature_stats,
)
from FastHydroMap.io.residue_qc import collect_residue_records
from FastHydroMap.utils.constants import AA

ROOT      = Path(__file__).resolve().parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"; MODEL_DIR.mkdir(exist_ok=True)

META_CSV  = DATA_DIR / "all_residue_results.csv"
PDB_DIR   = DATA_DIR / "rcsb_pdbs"
SPLITS_YML = DATA_DIR / "splits.yaml"
SASA_STATS_NPZ = MODEL_DIR / "sasa_feature_stats.npz"

BL = {"7TJQ","7MY2","1LVE","1OBQ","1IFG",
      "1F53","1PE0","1SVR","2AQZ","2ITG","3UNO","3S7H","6J62"}

FF = _ff.ForceField("amber14-all.xml", "amber14/tip3p.xml")

def compute_sasa_and_aa(pdb_path: Path):
    """Return per-residue SASA and residue metadata in PDB order."""
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues(); fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(False)
    fixer.findMissingAtoms(); fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)
    mod = Modeller(fixer.topology, fixer.positions)

    xyz_nm = np.asarray(mod.positions.value_in_unit(u.nanometer), np.float32)
    traj   = md.Trajectory(xyz_nm[np.newaxis,:,:],
                           md.Topology.from_openmm(mod.topology))
    components = residue_sasa_components(traj, strict=True)     # (L,7)
    sasa = components.sum(axis=1)

    records, _, nonstandard = collect_residue_records(pdb_path)
    if nonstandard:
        warnings.warn(f"{pdb_path.stem}: nonstandard polymer residues treated as unknown: {', '.join(nonstandard)}")

    return sasa, components, records


def _feature_stats(train_rows: pd.DataFrame, feature: str) -> tuple[np.ndarray, np.ndarray, np.float32, np.float32]:
    means = np.zeros(len(AA), dtype=np.float32)
    stds = np.ones(len(AA), dtype=np.float32)
    global_mean = np.float32(train_rows[feature].mean())
    global_std = np.float32(train_rows[feature].std() + 1e-8)

    for idx, aa in enumerate(AA):
        aa_rows = train_rows[train_rows["aa"] == aa]
        if aa_rows.empty:
            means[idx] = global_mean
            stds[idx] = global_std
        else:
            means[idx] = np.float32(aa_rows[feature].mean())
            stds[idx] = np.float32(aa_rows[feature].std() + 1e-8)
    return means, stds, global_mean, global_std

def main():
    df = ensure_residue_key_columns(pd.read_csv(META_CSV))
    df = df[~df.pdb_id.isin(BL)].copy()         # remove hard blacklist

    sasa_ser, aa_ser = [], []
    comp_sers = {name: [] for name in SASA_COMPONENT_NAMES}
    nterm_ser, cterm_ser = [], []
    chain_ser, icode_ser, uid_ser, order_ser = [], [], [], []
    skip = []
    for pid, g in tqdm(df.groupby("pdb_id"), desc="SASA"):
        pdb = PDB_DIR / f"{pid}.pdb"
        if not pdb.exists():
            warnings.warn(f"No PDB {pid}"); skip.append(pid); continue
        try:
            sasa, components, records = compute_sasa_and_aa(pdb)
        except Exception as e:
            warnings.warn(f"{pid}: {e}"); skip.append(pid); continue

        if len(sasa) != len(g) or len(records) != len(g) or len(components) != len(g):
            warnings.warn(
                f"{pid}: len mismatch ({len(sasa)} SASA / {len(components)} components / {len(records)} records vs {len(g)} rows)"
            )
            skip.append(pid); continue

        idx = sort_residue_rows(g).index
        rec_df = pd.DataFrame(records, copy=False)
        sasa_ser.append(pd.Series(sasa, index=idx))
        aa_ser.append(pd.Series(rec_df["aa"].to_numpy(), index=idx))
        for comp_idx, comp_name in enumerate(SASA_COMPONENT_NAMES):
            comp_sers[comp_name].append(pd.Series(components[:, comp_idx], index=idx))
        nterm_ser.append(pd.Series(rec_df["is_n_term"].to_numpy(dtype=np.bool_), index=idx))
        cterm_ser.append(pd.Series(rec_df["is_c_term"].to_numpy(dtype=np.bool_), index=idx))
        chain_ser.append(pd.Series(rec_df["chain_id"].to_numpy(), index=idx))
        icode_ser.append(pd.Series(rec_df["insertion_code"].to_numpy(), index=idx))
        uid_ser.append(pd.Series(rec_df["res_uid"].to_numpy(), index=idx))
        order_ser.append(pd.Series(np.arange(len(rec_df), dtype=np.int64), index=idx))

    # drop entire PDBs that failed
    if skip:
        print("⚠️  Skipped:", ", ".join(skip))
        df = df[~df.pdb_id.isin(skip)]

    df["sasa"] = pd.concat(sasa_ser).sort_index()
    df["aa"]   = pd.concat(aa_ser).sort_index()
    for comp_name, series_list in comp_sers.items():
        df[f"{comp_name}_sasa"] = pd.concat(series_list).sort_index()
    df["is_n_term"] = pd.concat(nterm_ser).sort_index()
    df["is_c_term"] = pd.concat(cterm_ser).sort_index()
    df["chain_id"] = pd.concat(chain_ser).sort_index()
    df["insertion_code"] = pd.concat(icode_ser).sort_index()
    df["res_uid"] = pd.concat(uid_ser).sort_index()
    df["res_order"] = pd.concat(order_ser).sort_index().astype(int)

    import yaml

    splits = yaml.safe_load(open(SPLITS_YML))
    trusted = (
        (df["pdb_id"].isin(splits["train"]))
        & (df["avg_n_waters"] > 7.0)
        & df["Fdewet_pred"].between(3.8, 8.7)
        & df["aa"].isin(list(AA))
    )
    train_rows = df.loc[trusted]
    stats = {}
    for feat in ("sasa", "sc_C_sasa", "sc_NOS_sasa"):
        means, stds, global_mean, global_std = _feature_stats(train_rows, feat)
        stats[f"{feat}_mean"] = means
        stats[f"{feat}_std"] = stds
        stats[f"{feat}_global_mean"] = np.float32(global_mean)
        stats[f"{feat}_global_std"] = np.float32(global_std)
    save_sasa_feature_stats(SASA_STATS_NPZ, stats)

    rel = compute_relative_sasa_features(
        df["aa"].to_numpy(),
        df["sasa"].to_numpy(np.float32),
        df["sc_C_sasa"].to_numpy(np.float32),
        df["sc_NOS_sasa"].to_numpy(np.float32),
        stats,
    )
    for name, values in rel.items():
        df[name] = values

    df.to_csv(META_CSV, index=False)
    print("✓ wrote heavy-atom SASA features to", META_CSV)
    print("✓ wrote SASA feature stats to", SASA_STATS_NPZ)

if __name__ == "__main__":
    main()
