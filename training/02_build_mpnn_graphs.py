#!/usr/bin/env python3
"""Build cached residue graphs for FastHydroMap training."""

import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from residue_keys import (
    ensure_residue_key_columns,
    has_explicit_residue_keys,
    residue_uid_from_biopdb,
)
from FastHydroMap.utils.constants import AA3_TO_1
from FastHydroMap.utils.atom_names import backbone_alias_priority, canonical_backbone_atom_name
from FastHydroMap.featurize.sasa import SASA_COMPONENT_NAMES, SASA_RELATIVE_NAMES

ROOT     = Path(__file__).resolve().parent / "data"
CSV_FILE = ROOT / "all_residue_results.csv"
PDB_DIR  = ROOT / "rcsb_pdbs"
OUT_DIR  = ROOT / "graphs"; OUT_DIR.mkdir(exist_ok=True)

K_NEIGH  = 16
RBF_DEFAULT_N = 16
RBF_MIN = 2.0
RBF_MAX = 20.0
SIGMA    = 2.0
AA       = "ACDEFGHIKLMNPQRSTVWY"
aa2idx   = {a:i for i,a in enumerate(AA)}
UNK      = 20


PAIR_LIST = [
    ("CA","CA"), ("N","N"), ("C","C"), ("O","O"), ("CB","CB"),
    ("CA","N"), ("CA","C"), ("CA","O"), ("CA","CB"),
    ("N","C"), ("N","O"), ("N","CB"),
    ("CB","C"), ("CB","O"), ("O","C"),
    ("N","CA"), ("C","CA"), ("O","CA"), ("CB","CA"),
    ("C","N"), ("O","N"), ("CB","N"),
    ("C","CB"), ("O","CB"), ("C","O")
]
EDGE_MODE_DEFAULT = "rbf"  # rbf or raw
PAIR_SET_DEFAULT = "full"  # full or ca_only

def rbf(d, centers, sigma): return np.exp(-(d[...,None]-centers)**2 / (2*sigma**2))

def one_hot(idx, n=20):
    out = np.zeros((len(idx), n), np.float32)
    out[np.arange(len(idx)), idx] = 1
    return out

def safe_xyz(res,name):
    best = None
    best_priority = 9999
    for atom in res:
        elem = (atom.element or "").strip().upper()
        if elem == "H":
            continue
        canonical = canonical_backbone_atom_name(atom.name, elem)
        if canonical != name:
            continue
        priority = backbone_alias_priority(atom.name)
        if priority < best_priority:
            best_priority = priority
            best = atom.coord.astype(np.float32)
    if best is not None:
        return best

    if name=="CB" and all(a in res and res[a].element!="H" for a in ("N","CA","C")):
        ca,n,c = (res[a].coord for a in ("CA","N","C"))
        b,cvec = ca-n, c-ca; a = np.cross(b,cvec)
        cb = (-0.58273431*a + 0.56802827*b - 0.54067466*cvec) + ca
        return cb.astype(np.float32)
    return np.full(3,np.nan,np.float32)

def local_frame(ad):
    ca,n,c = ad["CA"], ad["N"], ad["C"]
    if any(x is None or np.isnan(x).any() for x in (ca,n,c)): return None,None,None
    u,v = c-ca, n-ca
    u/=np.linalg.norm(u)+1e-8; v/=np.linalg.norm(v)+1e-8
    w = np.cross(u,v); w/=np.linalg.norm(w)+1e-8
    return u.astype(np.float32), v.astype(np.float32), w.astype(np.float32)


def row_for_residue(df_sub: pd.DataFrame, res):
    res_uid = residue_uid_from_biopdb(res)
    resid = res.id[1]
    if res_uid in df_sub.index:
        return df_sub.loc[res_uid], res_uid
    if resid in df_sub.index:
        return df_sub.loc[resid], str(resid)
    return None, None


def row_value(row, key: str, default=0.0):
    if key in row.index and pd.notna(row[key]):
        return row[key]
    return default

def _pair_list(pair_set: str):
    if pair_set == "ca_only":
        return [("CA", "CA")]
    if pair_set == "ca_cb":
        return [("CA", "CA"), ("CA", "CB"), ("CB", "CA"), ("CB", "CB")]
    return PAIR_LIST


def build_graph(
    pdb_path: Path,
    df_sub: pd.DataFrame,
    k_neigh: int = K_NEIGH,
    n_rbf: int = RBF_DEFAULT_N,
    edge_mode: str = EDGE_MODE_DEFAULT,
    pair_set: str = PAIR_SET_DEFAULT,
    rbf_min: float = RBF_MIN,
    rbf_max: float = RBF_MAX,
    rbf_sigma: float = SIGMA,
) -> Data:
    rbf_centers = np.linspace(rbf_min, rbf_max, n_rbf)
    pairs = _pair_list(pair_set)
    model = PDBParser(QUIET=True).get_structure("", pdb_path)[0]

    ca_xyz, atoms, frames = [],[],[]
    aa_idx, resid_list, y_vals, res_uid_list = [],[],[],[]
    raw_sasa = {name: [] for name in SASA_COMPONENT_NAMES}
    rel_sasa = {name: [] for name in SASA_RELATIVE_NAMES}
    is_n_term, is_c_term = [], []

    for res in model.get_residues():
        if res.id[0].strip() or "CA" not in res:
            continue
        row, res_uid = row_for_residue(df_sub, res)
        if row is None:
            continue

        ad = {a: safe_xyz(res,a) for a in ("N","CA","C","O","CB")}
        lf = local_frame(ad)
        if lf[0] is None: continue

        ca_xyz.append(ad["CA"]); atoms.append(ad); frames.append(lf)
        aa = AA3_TO_1.get(res.resname.strip().upper(), "X")
        aa_idx.append(aa2idx.get(aa, UNK))
        for comp_name in SASA_COMPONENT_NAMES:
            raw_sasa[comp_name].append(float(row_value(row, f"{comp_name}_sasa", 0.0)))
        for rel_name in SASA_RELATIVE_NAMES:
            rel_sasa[rel_name].append(float(row_value(row, rel_name, 0.0)))
        is_n_term.append(float(bool(row_value(row, "is_n_term", False))))
        is_c_term.append(float(bool(row_value(row, "is_c_term", False))))
        resid_list.append(res.id[1])
        res_uid_list.append(res_uid)
        y_vals.append(row.Fdewet_pred)

    L = len(ca_xyz)
    if L == 0: raise ValueError("No residues")

    ca_xyz = np.vstack(ca_xyz).astype(np.float32)
    node   = np.concatenate([
        one_hot(aa_idx),
        np.column_stack([np.asarray(raw_sasa[name], dtype=np.float32) for name in SASA_COMPONENT_NAMES]),
        np.column_stack([np.asarray(rel_sasa[name], dtype=np.float32) for name in SASA_RELATIVE_NAMES]),
        np.asarray(is_n_term, dtype=np.float32)[:, None],
        np.asarray(is_c_term, dtype=np.float32)[:, None],
    ], axis=1)

    knn = cKDTree(ca_xyz).query(ca_xyz, k=min(k_neigh + 1, L))[1][:,1:]
    src = np.repeat(np.arange(L), knn.shape[1]); dst = knn.reshape(-1); E=len(src)

    dist_block=[]
    for a1,a2 in pairs:
        xyz1 = np.array([atoms[i][a1] for i in src])
        xyz2 = np.array([atoms[i][a2] for i in dst])
        d=np.linalg.norm(xyz1-xyz2,axis=1); d[np.isnan(d)]=22.
        if edge_mode == "rbf":
            dist_block.append(rbf(d, rbf_centers, rbf_sigma))
        elif edge_mode == "raw":
            dist_block.append((d / 20.0)[:, None].astype(np.float32))
        else:
            raise ValueError(f"unknown edge_mode: {edge_mode}")
    dist_block = np.concatenate(dist_block,1)

    orient = np.zeros((E,9),np.float32)
    for k,(i,j) in enumerate(zip(src,dst)):
        ui,vi,wi = frames[i]; uj,vj,wj = frames[j]
        orient[k] = [ui@uj, ui@vj, ui@wj,
                     vi@uj, vi@vj, vi@wj,
                     wi@uj, wi@vj, wi@wj]
    edge_attr = np.concatenate([dist_block, orient],1)
    edge_attr  = torch.from_numpy(edge_attr).float()
    edge_index = torch.from_numpy(np.stack([src, dst], 0)).long()

    return Data(
        x      = torch.from_numpy(node).float(),
        pos    = torch.from_numpy(ca_xyz).float(),
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y      = torch.from_numpy(np.array(y_vals,  np.float32)),
        resid  = torch.from_numpy(np.array(resid_list, np.int64)),
        res_uid = res_uid_list,
        pdb_id = pdb_path.stem
    )

def _flt_tag(x: float) -> str:
    return str(x).replace("-", "m").replace(".", "p")


def _graph_cache_paths(
    k_neigh: int,
    n_rbf: int,
    edge_mode: str,
    pair_set: str,
    rbf_min: float,
    rbf_max: float,
    rbf_sigma: float,
) -> tuple[Path, Path]:
    suffix = ""
    if edge_mode != EDGE_MODE_DEFAULT:
        suffix += f"_{edge_mode}"
    if pair_set != PAIR_SET_DEFAULT:
        suffix += f"_{pair_set}"
    if n_rbf != RBF_DEFAULT_N:
        suffix += f"_rbf{n_rbf}"
    if edge_mode == "rbf" and (rbf_min != RBF_MIN or rbf_max != RBF_MAX or rbf_sigma != SIGMA):
        suffix += f"_r{_flt_tag(rbf_min)}to{_flt_tag(rbf_max)}_s{_flt_tag(rbf_sigma)}"
    return OUT_DIR / f"graphs_k{k_neigh}{suffix}.pt", OUT_DIR / f"pdb_ids_k{k_neigh}{suffix}.pt"


def build_dataset(
    k_neigh: int = K_NEIGH,
    n_rbf: int = RBF_DEFAULT_N,
    edge_mode: str = EDGE_MODE_DEFAULT,
    pair_set: str = PAIR_SET_DEFAULT,
    rbf_min: float = RBF_MIN,
    rbf_max: float = RBF_MAX,
    rbf_sigma: float = SIGMA,
):
    df = pd.read_csv(CSV_FILE)
    explicit_keys = has_explicit_residue_keys(df)
    df = ensure_residue_key_columns(df)
    index_col = "res_uid" if explicit_keys else "resid"
    df.set_index(["pdb_id", index_col], inplace=True)

    graphs, pids = [], []
    for pdb in tqdm(sorted(PDB_DIR.glob("*.pdb"))):
        pid = pdb.stem
        if pid not in df.index.levels[0]: continue
        try:
            graphs.append(
                build_graph(
                    pdb,
                    df.loc[pid],
                    k_neigh=k_neigh,
                    n_rbf=n_rbf,
                    edge_mode=edge_mode,
                    pair_set=pair_set,
                    rbf_min=rbf_min,
                    rbf_max=rbf_max,
                    rbf_sigma=rbf_sigma,
                )
            ); pids.append(pid)
        except Exception as e:
            print(f"⚠️  {pid} skipped → {e}")

    data,slices = InMemoryDataset.collate(graphs)
    gfile, pfile = _graph_cache_paths(k_neigh, n_rbf, edge_mode, pair_set, rbf_min, rbf_max, rbf_sigma)
    torch.save((data,slices), gfile); torch.save(pids, pfile)
    print(f"\n✅  {len(graphs)} graphs saved to {gfile}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=K_NEIGH, help="k-nearest neighbors for graph construction")
    parser.add_argument("--n-rbf", type=int, default=RBF_DEFAULT_N, help="number of RBF bins per atom-pair distance")
    parser.add_argument("--edge-mode", choices=["rbf", "raw"], default=EDGE_MODE_DEFAULT, help="distance encoding mode")
    parser.add_argument("--pair-set", choices=["full", "ca_only", "ca_cb"], default=PAIR_SET_DEFAULT, help="distance pair set")
    parser.add_argument("--rbf-min", type=float, default=RBF_MIN, help="minimum RBF center distance (A)")
    parser.add_argument("--rbf-max", type=float, default=RBF_MAX, help="maximum RBF center distance (A)")
    parser.add_argument("--rbf-sigma", type=float, default=SIGMA, help="RBF gaussian sigma (A)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        k_neigh=args.k,
        n_rbf=args.n_rbf,
        edge_mode=args.edge_mode,
        pair_set=args.pair_set,
        rbf_min=args.rbf_min,
        rbf_max=args.rbf_max,
        rbf_sigma=args.rbf_sigma,
    )
