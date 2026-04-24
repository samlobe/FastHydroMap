# FastHydroMap/featurize/graph.py
"""
Graph construction for FastHydroMap MPNN.

* Node dim = 32  (20 one-hot AA + 7 heavy-atom SASA + 3 relative SASA + 2 termini flags)
* Edge dim = 84 by default (25 atom-pair distances expanded with 3-RBF + 9 orientation cosines)
"""

from pathlib import Path
import numpy as np, pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
import torch

from ..tensor_graph import TensorGraph
from ..utils.constants import AA3_TO_1
from ..utils.atom_names import backbone_alias_priority, canonical_backbone_atom_name
from .sasa import SASA_COMPONENT_NAMES, SASA_RELATIVE_NAMES

# ───── constants ────────────────────────────────────────────────────────
AA       = "ACDEFGHIKLMNPQRSTVWY"
aa2idx   = {a: i for i, a in enumerate(AA)}
UNK      = 20                                 # index for non-standard
RBF_DEFAULT_N = 3
RBF_MIN = 2.0
RBF_MAX = 14.0
SIGMA    = 4.0
PAIR_LIST = [                                 # 25 heavy-atom pairs
    ("CA","CA"), ("N","N"), ("C","C"), ("O","O"), ("CB","CB"),
    ("CA","N"), ("CA","C"), ("CA","O"), ("CA","CB"),
    ("N","C"), ("N","O"), ("N","CB"),
    ("CB","C"), ("CB","O"), ("O","C"),
    ("N","CA"), ("C","CA"), ("O","CA"), ("CB","CA"),
    ("C","N"), ("O","N"), ("CB","N"),
    ("C","CB"), ("O","CB"), ("C","O")
]

# ───── helpers ──────────────────────────────────────────────────────────
def _rbf(d, centers, sigma):              # Gaussian basis
    return np.exp(-(d[..., None] - centers) ** 2 / (2 * sigma ** 2))

def _one_hot(idxs, n=20):
    idxs = np.asarray(idxs, dtype=np.int64)
    out = np.zeros((len(idxs), n), np.float32)
    valid = (idxs >= 0) & (idxs < n)
    out[np.arange(len(idxs))[valid], idxs[valid]] = 1
    return out

def _res_uid(res) -> str:
    chain_id = res.get_parent().id.strip() or "_"
    _, resid, icode = res.id
    return f"{chain_id}:{resid}{icode.strip()}"

def _row_for_residue(df_sub: pd.DataFrame, res):
    resid = res.id[1]
    res_uid = _res_uid(res)

    if isinstance(df_sub.index, pd.MultiIndex):
        chain_id = res.get_parent().id.strip() or "_"
        icode = res.id[2].strip()
        for key in ((chain_id, resid, icode), (chain_id, resid), (resid,)):
            try:
                row = df_sub.loc[key]
            except KeyError:
                continue
            else:
                return row
        return None

    if res_uid in df_sub.index:
        return df_sub.loc[res_uid]
    if resid in df_sub.index:
        return df_sub.loc[resid]
    return None


def _row_value(row, key: str, default=0.0):
    if key in row.index and pd.notna(row[key]):
        return row[key]
    return default

def _safe_xyz(res, name):
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

    if name == "CB" and all(a in res and res[a].element != "H" for a in ("N","CA","C")):
        ca, n, c = (res[a].coord for a in ("CA","N","C"))
        b, cvec  = ca - n, c - ca
        a        = np.cross(b, cvec)
        cb       = (-0.58273431*a + 0.56802827*b - 0.54067466*cvec) + ca
        return cb.astype(np.float32)
    return np.full(3, np.nan, np.float32)

def _local_frame(ad):
    ca, n, c = ad["CA"], ad["N"], ad["C"]
    if any(x is None or np.isnan(x).any() for x in (ca, n, c)):
        return None
    u = c - ca; v = n - ca
    u /= np.linalg.norm(u) + 1e-8
    v /= np.linalg.norm(v) + 1e-8
    w  = np.cross(u, v); w /= np.linalg.norm(w) + 1e-8
    return u.astype(np.float32), v.astype(np.float32), w.astype(np.float32)

# ───── public API ───────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
#  build_graph
# ──────────────────────────────────────────────────────────────────────────────
def build_graph(
    pdb_path  : Path,
    df_sub    : pd.DataFrame,
    *,
    atom_coords : dict[str, np.ndarray] | None = None,   # (L,3) Å per key
    k_nn      : int  = 16,
    n_rbf     : int  = RBF_DEFAULT_N,
    rbf_min   : float = RBF_MIN,
    rbf_max   : float = RBF_MAX,
    rbf_sigma : float = SIGMA,
    pid_ix    : int  = 0,
) -> TensorGraph:
    """
    Build a PyG graph for **either**
      • a static PDB structure   (use heavy-atom coords from the file), or
      • a trajectory *frame*     (pass per-residue `atom_coords` overrides).

    Parameters
    ----------
    pdb_path : Path
        Path to the PDB file (always needed for residue order / names).
    df_sub : DataFrame
        Index = residue IDs; must contain `Fdewet_pred` and should provide the
        heavy-atom SASA feature columns written by `00_build_sasa.py`.
    atom_coords : dict[str, np.ndarray] | None, default = None
        If given, must map the keys  **"N", "CA", "C", "O", "CB"**
        to arrays of shape ``(L,3)`` (Å).  NaNs are allowed.
        When *None*, coordinates are read from `pdb_path`.
    k_nn : int
        k-nearest neighbours (on Cα) to connect.
    pid_ix : int
        Integer identifier stored inside the graph so that batched
        PyG `collate` keeps track of which graph came from which PDB.

    Returns
    -------
    TensorGraph
    """
    rbf_centers = np.linspace(rbf_min, rbf_max, n_rbf)

    # ------------------------------------------------------------------
    # 1.  iterate residues in PDB order
    # ------------------------------------------------------------------
    struct = PDBParser(QUIET=True).get_structure("", str(pdb_path))[0]
    residue_iter = struct.get_residues()

    ca_xyz, atoms, frames = [], [], []
    aa_idx, resid_list, y_vals = [], [], []
    raw_sasa = {name: [] for name in SASA_COMPONENT_NAMES}
    rel_sasa = {name: [] for name in SASA_RELATIVE_NAMES}
    is_n_term, is_c_term = [], []

    coord_idx = 0
    for res in residue_iter:
        if res.id[0].strip():
            continue
        if "CA" not in res:
            continue
        row = _row_for_residue(df_sub, res)
        if row is None:
            coord_idx += 1
            continue

        # ----- coordinates -------------------------------------------------
        if atom_coords is None:                       # single-structure mode
            ad = {a: _safe_xyz(res, a) for a in ("N","CA","C","O","CB")}
        else:                                         # trajectory frame mode
            ad = {a: atom_coords[a][coord_idx] for a in ("N","CA","C","O","CB")}
        coord_idx += 1

        lf = _local_frame(ad)
        if lf is None:                                # skip malformed residue
            continue

        # ----- collect per-node data --------------------------------------
        ca_xyz.append(ad["CA"]); atoms.append(ad); frames.append(lf)

        aa1 = AA3_TO_1.get(res.resname.strip().upper(), "X")
        aa_idx  .append(aa2idx.get(aa1, UNK))
        for comp_name in SASA_COMPONENT_NAMES:
            raw_sasa[comp_name].append(float(_row_value(row, f"{comp_name}_sasa", 0.0)))
        for rel_name in SASA_RELATIVE_NAMES:
            rel_sasa[rel_name].append(float(_row_value(row, rel_name, 0.0)))
        is_n_term.append(float(bool(_row_value(row, "is_n_term", False))))
        is_c_term.append(float(bool(_row_value(row, "is_c_term", False))))
        resid_list.append(res.id[1])
        y_vals  .append(float(row["Fdewet_pred"]))

    L = len(ca_xyz)
    if L == 0:
        raise ValueError("no residues in graph")

    ca_xyz = np.vstack(ca_xyz).astype(np.float32)

    node = np.concatenate(
        [_one_hot(aa_idx, 20),
         np.column_stack([np.asarray(raw_sasa[name], dtype=np.float32) for name in SASA_COMPONENT_NAMES]),
         np.column_stack([np.asarray(rel_sasa[name], dtype=np.float32) for name in SASA_RELATIVE_NAMES]),
         np.asarray(is_n_term, dtype=np.float32)[:, None],
         np.asarray(is_c_term, dtype=np.float32)[:, None]], axis=1
    )

    # ------------------------------------------------------------------
    # 2.  k-NN edges (on Cα)
    # ------------------------------------------------------------------
    knn = cKDTree(ca_xyz).query(ca_xyz, k=min(k_nn + 1, L))[1][:, 1:]
    src = np.repeat(np.arange(L), knn.shape[1])
    dst = knn.reshape(-1)
    E   = len(src)

    # ------------------------------------------------------------------
    # 3.  edge attributes
    # ------------------------------------------------------------------
    dist_rows = []
    for a1, a2 in PAIR_LIST:
        xyz1 = np.array([atoms[i][a1] for i in src])
        xyz2 = np.array([atoms[i][a2] for i in dst])
        d = np.linalg.norm(xyz1 - xyz2, axis=1)
        d[np.isnan(d)] = 22.0             # push NaN pairs outside RBF window
        dist_rows.append(_rbf(d, rbf_centers, rbf_sigma))
    dist_block = np.concatenate(dist_rows, 1)         # (E,25*n_rbf)

    orient = np.zeros((E, 9), np.float32)
    for k, (i, j) in enumerate(zip(src, dst)):
        ui, vi, wi = frames[i];  uj, vj, wj = frames[j]
        orient[k] = [ui @ uj, ui @ vj, ui @ wj,
                     vi @ uj, vi @ vj, vi @ wj,
                     wi @ uj, wi @ vj, wi @ wj]

    edge_attr = np.concatenate([dist_block, orient], 1)  # (E,25*n_rbf+9)

    # ------------------------------------------------------------------
    # 4.  pack PyG Data object
    # ------------------------------------------------------------------
    return TensorGraph(
        x          = torch.from_numpy(node).float(),
        pos        = torch.from_numpy(ca_xyz).float(),
        edge_index = torch.from_numpy(np.stack([src, dst], 0)).long(),
        edge_attr  = torch.from_numpy(edge_attr).float(),
        y          = torch.from_numpy(np.asarray(y_vals , np.float32)),
        resid      = torch.from_numpy(np.asarray(resid_list, np.int64)),
        pdb_id     = torch.tensor([pid_ix], dtype=torch.long),
    )

__all__ = ["build_graph"]
