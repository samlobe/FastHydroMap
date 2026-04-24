import mdtraj as md
import numpy as np

from ..utils.constants import AA, AA2IDX
from ..utils.atom_names import canonical_backbone_atom_name


SASA_COMPONENT_NAMES = ("N", "CA", "C", "O", "CB", "sc_C", "sc_NOS")
SASA_RELATIVE_NAMES = ("sasa_rel", "sc_C_rel", "sc_NOS_rel")
SASA_STAT_FEATURES = ("sasa", "sc_C_sasa", "sc_NOS_sasa")


def residue_sasa(traj: md.Trajectory) -> np.ndarray:  # shape (L,)
    heavy = _heavy_atom_slice(traj)
    return md.shrake_rupley(heavy, mode="residue")[0].astype(np.float32)


def _heavy_atom_slice(traj: md.Trajectory) -> md.Trajectory:
    heavy_idx = [
        atom.index
        for atom in traj.topology.atoms
        if atom.element is not None and atom.element.symbol.upper() != "H"
    ]
    return traj.atom_slice(heavy_idx)


def _classify_sasa_atom(name: str, element: str) -> int | None:
    elem = element.strip().upper()
    canonical = canonical_backbone_atom_name(name, elem)

    if canonical == "N":
        return 0
    if canonical == "CA":
        return 1
    if canonical == "C":
        return 2
    if canonical == "O":
        return 3
    if canonical == "CB":
        return 4
    if elem == "C":
        return 5
    if elem in {"N", "O", "S"}:
        return 6
    return None


def residue_sasa_components(
    traj: md.Trajectory,
    *,
    strict: bool = False,
    return_other: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Decompose per-residue heavy-atom SASA into chemically meaningful buckets.

    Buckets:
    * ``N``, ``CA``, ``C``, ``O``, ``CB``
    * ``sc_C``: sidechain carbons beyond CB
    * ``sc_NOS``: sidechain nitrogens, oxygens, and sulfurs

    The decomposition is performed on a heavy-atom-only slice so hydrogen
    addition from PDBFixer does not leak SASA into the sidechain buckets.
    """

    heavy = _heavy_atom_slice(traj)
    atom_sasa = md.shrake_rupley(heavy, mode="atom").astype(np.float32)

    components = np.zeros(
        (heavy.n_frames, heavy.n_residues, len(SASA_COMPONENT_NAMES)),
        dtype=np.float32,
    )
    other = np.zeros((heavy.n_frames, heavy.n_residues), dtype=np.float32)

    for residue in heavy.topology.residues:
        res_idx = residue.index
        for atom in residue.atoms:
            elem = atom.element.symbol if atom.element is not None else ""
            bucket = _classify_sasa_atom(atom.name, elem)
            if bucket is None:
                other[:, res_idx] += atom_sasa[:, atom.index]
            else:
                components[:, res_idx, bucket] += atom_sasa[:, atom.index]

    if strict and np.any(other > 1e-6):
        bad = []
        for residue in heavy.topology.residues:
            if np.any(other[:, residue.index] > 1e-6):
                bad_atoms = [
                    f"{atom.name.strip()}:{atom.element.symbol if atom.element else '?'}"
                    for atom in residue.atoms
                    if _classify_sasa_atom(
                        atom.name, atom.element.symbol if atom.element else ""
                    )
                    is None
                ]
                bad.append(f"{residue} -> {bad_atoms}")
        raise ValueError("unmapped heavy atoms in SASA decomposition: " + "; ".join(bad))

    if traj.n_frames == 1:
        out = components[0]
        return (out, other[0]) if return_other else out

    return (components, other) if return_other else components


def save_sasa_feature_stats(path, stats: dict[str, np.ndarray]) -> None:
    np.savez(path, aa_order=np.array(list(AA), dtype="<U1"), **stats)


def load_sasa_feature_stats(path) -> dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=False)
    return {k: arr[k] for k in arr.files}


def compute_relative_sasa_features(
    aa_codes,
    sasa_total: np.ndarray,
    sc_c_sasa: np.ndarray,
    sc_nos_sasa: np.ndarray,
    stats: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    aa_codes = np.asarray(aa_codes)
    feat_values = {
        "sasa": np.asarray(sasa_total, dtype=np.float32),
        "sc_C_sasa": np.asarray(sc_c_sasa, dtype=np.float32),
        "sc_NOS_sasa": np.asarray(sc_nos_sasa, dtype=np.float32),
    }

    aa_idx = np.array([AA2IDX.get(str(aa), -1) for aa in aa_codes], dtype=np.int64)
    rel = {}
    for feat_name, out_name in zip(SASA_STAT_FEATURES, SASA_RELATIVE_NAMES):
        values = feat_values[feat_name]
        means = np.asarray(stats[f"{feat_name}_mean"], dtype=np.float32)
        stds = np.asarray(stats[f"{feat_name}_std"], dtype=np.float32)
        global_mean = np.float32(stats[f"{feat_name}_global_mean"])
        global_std = np.float32(stats[f"{feat_name}_global_std"])

        out = np.empty_like(values, dtype=np.float32)
        valid = aa_idx >= 0
        out[valid] = (values[valid] - means[aa_idx[valid]]) / stds[aa_idx[valid]]
        out[~valid] = (values[~valid] - global_mean) / global_std
        rel[out_name] = out
    return rel
