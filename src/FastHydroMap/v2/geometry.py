from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser
from ..utils.atom_names import canonical_backbone_atom_name
from ..io.residue_qc import residue_uid


def residue_uid_from_biopdb(res):
    return residue_uid(res.parent.id, res.id[1], res.id[2])


AA_ALIASES = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS"}

FUNCTIONAL_GROUP_ATOMS = {
    "ALA": ("CB",),
    "ARG": ("NE", "CZ", "NH1", "NH2"),
    "ASN": ("CG", "OD1", "ND2"),
    "ASP": ("CG", "OD1", "OD2"),
    "CYS": ("SG",),
    "GLN": ("CD", "OE1", "NE2"),
    "GLU": ("CD", "OE1", "OE2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("CG1", "CG2", "CD1"),
    "LEU": ("CG", "CD1", "CD2"),
    "LYS": ("NZ",),
    "MET": ("SD", "CE"),
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("CG", "CD"),
    "SER": ("OG",),
    "THR": ("OG1", "CG2"),
    "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": ("CG1", "CG2"),
}

POLAR_TIP_ATOMS = {
    "ARG": ("CZ", "NH1", "NH2"),
    "ASN": ("OD1", "ND2"),
    "ASP": ("OD1", "OD2"),
    "CYS": ("SG",),
    "GLN": ("OE1", "NE2"),
    "GLU": ("OE1", "OE2"),
    "HIS": ("ND1", "NE2"),
    "LYS": ("NZ",),
    "MET": ("SD",),
    "SER": ("OG",),
    "THR": ("OG1",),
    "TRP": ("NE1",),
    "TYR": ("OH",),
}

PLANAR_FRAME_SPECS = {
    "ARG": (("NE", "CZ", "NH1", "NH2"), ("NE",), "NH1"),
    "ASN": (("CG", "OD1", "ND2"), ("CG",), "ND2"),
    "ASP": (("CG", "OD1", "OD2"), ("CG",), "OD1"),
    "GLN": (("CD", "OE1", "NE2"), ("CD",), "NE2"),
    "GLU": (("CD", "OE1", "OE2"), ("CD",), "OE1"),
    "HIS": (("CG", "ND1", "CD2", "CE1", "NE2"), ("CE1",), "NE2"),
    "PHE": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"), ("CZ",), "CD1"),
    "TRP": (
        ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
        ("CH2",),
        "NE1",
    ),
    "TYR": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"), ("OH",), "CD1"),
}


def normalized(vector: np.ndarray) -> tuple[np.ndarray, float] | None:
    length = float(np.linalg.norm(vector))
    if length <= 1e-7:
        return None
    return (vector / length).astype(np.float32), length


def select_altloc(residue) -> None:
    """Match the deterministic HydroMap/PDBFixer alternate-location rule."""

    for atom in residue:
        if not atom.is_disordered() or not hasattr(atom, "child_dict"):
            continue
        choices = atom.child_dict
        if "A" in choices:
            atom.disordered_select("A")
            continue
        selected = max(
            choices,
            key=lambda key: (float(choices[key].get_occupancy() or 0.0), str(key)),
        )
        atom.disordered_select(selected)


def backbone_frame(atom_by_name: dict[str, np.ndarray]) -> np.ndarray | None:
    if not all(name in atom_by_name for name in ("N", "CA", "C")):
        return None
    e1_item = normalized(atom_by_name["C"] - atom_by_name["CA"])
    if e1_item is None:
        return None
    e1 = e1_item[0]
    n_vector = atom_by_name["N"] - atom_by_name["CA"]
    e2_item = normalized(n_vector - np.dot(n_vector, e1) * e1)
    if e2_item is None:
        return None
    e2 = e2_item[0]
    e3_item = normalized(np.cross(e1, e2))
    if e3_item is None:
        return None
    return np.stack((e1, e2, e3_item[0])).astype(np.float32)


def planar_frame(
    atom_by_name: dict[str, np.ndarray],
    residue_name: str,
) -> np.ndarray | None:
    """Return a deterministic (chemical axis, transverse axis, normal) frame."""

    spec = PLANAR_FRAME_SPECS.get(residue_name)
    if spec is None:
        return None
    plane_names, axis_names, secondary_name = spec
    required = (*plane_names, *axis_names, secondary_name)
    if not all(name in atom_by_name for name in required):
        return None
    coordinates = np.stack([atom_by_name[name] for name in plane_names])
    center = coordinates.mean(axis=0)
    _, _, vh = np.linalg.svd(coordinates - center, full_matrices=False)
    normal_item = normalized(vh[-1].astype(np.float32))
    axis_center = np.stack([atom_by_name[name] for name in axis_names]).mean(axis=0)
    axis_item = normalized(axis_center - center)
    if normal_item is None or axis_item is None:
        return None
    normal = normal_item[0]
    axis = axis_item[0]
    # Remove the small out-of-plane component of the chemically directed axis.
    axis_item = normalized(axis - float(np.dot(axis, normal)) * normal)
    if axis_item is None:
        return None
    axis = axis_item[0]
    secondary = atom_by_name[secondary_name] - center
    if float(np.dot(normal, np.cross(axis, secondary))) < 0:
        normal = -normal
    transverse_item = normalized(np.cross(normal, axis))
    if transverse_item is None:
        return None
    return np.stack((axis, transverse_item[0], normal)).astype(np.float32)


def canonical_atoms(residue) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    atom_by_name: dict[str, np.ndarray] = {}
    sidechain = []
    oxygen_names = {
        "O": "O", "OT1": "O", "O1": "O", "OC1": "O",
        "OXT": "OXT", "OT2": "OXT", "O2": "OXT", "OC2": "OXT",
    }
    for atom in residue:
        element = (atom.element or "").strip().upper()
        if element == "H":
            continue
        name = atom.name.strip().upper()
        coordinate = atom.coord.astype(np.float32)
        if name in oxygen_names:
            if element != "O":
                raise ValueError(f"{residue_uid_from_biopdb(residue)}: {name} must be an oxygen atom")
            canonical = oxygen_names[name]
        else:
            canonical = canonical_backbone_atom_name(name, element) or name
        if canonical in atom_by_name:
            raise ValueError(
                f"{residue_uid_from_biopdb(residue)}: ambiguous duplicate atom aliases for {canonical}"
            )
        atom_by_name[canonical] = coordinate
        if canonical not in {"N", "CA", "C", "O", "OXT"}:
            sidechain.append(coordinate)
    return atom_by_name, sidechain


def parse_geometry(pdb_path: Path) -> dict[str, dict[str, np.ndarray | None]]:
    model = PDBParser(QUIET=True).get_structure("", str(pdb_path))[0]
    output: dict[str, dict[str, np.ndarray | None]] = {}
    for residue in model.get_residues():
        if residue.id[0].strip() or "CA" not in residue:
            continue
        select_altloc(residue)
        atom_by_name, sidechain = canonical_atoms(residue)
        frame = backbone_frame(atom_by_name)
        if frame is None or not sidechain:
            continue
        centroid = np.mean(np.stack(sidechain), axis=0).astype(np.float32)
        residue_name = AA_ALIASES.get(
            residue.resname.strip().upper(), residue.resname.strip().upper()
        )
        names = FUNCTIONAL_GROUP_ATOMS.get(residue_name, ())
        functional = None
        if names and all(name in atom_by_name for name in names):
            functional = np.mean(
                np.stack([atom_by_name[name] for name in names]), axis=0
            ).astype(np.float32)
        polar_names = POLAR_TIP_ATOMS.get(residue_name, ())
        polar = None
        if polar_names and all(name in atom_by_name for name in polar_names):
            polar = np.mean(
                np.stack([atom_by_name[name] for name in polar_names]), axis=0
            ).astype(np.float32)
        output[residue_uid_from_biopdb(residue)] = {
            "ca": atom_by_name["CA"],
            "frame": frame,
            "centroid": centroid,
            "functional": functional,
            "polar": polar,
            "planar_frame": planar_frame(atom_by_name, residue_name),
        }
    return output


def node_features(
    residue_uids: list[str],
    geometry: dict[str, dict[str, np.ndarray | None]],
    kind: str,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    output = np.zeros((len(residue_uids), 5), dtype=np.float32)
    directions = np.zeros((len(residue_uids), 3), dtype=np.float32)
    centers = np.zeros((len(residue_uids), 3), dtype=np.float32)
    for index, uid in enumerate(residue_uids):
        item = geometry.get(uid)
        if item is None or item[kind] is None:
            continue
        vector_item = normalized(item[kind] - item["ca"])
        if vector_item is None:
            continue
        direction, length = vector_item
        directions[index] = direction
        centers[index] = item[kind]
        output[index, :3] = item["frame"] @ direction
        output[index, 3] = min(length / 6.0, 1.5)
        output[index, 4] = 1.0
    return torch.from_numpy(output), directions, centers


def radial_basis(distance: np.ndarray, bins: int = 8) -> np.ndarray:
    centers = np.linspace(2.0, 14.0, bins, dtype=np.float32)
    return np.exp(-0.5 * ((distance[:, None] - centers[None]) / 2.0) ** 2).astype(
        np.float32
    )


def edge_features(
    source: np.ndarray,
    target: np.ndarray,
    ca: np.ndarray,
    node: torch.Tensor,
    directions: np.ndarray,
    centers: np.ndarray,
) -> torch.Tensor:
    output = np.zeros((len(source), 12), dtype=np.float32)
    valid = node[:, 4].numpy().astype(bool)
    pair_valid = valid[source] & valid[target]
    if not np.any(pair_valid):
        return torch.from_numpy(output)
    delta = ca[target] - ca[source]
    delta_norm = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(delta_norm[:, None], 1e-8)
    group_distance = np.linalg.norm(centers[target] - centers[source], axis=1)
    selected = np.flatnonzero(pair_valid)
    output[selected, :8] = radial_basis(group_distance[selected])
    output[selected, 8] = np.sum(
        directions[source[selected]] * direction[selected], axis=1
    )
    output[selected, 9] = np.sum(
        directions[target[selected]] * -direction[selected], axis=1
    )
    output[selected, 10] = np.sum(
        directions[source[selected]] * directions[target[selected]], axis=1
    )
    output[selected, 11] = 1.0
    return torch.from_numpy(output)


def inbound_neighbors(
    ca: np.ndarray, maximum: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(ca)
    query = min(maximum + 1, count)
    _, indices = cKDTree(ca).query(ca, k=query)
    if indices.ndim == 1:
        indices = indices[:, None]
    source, target, rank = [], [], []
    for focal, candidates in enumerate(indices):
        selected = [
            int(value) for value in np.atleast_1d(candidates) if int(value) != focal
        ]
        for neighbor_rank, neighbor in enumerate(selected[:maximum], start=1):
            source.append(neighbor)
            target.append(focal)
            rank.append(neighbor_rank)
    return (
        np.asarray(source, dtype=np.int64),
        np.asarray(target, dtype=np.int64),
        np.asarray(rank, dtype=np.int64),
    )
