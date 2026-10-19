from pathlib import Path
import json
from functools import lru_cache
import numpy as np
from Bio.PDB import PDBParser


@lru_cache(maxsize=1)
def templates():
    data = json.loads((Path(__file__).parent / "chemistry.json").read_text())
    # The source force field names the two terminal oxygens OC1/OC2.
    # Canonicalize names only; retain every atom and its original parameters.
    normalized = {}
    for name, atoms in data["templates"].items():
        renamed = { {"OC1": "O", "OC2": "OXT"}.get(atom, atom): values
                    for atom, values in atoms.items() }
        if len(renamed) != len(atoms):
            raise ValueError(f"Ambiguous terminal oxygen names in template {name}")
        normalized[name] = renamed
    return normalized, data["water"]


RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}

GEOMETRY = [
    "log_volume",
    "bulk_connected_fraction",
    "log_disconnected_volume",
    "clearance_mean",
    "clearance_std",
    "clearance_q10",
    "clearance_q90",
    "own_distance_mean",
    "own_distance_std",
    "inner_fraction",
    "cov_eigen_small",
    "cov_eigen_mid",
    "cov_eigen_large",
    "extent_rms",
    "centroid_offset",
    "neighbor_volume_r2_mean",
    "neighbor_volume_r2_std",
    "neighbor_volume_r35_mean",
    "neighbor_volume_r35_std",
    "nearest_C_fraction",
    "nearest_N_fraction",
    "nearest_O_fraction",
    "nearest_S_fraction",
    "empty_shell",
]

ANGULAR = [
    "allowed_fraction_mean",
    "allowed_fraction_std",
    "anisotropy_mean",
    "anisotropy_std",
    "P1_mean",
    "P2_mean",
    "P3_mean",
    "angle_40_50_capacity",
    "angle_100_120_capacity",
    "angle_140_150_capacity",
    "angular_capacity_std",
    "empty_angular_samples",
]

FIELD_BASE = [
    "potential",
    "field_norm",
    "lj_oxygen",
    "donor_density",
    "acceptor_density",
    "polar_balance",
    "field_alignment",
]

FIELD = [f"{n}_{s}" for n in FIELD_BASE for s in ("mean", "std", "q10", "q90")] + [
    "own_charge",
    "shell_charge",
    "shell_abs_charge",
    "mean_donor_acceptor_product",
]

CHEM_TYPES = [
    "aliphatic_C",
    "aromatic_C",
    "donor",
    "acceptor",
    "sulfur",
    "united_charge",
    "abs_charge",
    "backbone",
    "positive_charge",
    "negative_charge",
]

ATOM_ENV = [f"r{r}_{n}" for r in (3, 5, 8, 12) for n in CHEM_TYPES]

NAMES = dict(cavity=GEOMETRY, angular=ANGULAR, field=FIELD, atom_env=ATOM_ENV)


def parse(path, uids, histidine_states):
    # Imported here so feature workers need no graph/torch imports.
    from .geometry import canonical_atoms, select_altloc, backbone_frame
    from .geometry import residue_uid_from_biopdb
    from ..io.residue_qc import _peptide_bond_present

    model = PDBParser(QUIET=True).get_structure("", str(path))[0]
    if len(uids) != len(histidine_states):
        raise ValueError("Each residue must have a matching histidine-state entry")
    caps = {"ACE", "NME", "NHE", "NH2"}
    if any(r.resname in caps for r in model.get_residues()):
        raise ValueError("Capped peptides are unsupported; supply uncapped standard protein residues")
    lookup = {
        residue_uid_from_biopdb(r): r
        for r in model.get_residues()
        if not r.id[0].strip() and "CA" in r
    }
    ts, water = templates()
    res = []
    for uid in uids:
        r = lookup[uid]
        select_altloc(r)
        a, _ = canonical_atoms(r)
        missing = {"N", "CA", "C", "O"} - a.keys()
        if missing:
            raise ValueError(f"{uid}: missing backbone heavy atoms {sorted(missing)}")
        res.append((r, a))
    sg = [
        (i, a["SG"]) for i, (r, a) in enumerate(res) if r.resname == "CYS" and "SG" in a
    ]
    ss = set()
    for i, (ri, xi) in enumerate(sg):
        for rj, xj in sg[i + 1 :]:
            if np.linalg.norm(xi - xj) < 2.5:
                ss.update((ri, rj))
    xyz = []
    props = []
    owner = []
    charge_checks = []
    frames = []
    centers = []
    for i, ((residue, atoms), state) in enumerate(zip(res, histidine_states)):
        name = residue.resname
        if name in ("HIS", "HID", "HIE", "HIP"):
            name = state if state in ("HID", "HIE", "HIP") else "HIE"
        if i in ss:
            name = "CYX"
        prev = (
            i > 0
            and res[i - 1][0].parent.id == residue.parent.id
            and _peptide_bond_present(res[i - 1][0], residue)
        )
        nxt = (
            i < len(res) - 1
            and res[i + 1][0].parent.id == residue.parent.id
            and _peptide_bond_present(residue, res[i + 1][0])
        )
        if not prev and not nxt:
            raise ValueError(
                f"{uids[i]}: singleton/disconnected residue requires a combined N/C-terminal "
                "template, which is not available"
            )
        if "OXT" in atoms and nxt:
            raise ValueError(f"{uids[i]}: OXT is incompatible with a following peptide bond")
        if not nxt:
            if "OXT" not in atoms:
                raise ValueError(
                    f"{uids[i]}: C-terminal OXT is missing; use --prepare for a single "
                    "structure. Trajectory topology must already contain terminal oxygens"
                )
            name = "C" + name
        elif not prev:
            name = "N" + name
        if name not in ts:
            raise ValueError(f"{uids[i]}: unsupported terminal chemistry {name}")
        template = ts[name]
        if set(atoms) != set(template):
            raise ValueError(
                f"{uids[i]} {name}: unexpected or missing heavy atoms {sorted(set(atoms)^set(template))}. Use --prepare for missing atoms."
            )
        frame = backbone_frame(atoms)
        if frame is None:
            raise ValueError(f"{uids[i]}: missing or degenerate backbone frame")
        frames.append(frame)
        centers.append(atoms["CA"])
        charge_checks.append(
            dict(
                uid=uids[i],
                template=name,
                charge=sum(a["charge"] for a in template.values()),
            )
        )
        for key, coord in atoms.items():
            a = template[key]
            el = a["element"]
            q = a["charge"]
            donor = el in ("N", "O") and a["hydrogens"] > 0
            accept = (
                el == "O" or (el == "N" and a["type"] in ("NB", "NC")) or (el == "S")
            )
            aromatic = el == "C" and a["type"] not in (
                "CT",
                "C1",
                "C3",
                "C4",
                "C7",
                "C8",
                "C9",
                "C",
            )
            xyz.append(coord)
            owner.append(i)
            props.append(
                [
                    RADII[el],
                    q,
                    a["sigma"],
                    a["epsilon"],
                    donor,
                    accept,
                    el == "C",
                    el == "N",
                    el == "O",
                    el == "S",
                    el == "C" and not aromatic,
                    aromatic,
                    key in ("N", "CA", "C", "O", "OXT"),
                ]
            )
    return (
        np.asarray(xyz, dtype=float),
        np.asarray(props, dtype=float),
        np.asarray(owner),
        np.asarray(frames),
        np.asarray(centers),
        charge_checks,
        water,
    )


def clearance(points, tree, radii):
    d, idx = tree.query(points, k=min(8, len(radii)), workers=1)
    return np.min(d - radii[idx], axis=-1), idx[:, 0]


def summarize(values):
    return np.column_stack(
        (
            values.mean(0),
            values.std(0),
            np.quantile(values, 0.1, axis=0),
            np.quantile(values, 0.9, axis=0),
        )
    ).ravel()
