from __future__ import annotations
from pathlib import Path
import os, tempfile, atexit

import numpy as np
import mdtraj as md
from pdbfixer import PDBFixer
from openmm.app import Modeller, element
from openmm import unit as u
from Bio.PDB import PDBParser, PDBIO

from .residue_qc import collect_residue_records, residue_uid

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

_TEMP_PDBS: list[str] = []  # collect tmp paths for cleanup at exit

def _tmp_noH_pdb(original: Path) -> Path:
    """Return a *temporary* PDB file identical to *original* but **without hydrogens**.

    The file is created once per call and deleted automatically when the Python
    interpreter exits.  Keeping the fixer input hydrogen-free is a defensive
    choice against inconsistent hydrogen records across user-provided PDBs.
    """
    fd, tmp = tempfile.mkstemp(prefix="noH_", suffix=".pdb")
    os.close(fd)  # we will reopen with Python I/O

    with open(original, "r") as fin, open(tmp, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                # element column is 77‑78 (1‑indexed) per PDB spec
                element_sym = line[76:78].strip().upper()
                atom_name   = line[12:16].strip()  # fallback
                if element_sym == "H" or atom_name.startswith("H"):
                    continue  # ✂ drop
            fout.write(line)

    _TEMP_PDBS.append(tmp)
    return Path(tmp)


def _cleanup_tmp_pdbs():
    for p in _TEMP_PDBS:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

atexit.register(_cleanup_tmp_pdbs)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_traj(
    pdb_path: Path,
    dcd_path: Path | None = None,
    drop_hydrogens: bool = False,
) -> md.Trajectory:
    """Load a PDB (+ optional trajectory such as DCD/XTC) with stable preprocessing.

    FastHydroMap featurization is heavy-atom-only, so hydrogens are ignored
    downstream even when present in topology. ``drop_hydrogens=True`` remains
    available as a defensive fallback for unusual hydrogen records.
    """

    if dcd_path is None:
        # ── single‑structure path ───────────────────────────────────────────
        pdb_for_fixer = _tmp_noH_pdb(pdb_path) if drop_hydrogens else pdb_path
        fixer = PDBFixer(filename=str(pdb_for_fixer))

        # Minimal fixer pipeline.
        fixer.findMissingResidues()
        # Do not invent residues from SEQRES during inference.
        fixer.missingResidues = {}
        fixer.findNonstandardResidues(); fixer.replaceNonstandardResidues()
        # Keep inference focused on polymer residues only.
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms(); fixer.addMissingAtoms()
        modeller = Modeller(fixer.topology, fixer.positions)
        xyz_nm = np.asarray(modeller.positions.value_in_unit(u.nanometer), dtype=np.float32)
        top = md.Topology.from_openmm(modeller.topology)
        traj = md.Trajectory(xyz_nm[None, ...], top)

    else:
        # ── PDB topology + external trajectory coordinates ───────────────────
        try:
            traj = md.load(str(dcd_path), top=str(pdb_path))
        except Exception as exc:
            raise ValueError(
                f"failed to load trajectory '{dcd_path}' with topology '{pdb_path}'. "
                "Ensure atom count and atom order match exactly between topology and "
                "trajectory (common mismatch for DCD/XTC files)."
            ) from exc

    print(f"frames: {traj.n_frames} | residues: {traj.n_residues}")
    return traj


def load_pdb(
    pdb_path: Path,
    drop_hydrogens: bool = False,
) -> md.Trajectory:
    """Backwards-compatible alias for loading a single-structure PDB."""
    return load_traj(pdb_path, dcd_path=None, drop_hydrogens=drop_hydrogens)

# -----------------------------------------------------------------------------
# Utility: write B‑factors
# -----------------------------------------------------------------------------

def write_bfactor(
    pdb_in: Path | str,
    values: np.ndarray,
    pdb_out: Path | str,
) -> None:
    """Inject per‑residue ``values`` into the B‑factor column of ``pdb_in``.

    *Only* residues that contain at least one heavy atom are counted so indices
    stay consistent when input files list hydrogens separately.
    """

    struct = PDBParser(QUIET=True).get_structure("struct", str(pdb_in))
    residue_records, _, _ = collect_residue_records(pdb_in)

    if len(values) != len(residue_records):
        raise ValueError(
            f"{len(residue_records)} protein-like residues in PDB but {len(values)} values provided"
        )

    value_by_uid = {record.res_uid: float(val) for record, val in zip(residue_records, values)}

    assigned = 0
    for res in struct.get_residues():
        if res.id[0].strip():
            continue
        uid = residue_uid(res.get_parent().id, res.id[1], res.id[2])
        if uid not in value_by_uid:
            continue
        for atom in res.get_atoms():
            atom.set_bfactor(value_by_uid[uid])
        assigned += 1

    if assigned != len(value_by_uid):
        raise ValueError(
            f"assigned B-factors to {assigned} residues, expected {len(value_by_uid)}"
        )

    io = PDBIO(); io.set_structure(struct); io.save(str(pdb_out))
    print(f"✓ B‑factors written → {pdb_out}")
