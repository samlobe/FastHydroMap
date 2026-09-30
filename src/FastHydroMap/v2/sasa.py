from pathlib import Path
import numpy as np
import mdtraj as md
from ..utils.constants import AA3
from ..featurize.sasa import residue_sasa_components

STANDARD_RESNAMES = set(AA3)


def compute_sasa(source: Path, expected: int) -> np.ndarray:
    traj = md.load_pdb(str(source), standard_names=True)
    # Match collect_residue_records/build_graph exactly: a polymer residue is
    # graphable only when it has a CA.  MDTraj's broader ``protein`` selection
    # also includes incomplete polymer stubs, which would shift every label
    # after the stub by one residue.
    graphable_residues = [
        residue
        for residue in traj.topology.residues
        if residue.is_protein
        and residue.name.strip().upper() in STANDARD_RESNAMES
        and any(atom.name.strip() == "CA" for atom in residue.atoms)
    ]
    protein_atoms = np.asarray(
        [atom.index for residue in graphable_residues for atom in residue.atoms],
        dtype=np.int64,
    )
    if len(protein_atoms) == 0:
        raise ValueError(f"{source}: MDTraj found no protein atoms")
    protein = traj.atom_slice(protein_atoms)
    components = residue_sasa_components(protein, strict=True)
    if len(components) != expected:
        raise ValueError(
            f"{source}: {len(components)} SASA residues != {expected} label rows"
        )
    return np.asarray(components, dtype=np.float32)
