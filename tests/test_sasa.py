"""
pytest for the SASA 
"""

from pathlib import Path
import numpy as np
import pytest
import mdtraj as md

from FastHydroMap.io.pdb import load_pdb
from FastHydroMap.featurize.sasa import residue_sasa, residue_sasa_components

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"
TOY_PDB = DATA_DIR / "ala_gly.pdb"
PROTEIN_PDB = DATA_DIR / "1A1U.pdb"

@pytest.fixture(scope="session")
def mini_traj():
    """Return a one-frame trajectory for a 2-residue peptide (ala-gly)."""
    return load_pdb(TOY_PDB)

def test_sasa_shape(mini_traj):
    sasa = residue_sasa(mini_traj)
    assert sasa.shape == (mini_traj.n_residues,)


def test_sasa_values(mini_traj):
    """Heavy-atom-only numeric check against reference computed once in MDTraj."""
    sasa_ref = np.array([1.6479776, 1.4086473], dtype=np.float32)
    sasa      = residue_sasa(mini_traj)
    assert np.allclose(sasa, sasa_ref, atol=0.1)


def test_sasa_components_shape_and_partition(mini_traj):
    comp, other = residue_sasa_components(mini_traj, return_other=True)
    assert comp.shape == (mini_traj.n_residues, 7)
    assert other.shape == (mini_traj.n_residues,)
    assert np.allclose(other, 0.0)

    heavy = mini_traj.atom_slice(
        [atom.index for atom in mini_traj.topology.atoms if atom.element.symbol != "H"]
    )
    heavy_res_sasa = md.shrake_rupley(heavy, mode="residue")[0].astype(np.float32)
    assert np.allclose(comp.sum(axis=1), heavy_res_sasa, atol=1e-6)


def test_sasa_components_match_simple_chemistry(mini_traj):
    comp = residue_sasa_components(mini_traj)
    ala, gly = comp

    # ALA has only CB beyond the backbone; GLY has no sidechain heavy atoms.
    assert ala[4] > 0.0
    assert np.isclose(ala[5], 0.0)
    assert np.isclose(ala[6], 0.0)
    assert np.isclose(gly[4], 0.0)
    assert np.isclose(gly[5], 0.0)
    assert np.isclose(gly[6], 0.0)


def test_sasa_components_real_protein_sanity():
    traj = md.load(str(PROTEIN_PDB))
    comp, other = residue_sasa_components(traj, strict=True, return_other=True)

    assert comp.shape == (traj.n_residues, 7)
    assert np.allclose(other, 0.0)

    # GLU26 has sidechain carbons and sidechain oxygens.
    assert comp[0, 5] > 0.0
    assert comp[0, 6] > 0.0
    # PHE28 has aromatic sidechain carbons but no sidechain N/O/S.
    assert comp[2, 5] > 0.0
    assert np.isclose(comp[2, 6], 0.0)


def test_sasa_components_handles_terminal_oxygen_aliases(tmp_path):
    alias_pdb = tmp_path / "1A1U_ot1_ot2.pdb"
    lines = []
    for line in PROTEIN_PDB.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atom_name = line[12:16]
            if atom_name == " O  ":
                # Swap canonical O names to terminal aliases.
                atom_serial = int(line[6:11])
                replacement = f"{'OT1' if atom_serial % 2 else 'OT2':>4}"
                line = f"{line[:12]}{replacement}{line[16:]}"
        lines.append(line)
    alias_pdb.write_text("\n".join(lines) + "\n")

    traj = md.load(str(alias_pdb))
    comp, other = residue_sasa_components(traj, strict=True, return_other=True)
    assert comp.shape == (traj.n_residues, 7)
    assert np.allclose(other, 0.0)
