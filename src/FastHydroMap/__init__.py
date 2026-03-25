from importlib import resources
from pathlib import Path

# from .predictors.fdewet import FdewetPredictor   # public API
# __all__ = ["FdewetPredictor"]


def selftest(verbose: bool = True) -> None:
    """
    Minimal sanity check:
      1. loads a built-in 2-residue PDB,
      2. runs SASA,
      3. prints a success banner.
    """
    from .io.pdb import load_traj
    from .featurize.sasa import residue_sasa
    pdb_path = Path(resources.files("FastHydroMap")) / "tests" / "ala_gly.pdb"
    traj = load_traj(pdb_path)
    sasa = residue_sasa(traj)
    if verbose:
        print("✓ mdtraj SASA OK —", [round(x, 2) for x in sasa])
    if verbose:
        print("🎉 FastHydroMap self-test passed")
