"""
Checks that `featurize.graph.build_graph` returns
• node dim = 32
• edge_attr dim = 84
"""
import numpy as np, pandas as pd, torch
from pathlib import Path
from FastHydroMap.featurize import graph as fg
from FastHydroMap.utils.constants import AA2IDX

CSV  = Path(__file__).parent / "data" / "1A1U_features.csv"
PDB  = Path(__file__).parent / "data" / "1A1U.pdb"

def test_graph_shapes():
    df = pd.read_csv(CSV).set_index(["pdb_id", "resid"])
    g  = fg.build_graph(PDB, df.loc["1A1U"], k_nn=12, pid_ix=0)
    assert g.x.shape[1]        == 32
    assert g.edge_attr.shape[1]== 84
    assert g.edge_index.shape[0]==2


def test_graph_uses_true_one_letter_amino_acid_code():
    df = pd.read_csv(CSV).set_index(["pdb_id", "resid"])
    g = fg.build_graph(PDB, df.loc["1A1U"], k_nn=12, pid_ix=0)
    first_aa_idx = int(torch.argmax(g.x[0, :20]).item())
    assert first_aa_idx == AA2IDX["E"]
