"""
Load final MPNN weights and do a single forward pass
on the tiny graph from the previous test.
"""
import torch, pandas as pd
from pathlib import Path
from FastHydroMap.models.mpnn import FdewetMPNN
from FastHydroMap.featurize import graph as fg
from importlib.resources import files

W = files("FastHydroMap.weights") / "mpnn_latest.pt"
CSV     = Path(__file__).parent / "data" / "1A1U_features.csv"
PDB     = Path(__file__).parent / "data" / "1A1U.pdb"

def test_mpnn_pass():
    df = pd.read_csv(CSV).set_index(["pdb_id","resid"])
    g  = fg.build_graph(PDB, df.loc["1A1U"], k_nn=12, pid_ix=0)
    model = FdewetMPNN(); model.load_state_dict(torch.load(W, map_location="cpu"))
    out = model(g)
    assert out.shape[0] == g.x.shape[0]


def test_mpnn_parts_sum_to_total():
    df = pd.read_csv(CSV).set_index(["pdb_id","resid"])
    g  = fg.build_graph(PDB, df.loc["1A1U"], k_nn=12, pid_ix=0)
    model = FdewetMPNN()
    parts = model(g, return_parts=True)
    assert set(parts) == {"total", "intrinsic", "context"}
    assert parts["total"].shape[0] == g.x.shape[0]
    torch.testing.assert_close(parts["total"], parts["intrinsic"] + parts["context"])
