from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch
from types import SimpleNamespace
from Bio.PDB import PDBParser, PDBIO
from FastHydroMap.v2 import Predictor
from FastHydroMap.cli import main
from FastHydroMap.v2.model import SharedWaterModel
from FastHydroMap.v2.chemistry import NAMES

DATA = Path(__file__).parent / "data"
PDB = DATA / "v2_chain.pdb"


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def test_selected_model_regression_and_single_head(predictor):
    with pytest.warns(UserWarning, match="HIE"):
        got = predictor.predict(PDB)
    expected = pd.DataFrame(json.loads((DATA/'v2_shared_chain_expected.json').read_text())['records'])
    assert got.res_uid.tolist() == expected.res_uid.tolist()
    for target in predictor.quantities:
        np.testing.assert_allclose(got[target], expected[target], atol=5e-6, rtol=0)
    with pytest.raises(ValueError, match="intrinsic/context"):
        predictor.predict(PDB, parts=True)
    assert got.terminal_caution.equals(got.is_n_term | got.is_c_term)
    assert got.zero_sasa_caution.equals(got.sasa_A2 <= 0)
    from FastHydroMap.v2.sasa import compute_sasa
    np.testing.assert_allclose(got.sasa_A2, 100 * compute_sasa(PDB, len(got)).sum(1), atol=1e-5, rtol=1e-6)
    assert got.no_probe_caution.equals(got.probe_count == 0)
    assert not got.assembly_caution.any()
    assert not any("intrinsic" in c or "context" in c for c in got.columns)


def test_histidine_validation_and_charge(predictor):
    base = predictor.predict(PDB)
    hist = base.loc[base.aa.eq("H"), "res_uid"].tolist()
    charged = predictor.predict(PDB, histidine_states={uid: "HIP" for uid in hist})
    assert charged.loc[charged.aa.eq("H"), "histidine_state"].eq("HIP").all()
    assert not np.allclose(base.Fdewet, charged.Fdewet)
    with pytest.raises(ValueError, match="unknown or non-histidine"):
        predictor.predict(PDB, histidine_states={"X:99999": "HIP"})
    with pytest.raises(ValueError, match="HID, HIE or HIP"):
        predictor.predict(PDB, histidine_states={hist[0]: "auto"})


def test_compact_unused_features_and_disconnected_nodes():
    torch.manual_seed(17)
    model = SharedWaterModel().eval()
    assert len(model.layers) == 2
    assert sum(p.numel() for p in model.parameters()) == 8740
    assert not hasattr(model, "intrinsic_head")
    n = 8
    g = SimpleNamespace(
        x=torch.randn(n, 33), point=torch.randn(n, 24, 32),
        probe_atom=torch.randn(n, 24, 8, 21), point_mask=torch.ones(n, 24, dtype=torch.bool),
        graph_edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        graph_edge_attr=torch.randn(4, 20),
        **{"rich_" + k: torch.randn(n, len(v)) for k, v in NAMES.items()},
    )
    changed = copy.deepcopy(g)
    changed.point[..., 11:] = 999
    changed.probe_atom[..., [2, 3]] = 999
    changed.rich_cavity[:, 1:-1] = 999
    for bank in ("angular", "field", "atom_env"):
        getattr(changed, "rich_" + bank)[:] = 999
    changed.target = torch.full((n,), 999.)
    changed.avg_n_waters = torch.full((n,), 999.)
    with torch.inference_mode():
        expected = model(model.inputs(g))
        torch.testing.assert_close(model(model.inputs(changed)), expected, rtol=0, atol=0)
        # Nodes 4..7 have no edges to 0..3. Their embeddings cannot alter the readout.
        changed.x[4:] = 999
        changed.point[4:] = 999
        changed.probe_atom[4:] = 999
        torch.testing.assert_close(model(model.inputs(changed))[:4], expected[:4], rtol=0, atol=0)


def test_compact_graph_degree_and_distance_cap():
    from FastHydroMap.v2.graph import core
    from FastHydroMap.io.residue_qc import collect_residue_records
    from dataclasses import asdict
    records, _, _ = collect_residue_records(PDB)
    rows = [dict(asdict(r), histidine_charged=0) for r in records]
    g = core(PDB, rows)
    source, target = g.graph_edge_index
    assert len(g.x) == len(records)  # Every residue is context, not only surface sites.
    assert torch.bincount(target, minlength=len(records)).max() <= 24
    assert torch.all(torch.linalg.vector_norm(g.pos[source].double() - g.pos[target].double(), dim=1) <= 12)


def test_disconnected_component_does_not_change_predictions(predictor, tmp_path):
    model = PDBParser(QUIET=True).get_structure("", str(PDB))[0]
    extra = copy.deepcopy(model["A"])
    extra.id = "B"
    for atom in extra.get_atoms():
        atom.coord = atom.coord + np.array([150.0, 0.0, 0.0])
    model.add(extra)
    path = tmp_path / "remote.pdb"
    writer = PDBIO()
    writer.set_structure(model)
    writer.save(str(path))
    reference = predictor.predict(PDB)
    with pytest.warns(UserWarning, match="independently"):
        got = predictor.predict(path)
    np.testing.assert_allclose(
        got.loc[got.chain_id.eq("A"), list(predictor.quantities)],
        reference[list(predictor.quantities)],
        atol=5e-6,
        rtol=0,
    )


def test_incomplete_atoms_fail_clearly(predictor, tmp_path):
    model = PDBParser(QUIET=True).get_structure("", str(PDB))
    next(model.get_residues()).detach_child("CB")
    path = tmp_path / "missing.pdb"
    writer = PDBIO()
    writer.set_structure(model)
    writer.save(str(path))
    with pytest.raises(ValueError, match="Incomplete or unsupported"):
        predictor.predict(path)


def test_cli_all_and_overwrite_guard(monkeypatch, tmp_path):
    root = tmp_path / "nested" / "prediction"
    argv = [
        "fasthydromap",
        "predict",
        str(PDB),
        "--model",
        "v2",
        "--quantity",
        "all",
        "-o",
        str(root),
    ]
    monkeypatch.setattr("sys.argv", argv)
    main()
    frame = pd.read_csv(str(root) + ".csv")
    assert len(frame) == 73
    assert json.loads(Path(str(root) + ".json").read_text())["model"] == "latest"
    for target in ("fdewet", "pc1", "pc2", "pc3"):
        assert Path(str(root) + "_" + target + ".pdb").exists()
    before = Path(str(root) + ".csv").read_bytes()
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert Path(str(root) + ".csv").read_bytes() == before


def test_invalid_options():
    with pytest.raises(ValueError, match="positive integer"):
        Predictor(threads=0)
    with pytest.raises(ValueError, match="Choose"):
        Predictor(quantities=["other"])


def test_named_histidine_and_insertion_code(predictor, tmp_path):
    model = PDBParser(QUIET=True).get_structure("", str(PDB))
    his = next(r for r in model.get_residues() if r.resname == "HIS")
    his.resname = "HIP"
    old_id = his.id
    his.id = (old_id[0], old_id[1], "A")
    path = tmp_path / "named.pdb"
    writer = PDBIO()
    writer.set_structure(model)
    writer.save(str(path))
    got = predictor.predict(path)
    row = got.loc[got.res_uid.eq(f"A:{old_id[1]}A")].iloc[0]
    assert row.histidine_state == "HIP"


def test_streaming_trajectory(monkeypatch, tmp_path, predictor):
    import mdtraj as md

    frame = md.load_pdb(str(PDB))
    trajectory = md.join([frame, frame])
    dcd = tmp_path / "two.dcd"
    trajectory.save_dcd(str(dcd))
    root = tmp_path / "trajectory"
    monkeypatch.setattr(
        "sys.argv",
        [
            "fasthydromap",
            "predict-trajectory",
            str(PDB),
            str(dcd),
            "--model",
            "v2",
            "-o",
            str(root),
        ],
    )
    main()
    total = np.load(str(root) + ".npz", allow_pickle=False)["scores"]
    assert not Path(str(root) + "_fdewet.pdb").exists()
    assert not Path(str(root) + "_pc3.pdb").exists()
    assert not Path(str(root) + "_total.csv").exists()
    assert total.shape == (2, 73, 4)
    np.testing.assert_allclose(total[0], total[1], atol=1e-6, rtol=0)
    assert not Path(str(root) + "_intrinsic.csv").exists()
    assert not Path(str(root) + "_context.csv").exists()
    reference = predictor.predict(PDB)[list(predictor.quantities)].to_numpy()
    np.testing.assert_allclose(total[0], reference, atol=1e-5, rtol=0)
