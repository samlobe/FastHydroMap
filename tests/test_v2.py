from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import pytest
from Bio.PDB import PDBParser, PDBIO
from FastHydroMap.v2 import Predictor
from FastHydroMap.cli import main

DATA = Path(__file__).parent / "data"
PDB = DATA / "v2_chain.pdb"


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def test_selected_model_regression_and_parts(predictor):
    with pytest.warns(UserWarning, match="HIE"):
        got = predictor.predict(PDB, parts=True)
    expected = pd.read_json(DATA / "v2_expected.json")
    assert got.res_uid.tolist() == expected.uid.tolist()
    for target in predictor.quantities:
        np.testing.assert_allclose(got[target], expected[target], atol=5e-6, rtol=0)
        np.testing.assert_allclose(
            got[target],
            got[target + "_intrinsic"] + got[target + "_context"],
            atol=5e-6,
            rtol=0,
        )


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
        "--parts",
        "-o",
        str(root),
    ]
    monkeypatch.setattr("sys.argv", argv)
    main()
    frame = pd.read_csv(str(root) + ".csv")
    assert len(frame) == 73
    assert json.loads(Path(str(root) + ".json").read_text())["model"] == "v2"
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
            "--parts",
            "-o",
            str(root),
        ],
    )
    main()
    total = pd.read_csv(str(root) + ".csv").drop(columns="frame").to_numpy()
    intrinsic = (
        pd.read_csv(str(root) + "_intrinsic.csv").drop(columns="frame").to_numpy()
    )
    context = pd.read_csv(str(root) + "_context.csv").drop(columns="frame").to_numpy()
    assert total.shape == (2, 73)
    np.testing.assert_allclose(total[0], total[1], atol=1e-6, rtol=0)
    np.testing.assert_allclose(total, intrinsic + context, atol=5e-6, rtol=0)
    reference = predictor.predict(PDB).Fdewet.to_numpy()
    np.testing.assert_allclose(total[0], reference, atol=1e-5, rtol=0)
