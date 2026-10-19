"""Output contract, streaming serialization and failure safety."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from FastHydroMap import Predictor
from FastHydroMap.cli import main
from FastHydroMap.v2.trajectory import TrajectoryArchive

PDB = Path(__file__).parent / "data/v2_chain.pdb"
TARGETS = ["Fdewet", "PC1", "PC2", "PC3"]


@pytest.mark.parametrize("command", ["predict", "predict-trajectory"])
@pytest.mark.parametrize("extension", ["dcd", "xtc"])
def test_opt_in_exports(monkeypatch, tmp_path, command, extension):
    import mdtraj as md
    frame = md.load_pdb(str(PDB))
    traj = md.join([frame, frame])
    traj.time = np.array([0., 2.])
    path = tmp_path / ("input." + extension)
    traj.save(str(path))
    root = tmp_path / "maps"
    monkeypatch.setattr("sys.argv", ["fasthydromap", command, str(PDB), str(path),
                                      "--csv", "--embeddings", "-o", str(root)])
    main()
    assert {p.name for p in tmp_path.glob("maps*")} == {"maps.npz", "maps.json", "maps_total.csv"}
    with np.load(str(root) + ".npz", allow_pickle=False) as archive:
        assert archive["scores"].shape == (2, 73, 4)
        assert archive["embedding"].shape == (2, 73, 16)
        assert archive["embedding_seed"] == 17
        assert archive["frame"].tolist() == [0, 1]
        expected_time = md.load(str(path), top=str(PDB)).time
        np.testing.assert_allclose(archive["time_ps"], expected_time)
        if extension == "dcd":  # XTC compression can change coordinates slightly.
            reference = Predictor().predict(PDB, embeddings=True)
            np.testing.assert_allclose(archive["scores"][0], reference[TARGETS], atol=1e-5, rtol=0)
            np.testing.assert_allclose(archive["embedding"][0],
                reference[[f"embedding_{j}" for j in range(16)]], atol=1e-5, rtol=0)
        csv = pd.read_csv(str(root) + "_total.csv")
        for i, target in enumerate(TARGETS):
            np.testing.assert_allclose(csv.filter(regex="^" + target + ":").to_numpy(),
                                       archive["scores"][:, :, i], atol=1e-6)
        meta = json.loads(Path(str(root) + ".json").read_text())
        assert meta["embeddings_exported"] and meta["trajectory_format"]["csv_exported"]
        for name in archive.files:
            assert not archive[name].dtype.hasobject


def test_writer_multiple_frames_and_identity_guard(tmp_path):
    writer = TrajectoryArchive(tmp_path, TARGETS)
    table = pd.DataFrame({"res_uid": ["A:1", "B:2A"], **{t: [1., 2.] for t in TARGETS}})
    for i in range(100):
        writer.append(table, i * .5)
    assert writer.count == 100
    writer.finish(tmp_path / "scores.npz")
    with np.load(tmp_path / "scores.npz", allow_pickle=False) as z:
        assert z["scores"].shape == (100, 2, 4)
        assert z["scores"].dtype == np.float32
        assert z["time_ps"][-1] == 49.5
        assert z["res_uid"].tolist() == ["A:1", "B:2A"]
    with pytest.raises(ValueError, match="identities changed"):
        writer.append(table.iloc[::-1], 50.)


@pytest.mark.parametrize("failure", ["empty", "second_frame"])
def test_failed_run_preserves_existing_outputs(monkeypatch, tmp_path, failure):
    import mdtraj as md
    from FastHydroMap.v2 import command
    root = tmp_path / "maps"
    for suffix in (".npz", ".json", "_total.csv"):
        Path(str(root) + suffix).write_bytes(b"existing output")
    block = md.load_pdb(str(PDB))
    monkeypatch.setattr(md, "iterload", lambda *a, **k: iter([] if failure == "empty" else [block, block]))
    original = command.Predictor.predict
    calls = 0
    def predict(self, *a, **k):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("simulated prediction failure")
        return original(self, *a, **k)
    monkeypatch.setattr(command.Predictor, "predict", predict)
    monkeypatch.setattr("sys.argv", ["fasthydromap", "predict-trajectory", str(PDB),
        str(tmp_path / "fake.dcd"), "--csv", "--overwrite", "-o", str(root)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert len(list(tmp_path.iterdir())) == 3
    assert all(p.read_bytes() == b"existing output" for p in tmp_path.iterdir())


@pytest.mark.parametrize("option", ["--csv", "--embeddings"])
def test_legacy_rejects_new_trajectory_options_before_loading(monkeypatch, option):
    monkeypatch.setattr("FastHydroMap.cli._load_predictor_or_exit", lambda: pytest.fail("must reject first"))
    monkeypatch.setattr("sys.argv", ["fasthydromap", "predict-trajectory", str(PDB),
                                      "fake.dcd", "--model", "v0", option])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


@pytest.mark.parametrize("case", ["existing", "collision", "directory"])
def test_archive_destinations_checked_before_predictor(monkeypatch, tmp_path, case):
    from FastHydroMap.v2 import command
    root = tmp_path / "maps"
    archive = tmp_path / "maps.npz"
    if case == "directory":
        archive.mkdir()
    else:
        archive.write_bytes(b"do not replace")
    args = ["fasthydromap", "predict-trajectory", str(PDB),
            str(archive if case == "collision" else tmp_path / "missing.dcd"), "-o", str(root)]
    if case != "existing":
        args.append("--overwrite")
    monkeypatch.setattr(command, "Predictor", lambda *a, **k: pytest.fail("must reject first"))
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert len(list(tmp_path.iterdir())) == 1
    if case != "directory":
        assert archive.read_bytes() == b"do not replace"


def test_structure_csv_flag_rejected_early(monkeypatch, tmp_path):
    monkeypatch.setattr("FastHydroMap.cli._load_predictor_or_exit", lambda: pytest.fail("must reject first"))
    monkeypatch.setattr("sys.argv", ["fasthydromap", "predict", str(PDB), "--csv", "-o", str(tmp_path / "maps")])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2 and not list(tmp_path.iterdir())
