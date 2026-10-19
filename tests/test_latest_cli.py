"""Release defaults, compatibility columns and optional embeddings."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from FastHydroMap import Predictor
from FastHydroMap.cli import _build_parser, main
from FastHydroMap.v2.display import PC_DISPLAY_RANGES

PDB = Path(__file__).parent / "data/v2_chain.pdb"


@pytest.mark.parametrize("value, expected", [(None,"latest"),("latest","latest"),("v0","v0"),("v1","v0"),("v2","latest")])
def test_model_selection(value, expected):
    args = ["predict", str(PDB)] + ([] if value is None else ["--model", value])
    assert _build_parser().parse_args(args).model == expected


def test_default_prediction_columns_provenance_and_embeddings(monkeypatch, tmp_path):
    root = tmp_path / "default"
    monkeypatch.setattr("sys.argv", ["fasthydromap","predict",str(PDB),"-o",str(root)])
    main()
    f = pd.read_csv(str(root)+".csv", keep_default_na=False)
    assert list(f.columns[:5]) == ["residue","Fdewet","PC1","PC2","PC3"]
    assert f.res_uid.str.startswith("A:").all()
    assert [str(r) for r in f.residue] == [uid.split(":",1)[1] for uid in f.res_uid]
    meta = json.loads(Path(str(root)+".json").read_text())
    assert meta["model"] == "latest" and meta["model_alias"] == "latest"
    assert meta["training_stage"] == "full800" and meta["fit_proteins"] == 800
    assert meta["feature_version"] == "terminal_carboxylate_v2"
    assert meta["quantities"] == ["Fdewet", "PC1", "PC2", "PC3"]
    assert "terminal_chemistry_note" not in meta
    assert "excluded from fitting targets" in meta["terminal_caution_note"]
    for target in ("fdewet", "pc1", "pc2", "pc3"):
        assert Path(str(root) + "_" + target + ".pdb").is_file()
    assert not Path(str(root) + ".pdb").exists()
    assert meta["display_ranges"] == {k:list(v) for k,v in PC_DISPLAY_RANGES.items()}
    archive = np.load(str(root)+"_embeddings.npz", allow_pickle=False)
    assert archive["embedding"].shape == (len(f),16)
    assert len(list(tmp_path.iterdir())) == 7
    reference = Predictor().predict(PDB)
    np.testing.assert_allclose(f[list(Predictor().quantities)], reference[list(Predictor().quantities)], atol=1e-6, rtol=0)


def test_parts_requires_legacy(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("sys.argv", ["fasthydromap","predict",str(PDB),"--parts","-o",str(tmp_path/"parts")])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2 and "--model v0" in capsys.readouterr().err


def test_multichain_compatibility_labels():
    from FastHydroMap.v2.command import residue_labels
    f = pd.DataFrame(dict(chain_id=["A","B"],resid=[2,2],insertion_code=["","A"],res_uid=["A:2","B:2A"]))
    assert residue_labels(f) == ["A:2","B:2A"]


def test_default_trajectory_filename_and_columns(monkeypatch, tmp_path):
    import mdtraj as md
    trajectory = tmp_path / "frames.dcd"
    md.load_pdb(str(PDB)).save_dcd(str(trajectory))
    monkeypatch.setattr("sys.argv", ["fasthydromap","predict-trajectory",str(PDB),str(trajectory)])
    main()
    root = tmp_path / "frames_maps_traj"
    f = np.load(str(root)+".npz", allow_pickle=False)
    assert f["scores"].shape == (1, 73, 4)
    assert f["targets"].tolist() == ["Fdewet", "PC1", "PC2", "PC3"]
    assert f["frame"].tolist() == [0]
    assert "embedding" not in f.files
    assert {p.name for p in tmp_path.iterdir()} == {
        "frames.dcd", "frames_maps_traj.npz", "frames_maps_traj.json"}
    mapping = json.loads(Path(str(root)+".json").read_text())["residue_ids"]
    assert set(f["res_uid"]) == set(mapping.values())


def test_default_structure_filenames(monkeypatch, tmp_path):
    pdb = tmp_path / "protein.pdb"
    pdb.write_bytes(PDB.read_bytes())
    monkeypatch.setattr("sys.argv", ["fasthydromap", "predict", str(pdb)])
    main()
    assert (tmp_path / "protein_maps.csv").exists()
    assert (tmp_path / "protein_maps_pc1.pdb").exists()
    assert (tmp_path / "protein_maps_fdewet.pdb").exists()


@pytest.mark.parametrize("target", ["fdewet", "pc1", "pc2", "pc3"])
def test_explicit_single_output_gives_migration_message(monkeypatch, tmp_path, target, capsys):
    root = tmp_path / target
    monkeypatch.setattr("sys.argv", ["fasthydromap", "predict", str(PDB),
        "--quantity", target, "-o", str(root)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "always exports all four" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("command", ["predict", "predict-trajectory"])
@pytest.mark.parametrize("model", ["latest", "v0"])
@pytest.mark.parametrize("option", ["--prepare"])
def test_trajectory_options_rejected_before_loading_model_or_writing(
        monkeypatch, tmp_path, capsys, command, model, option):
    def unexpected_load():
        pytest.fail("Invalid trajectory options must fail before loading any predictor")
    monkeypatch.setattr("FastHydroMap.cli._load_predictor_or_exit", unexpected_load)
    root = tmp_path / "not-created" / "result"
    monkeypatch.setattr("sys.argv", ["fasthydromap", command, str(PDB),
        str(tmp_path / "nonexistent.dcd"), "--model", model, option, "-o", str(root)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert option in capsys.readouterr().err
    assert not root.parent.exists()


def test_direct_run_rejects_trajectory_preparation_before_loading(monkeypatch):
    from types import SimpleNamespace
    from FastHydroMap.v2 import command
    monkeypatch.setattr(command, "Predictor", lambda *a, **k: pytest.fail("Must reject first"))
    with pytest.raises(ValueError, match="--prepare.*single structure"):
        command.run(SimpleNamespace(dcd="trajectory.dcd", prepare=True))


def test_legacy_default_remains_fdewet(monkeypatch):
    class SelectedLegacy(Exception):
        pass
    def load():
        def construct(*, quantity):
            assert quantity == "fdewet"
            raise SelectedLegacy()
        return construct
    monkeypatch.setattr("FastHydroMap.cli._load_predictor_or_exit", load)
    monkeypatch.setattr("sys.argv", ["fasthydromap", "predict", str(PDB), "--model", "v0"])
    with pytest.raises(SelectedLegacy):
        main()


def test_original_display_limits():
    assert PC_DISPLAY_RANGES == {"PC1": (-8., 8.), "PC2": (-2., 8.), "PC3": (-2., 2.)}


def test_selected_weights_are_full_refits_not_provisional():
    import hashlib
    import torch
    root = Path(__file__).resolve().parents[1]
    weights = root / "src/FastHydroMap/weights/v2"
    manifest = json.loads((weights / "manifest.json").read_text())
    protocol = json.loads((root / "training/shared_protocol.json").read_text())
    import tomllib
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert manifest["status"] == "released"
    assert manifest["release_version"] == project["project"]["version"]
    assert manifest["training_stage"] == "full800"
    assert manifest["fit_proteins"] == 800
    assert manifest["feature_version"] == "terminal_carboxylate_v2"
    assert set(manifest["feature_normalization"]["fit_ids"]) == set(protocol["fit_ids"])
    assert len(set(protocol["fit_ids"])) == 800
    assert [m["seed"] for m in manifest["members"]] == [17, 29, 43]
    for member in manifest["members"]:
        path = weights / member["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == member["sha256"]
        meta = torch.load(path, map_location="cpu", weights_only=True)["metadata"]
        assert meta["fit_proteins"] == 800 and meta["arm"] == "control"
        assert meta["feature_version"] == "terminal_carboxylate_v2"
        assert meta["status"] == "released"
        assert meta["release_version"] == manifest["release_version"]
        assert meta["best_step"] == member["steps"]
        assert meta["seed"] == member["seed"]
        assert meta["feature_normalization"] == manifest["feature_normalization"]


def test_latest_does_not_load_superseded_architectures(tmp_path):
    (tmp_path/"manifest.json").write_text('{"architecture":"probe_local_ca24_r12_v1"}')
    with pytest.raises(ValueError, match="shared-model weights"):
        Predictor(weights=tmp_path)


def test_installed_model_selftest(capsys):
    from FastHydroMap import selftest
    selftest(verbose=False)
    assert capsys.readouterr().out == ""
    selftest()
    assert "3 ensemble members" in capsys.readouterr().out


def test_package_contains_only_selected_and_legacy_weights():
    weights = Path(__file__).resolve().parents[1] / "src/FastHydroMap/weights"
    assert {p.name for p in (weights / "v2").glob("*.pt")} == {f"shared16_seed{s}.pt" for s in (17, 29, 43)}
    assert len(list(weights.glob("*.pt"))) == 4
