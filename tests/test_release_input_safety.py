"""Reject ambiguous inputs before producing apparently complete predictions."""

from pathlib import Path

import pytest

from FastHydroMap import Predictor
from FastHydroMap.cli import main
from FastHydroMap.install_torch import torch_install_command
from FastHydroMap.v2.command import prepare
from FastHydroMap.v2.predictor import FEATURE_VERSION, TERMINAL_CAUTION_NOTE


PDB = Path(__file__).parent / "data/v2_chain.pdb"


def test_terminal_extrapolation_is_recorded_without_chemistry_warning():
    import warnings

    model = Predictor()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.predict(PDB)
    assert not any("terminal" in str(w.message).lower() for w in caught)
    assert "terminal_chemistry_note" not in result.attrs
    assert result.attrs["terminal_caution_note"] == TERMINAL_CAUTION_NOTE
    assert result.attrs["feature_version"] == FEATURE_VERSION
    assert result.loc[result.is_c_term, "terminal_caution"].all()


@pytest.mark.parametrize("state", ["HIS", "HID", "HIE", "HIP"])
def test_histidine_missing_ca_is_not_silently_omitted(tmp_path, state):
    lines = PDB.read_text().splitlines(keepends=True)
    residue = next(line[21:27] for line in lines
                   if line.startswith("ATOM") and line[17:20] == "HIS")
    modified = []
    for line in lines:
        if line.startswith("ATOM") and line[21:27] == residue:
            if line[12:16].strip() == "CA":
                continue
            line = line[:17] + state + line[20:]
        modified.append(line)
    path = tmp_path / "missing_ca.pdb"
    path.write_text("".join(modified))
    with pytest.raises(ValueError, match="missing its CA"):
        Predictor().predict(path)


@pytest.mark.parametrize("cap", ["ACE", "NME"])
def test_atom_record_caps_are_not_silently_omitted(tmp_path, cap):
    lines = PDB.read_text().splitlines(keepends=True)
    residue = next(line[21:27] for line in lines if line.startswith("ATOM"))
    modified = []
    for line in lines:
        if line.startswith("ATOM") and line[21:27] == residue:
            if line[12:16].strip() == "CA":
                continue
            line = line[:17] + cap + line[20:]
        modified.append(line)
    path = tmp_path / "capped.pdb"
    path.write_text("".join(modified))
    with pytest.raises(ValueError, match="Unsupported polymer residues or caps"):
        Predictor().predict(path)


def test_prepare_does_not_silently_select_first_model(tmp_path):
    body = "".join(line for line in PDB.read_text().splitlines(keepends=True)
                   if line.startswith(("ATOM", "HETATM")))
    path = tmp_path / "ensemble.pdb"
    path.write_text(f"MODEL        1\n{body}ENDMDL\nMODEL        2\n{body}ENDMDL\nEND\n")
    output = tmp_path / "prepared.pdb"
    with pytest.raises(ValueError, match="one PDB model"):
        prepare(path, output)
    assert not output.exists()


def test_missing_terminal_oxygen_requires_preparation(tmp_path):
    lines = PDB.read_text().splitlines(keepends=True)
    path = tmp_path / "missing_oxt.pdb"
    path.write_text("".join(line for line in lines
                            if not (line.startswith("ATOM") and line[12:16].strip() == "OXT")))
    with pytest.raises(ValueError, match="C-terminal OXT is missing; use --prepare"):
        Predictor().predict(path)


@pytest.mark.parametrize("record", ["ATOM  ", "HETATM"])
def test_prepare_rejects_caps_instead_of_removing_them(tmp_path, record):
    source = PDB.read_text()
    atom = next(line for line in source.splitlines(keepends=True) if line.startswith("ATOM"))
    cap = record + atom[6:17] + "ACE" + atom[20:22] + "9999" + atom[26:]
    path = tmp_path / "capped.pdb"
    path.write_text("".join(line for line in source.splitlines(keepends=True)
                            if line[:6].strip() not in {"END", "TER"}) + cap + "END\n")
    with pytest.raises(ValueError, match="does not remove peptide caps"):
        prepare(path, tmp_path / "prepared.pdb")


@pytest.mark.parametrize("option", [["--threads", "0"], ["--threads", "-2"], ["--parts"]])
def test_invalid_options_fail_before_loading(monkeypatch, option):
    monkeypatch.setattr("FastHydroMap.cli._load_predictor_or_exit",
                        lambda: pytest.fail("Invalid options must fail before model loading"))
    monkeypatch.setattr("sys.argv", ["fasthydromap", "predict", str(PDB), *option])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_macos_cpu_installer_uses_pypi(monkeypatch):
    monkeypatch.setattr("FastHydroMap.install_torch.sys.platform", "darwin")
    command = torch_install_command("cpu")
    assert command[-2:] == ["--index-url", "https://pypi.org/simple"]
    with pytest.raises(ValueError, match="CUDA wheels are not available on macOS"):
        torch_install_command("cu121")


def test_cli_reports_io_failure_without_traceback(monkeypatch, capsys):
    def fail(args):
        raise OSError("No space left on device")

    monkeypatch.setattr("FastHydroMap.v2.command.run", fail)
    monkeypatch.setattr("sys.argv", ["fasthydromap", "predict", str(PDB)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "error: No space left on device" in capsys.readouterr().err


def test_nonfinite_trajectory_frame_is_rejected_without_outputs(monkeypatch, tmp_path, capsys):
    import mdtraj as md
    import numpy as np

    block = md.load_pdb(str(PDB))
    block.xyz[0, 0, 0] = np.nan
    monkeypatch.setattr(md, "iterload", lambda *a, **k: iter([block]))
    monkeypatch.setattr("sys.argv", ["fasthydromap", "predict-trajectory", str(PDB),
                                    "unused.dcd", "-o", str(tmp_path / "maps")])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "frame 0 contains nonfinite coordinates" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
