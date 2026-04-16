from pathlib import Path
import sys
import warnings

import mdtraj as md
import numpy as np
import pandas as pd
import pytest

from FastHydroMap.cli import main
from FastHydroMap.predictors.fdewet import FdewetPredictor


DATA_DIR = Path(__file__).parent / "data"
PDB_PATH = DATA_DIR / "1A1U.pdb"
REF_CSV = DATA_DIR / "1A1U_fdewet.csv"
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
TRAJ_PDB_PATH = EXAMPLES_DIR / "proteinG.pdb"
TRAJ_DCD_PATH = EXAMPLES_DIR / "proteinG_short.dcd"


def _write_1a1u_with_53a(src: Path, dst: Path) -> None:
    lines = []
    for line in src.read_text().splitlines():
        if (
            line.startswith(("ATOM", "HETATM"))
            and line[21] == "A"
            and line[22:26] == "  54"
        ):
            line = f"{line[:22]}  53A{line[27:]}"
        lines.append(line)
    dst.write_text("\n".join(lines) + "\n")


def _append_water_heterogen(src: Path, dst: Path) -> None:
    lines = src.read_text().splitlines()
    water = [
        f"HETATM{9991:5d} {'O':>4} {'HOH':>3} {'Z':1}{1:4d}    {20.000:8.3f}{20.000:8.3f}{20.000:8.3f}{1.00:6.2f}{20.00:6.2f}          {'O':>2}",
        f"HETATM{9992:5d} {'H1':>4} {'HOH':>3} {'Z':1}{1:4d}    {20.300:8.3f}{20.500:8.3f}{20.000:8.3f}{1.00:6.2f}{20.00:6.2f}          {'H':>2}",
        f"HETATM{9993:5d} {'H2':>4} {'HOH':>3} {'Z':1}{1:4d}    {19.700:8.3f}{20.500:8.3f}{20.000:8.3f}{1.00:6.2f}{20.00:6.2f}          {'H':>2}",
    ]
    try:
        end_idx = next(i for i, line in enumerate(lines) if line.startswith("END"))
    except StopIteration:
        end_idx = len(lines)
    new_lines = lines[:end_idx] + water + lines[end_idx:]
    dst.write_text("\n".join(new_lines) + "\n")


def _rewrite_backbone_oxygen_aliases(src: Path, dst: Path) -> None:
    lines = []
    for line in src.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16] == " O  ":
            serial = int(line[6:11])
            alias = f"{'OT1' if serial % 2 else 'OT2':>4}"
            line = f"{line[:12]}{alias}{line[16:]}"
        lines.append(line)
    dst.write_text("\n".join(lines) + "\n")


def _rewrite_selected_residues_to_unk(src: Path, dst: Path, residue_numbers: set[int]) -> None:
    lines = []
    for line in src.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            resid = int(line[22:26])
            if resid in residue_numbers:
                line = f"{line[:17]}UNK{line[20:]}"
        lines.append(line)
    dst.write_text("\n".join(lines) + "\n")


def _append_unknown_polymer_chain(src: Path, dst: Path) -> None:
    lines = src.read_text().splitlines()
    extra = []
    serial = 9001
    coords = [
        ("N", 30.0, 10.0, 10.0, "N"),
        ("CA", 31.4, 10.1, 10.2, "C"),
        ("C", 32.1, 11.4, 10.0, "C"),
        ("O", 33.2, 11.5, 10.2, "O"),
        ("CB", 31.8, 9.0, 11.2, "C"),
    ]
    for resid in (1, 2):
        x_shift = 3.8 * (resid - 1)
        for atom_name, x, y, z, elem in coords:
            extra.append(
                f"ATOM  {serial:5d} {atom_name:>4} {'UNK':>3} {'B'}{resid:4d}    "
                f"{x + x_shift:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{20.00:6.2f}          {elem:>2}"
            )
            serial += 1
    try:
        end_idx = next(i for i, line in enumerate(lines) if line.startswith("END"))
    except StopIteration:
        end_idx = len(lines)
    new_lines = lines[:end_idx] + extra + lines[end_idx:]
    dst.write_text("\n".join(new_lines) + "\n")


def test_predictor_matches_1a1u_reference():
    predictor = FdewetPredictor()
    scores, res_ids = predictor(PDB_PATH)

    ref = pd.read_csv(REF_CSV)
    ref_scores = ref["fdewet (kJ/mol/water)"].to_numpy(np.float32)
    np.testing.assert_array_equal(np.asarray(res_ids), ref["resid"].astype(str).to_numpy())
    assert np.mean(np.abs(scores - ref_scores)) < 0.12
    assert np.max(np.abs(scores - ref_scores)) < 0.7


def test_predictor_return_parts_matches_total():
    predictor = FdewetPredictor()
    parts, res_ids = predictor(PDB_PATH, return_parts=True)

    assert res_ids[0] == "26"
    assert set(parts) == {"total", "intrinsic", "context"}
    np.testing.assert_allclose(parts["total"], parts["intrinsic"] + parts["context"], atol=1e-6)


def test_cli_predict_writes_outputs_for_path_outroot(monkeypatch, tmp_path):
    outroot = tmp_path / "1A1U_fdewet"
    monkeypatch.setattr(
        sys,
        "argv",
        ["fasthydromap", "predict", str(PDB_PATH), "-o", str(outroot)],
    )

    main()

    csv_path = Path(f"{outroot}.csv")
    pdb_path = Path(f"{outroot}.pdb")
    assert csv_path.exists()
    assert pdb_path.exists()

    df = pd.read_csv(csv_path)
    ref = pd.read_csv(REF_CSV)
    ref_scores = ref["fdewet (kJ/mol/water)"].to_numpy(np.float32)
    assert df.shape == (1, len(ref) + 1)
    out_scores = df.iloc[0, 1:].to_numpy(np.float32)
    assert np.mean(np.abs(out_scores - ref_scores)) < 0.15
    assert np.max(np.abs(out_scores - ref_scores)) < 0.5


def test_cli_predict_trajectory_writes_decomposition_outputs(monkeypatch, tmp_path):
    outroot = tmp_path / "proteinG_traj"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fasthydromap",
            "predict-trajectory",
            str(TRAJ_PDB_PATH),
            str(TRAJ_DCD_PATH),
            "-o",
            str(outroot),
        ],
    )

    main()

    total_csv = Path(f"{outroot}_total.csv")
    intrinsic_csv = Path(f"{outroot}_intrinsic.csv")
    context_csv = Path(f"{outroot}_context.csv")
    summary_csv = Path(f"{outroot}_parts_summary.csv")
    pdb_path = Path(f"{outroot}.pdb")

    for path in (total_csv, intrinsic_csv, context_csv, summary_csv, pdb_path):
        assert path.exists()

    total_df = pd.read_csv(total_csv)
    summary_df = pd.read_csv(summary_csv)
    assert total_df.shape[0] >= 1
    assert summary_df.shape[0] == total_df.shape[0]
    assert {"total_mean", "intrinsic_mean", "context_mean"}.issubset(summary_df.columns)


def test_predictor_distinguishes_insertion_codes(tmp_path):
    pdb_53a = tmp_path / "1A1U_53A.pdb"
    _write_1a1u_with_53a(PDB_PATH, pdb_53a)

    predictor = FdewetPredictor()
    scores, res_ids = predictor(pdb_53a)

    assert len(scores) == 29
    assert len(res_ids) == 29
    assert res_ids[-2:] == ["53", "53A"]
    assert len(set(res_ids)) == len(res_ids)


def test_predictor_ignores_nonprotein_residues_with_warning(tmp_path):
    pdb_with_water = tmp_path / "1A1U_plus_water.pdb"
    _append_water_heterogen(PDB_PATH, pdb_with_water)

    predictor = FdewetPredictor()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        scores, res_ids = predictor(pdb_with_water)

    assert any("ignoring non-protein residues" in str(w.message) for w in seen)
    assert len(scores) == 29
    assert len(res_ids) == 29


def test_predictor_handles_oxygen_alias_names(tmp_path):
    alias_pdb = tmp_path / "1A1U_ot_alias.pdb"
    _rewrite_backbone_oxygen_aliases(PDB_PATH, alias_pdb)

    predictor = FdewetPredictor()
    ref_scores, ref_ids = predictor(PDB_PATH)
    alias_scores, alias_ids = predictor(alias_pdb)

    np.testing.assert_array_equal(alias_ids, ref_ids)
    assert np.mean(np.abs(alias_scores - ref_scores)) < 0.05


def test_predictor_handles_unknown_residue_names_with_warning(tmp_path):
    unk_pdb = tmp_path / "1A1U_unk.pdb"
    _rewrite_selected_residues_to_unk(PDB_PATH, unk_pdb, {31, 32, 33})

    predictor = FdewetPredictor()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        scores, res_ids = predictor(unk_pdb)

    assert any("nonstandard polymer residues" in str(w.message) for w in seen)
    assert len(scores) == 29
    assert len(res_ids) == 29
    assert np.isfinite(scores).all()


def test_cli_predict_writes_bfactors_with_nonprotein_and_unknown_residues(monkeypatch, tmp_path):
    mixed_pdb = tmp_path / "1A1U_mixed.pdb"
    water_pdb = tmp_path / "1A1U_water.pdb"
    _append_water_heterogen(PDB_PATH, water_pdb)
    _append_unknown_polymer_chain(water_pdb, mixed_pdb)

    outroot = tmp_path / "mixed_fdewet"
    monkeypatch.setattr(
        sys,
        "argv",
        ["fasthydromap", "predict", str(mixed_pdb), "-o", str(outroot)],
    )

    main()

    assert Path(f"{outroot}.csv").exists()
    assert Path(f"{outroot}.pdb").exists()


def test_predictor_accepts_xtc_trajectory(tmp_path):
    xtc_path = tmp_path / "proteinG_short.xtc"
    traj = md.load(str(TRAJ_DCD_PATH), top=str(TRAJ_PDB_PATH))
    traj.save_xtc(str(xtc_path))

    predictor = FdewetPredictor()
    dcd_scores, dcd_ids = predictor(TRAJ_PDB_PATH, TRAJ_DCD_PATH)
    xtc_scores, xtc_ids = predictor(TRAJ_PDB_PATH, xtc_path)

    np.testing.assert_array_equal(xtc_ids, dcd_ids)
    assert xtc_scores.shape == dcd_scores.shape
    assert np.mean(np.abs(xtc_scores - dcd_scores)) < 0.02


def test_predictor_trajectory_atom_mismatch_raises_clear_error():
    predictor = FdewetPredictor()
    with pytest.raises(ValueError, match="atom count and atom order match exactly"):
        predictor(PDB_PATH, TRAJ_DCD_PATH)
