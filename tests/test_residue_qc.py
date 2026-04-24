from pathlib import Path

from FastHydroMap.io.residue_qc import collect_residue_records


DATA_DIR = Path(__file__).parent / "data"
PDB_PATH = DATA_DIR / "1A1U.pdb"


def _shift_chain_a_atom(src: Path, dst: Path, resid: int, atom_name: str, dx: float) -> None:
    lines = []
    for line in src.read_text().splitlines():
        if (
            line.startswith(("ATOM", "HETATM"))
            and line[21] == "A"
            and int(line[22:26]) == resid
            and line[12:16].strip() == atom_name
        ):
            x = float(line[30:38]) + dx
            line = f"{line[:30]}{x:8.3f}{line[38:]}"
        lines.append(line)
    dst.write_text("\n".join(lines) + "\n")


def test_chain_breaks_are_marked_as_termini(tmp_path):
    broken = tmp_path / "1A1U_broken.pdb"
    _shift_chain_a_atom(PDB_PATH, broken, resid=40, atom_name="N", dx=100.0)

    records, ignored, nonstandard = collect_residue_records(broken)
    assert not ignored
    assert not nonstandard

    by_resid = {record.resid: record for record in records}
    assert by_resid[39].is_c_term
    assert by_resid[40].is_n_term
    assert not by_resid[40].is_c_term
    assert not by_resid[39].is_n_term
