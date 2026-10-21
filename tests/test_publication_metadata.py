"""Distribution metadata retains upstream notices and usable PyPI links."""

from pathlib import Path
import csv
from collections import Counter
import json
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_parameter_data_license_is_declared_and_available():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license"] == "MIT AND LicenseRef-VIPARR-FF"
    assert set(project["license-files"]) == {
        "LICENSE", "licenses/VIPARR-FF-LICENSE.txt",
    }
    for name in project["license-files"]:
        assert (ROOT / name).is_file()
    notice = (ROOT / "licenses/VIPARR-FF-LICENSE.txt").read_text()
    assert "Copyright 2018, D. E. Shaw Research. All rights reserved." in notice
    assert "Redistributions in binary form must reproduce" in notice


def test_pypi_readme_links_do_not_depend_on_a_checkout():
    readme = (ROOT / "README.md").read_text()
    assert "python -m pip install fasthydromap" in readme
    assert "python -m pip install -e ." not in readme
    assert "not yet available on PyPI" not in readme
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", readme)
    assert links
    assert all(link.startswith("https://") for link in links)


def test_public_protein_list_matches_final_fitting_protocol():
    with (ROOT / "training/proteins.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({row["pdb_id"] for row in rows}) == 1000
    assert Counter(row["split"] for row in rows) == {"train": 800, "validation": 100, "test": 100}
    protocol = json.loads((ROOT / "training/shared_protocol.json").read_text())
    assert {row["pdb_id"] for row in rows if row["split"] == "train"} == set(protocol["fit_ids"])
    for row in rows:
        assert len(row["sequence"]) == int(row["sequence_length"])
        assert row["replicate_count"] == "3"
        assert row["source_author_chain_id"] and row["source_label_asym_id"]
        assert row["pdb_url"] == "https://www.rcsb.org/structure/" + row["pdb_id"]
        assert re.fullmatch(r"[0-9a-f]{64}", row["structure_sha256"])
        assert re.fullmatch(r"[0-9a-f]{64}", row["simulation_input_sha256"])
        assert row["structure_file"] == "structures/" + row["pdb_id"] + ".pdb"


def test_model_selection_receipt_matches_release_manifest():
    import hashlib
    manifest_path = ROOT / "src/FastHydroMap/weights/v2/manifest.json"
    receipt = json.loads((ROOT / "RELEASE_SELECTION.json").read_text())
    assert receipt["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert receipt["model_release_version"] == json.loads(manifest_path.read_text())["release_version"]


def test_preferred_citation_has_correct_author_names():
    import yaml
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    authors = citation["preferred-citation"]["authors"]
    assert [(a["family-names"], a["given-names"]) for a in authors] == [
        ("Lobo", "Samuel"), ("Najafi", "Saeed"),
        ("Shea", "Joan-Emma"), ("Shell", "M. Scott"),
    ]
