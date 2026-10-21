"""Distribution metadata retains upstream notices and usable PyPI links."""

from pathlib import Path
import csv
from collections import Counter
import json
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_parameter_data_license_is_declared_and_available():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "MIT AND LicenseRef-VIPARR-FF"
    assert set(project["license-files"]) == {
        "LICENSE", "licenses/VIPARR-FF-LICENSE.txt",
    }
    for name in project["license-files"]:
        assert (ROOT / name).is_file()
    notice = (ROOT / "licenses/VIPARR-FF-LICENSE.txt").read_text(encoding="utf-8")
    assert "Copyright 2018, D. E. Shaw Research. All rights reserved." in notice
    assert "Redistributions in binary form must reproduce" in notice


def test_pypi_readme_links_do_not_depend_on_a_checkout():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python -m pip install fasthydromap" in readme
    assert "python -m pip install -e ." not in readme
    assert "not yet available on PyPI" not in readme
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", readme)
    assert links
    assert all(link.startswith("https://") for link in links)


def test_public_protein_list_matches_final_fitting_protocol():
    with (ROOT / "training/proteins.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({row["pdb_id"] for row in rows}) == 1000
    assert Counter(row["split"] for row in rows) == {"train": 800, "validation": 100, "test": 100}
    protocol = json.loads((ROOT / "training/shared_protocol.json").read_text(encoding="utf-8"))
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
    receipt = json.loads((ROOT / "RELEASE_SELECTION.json").read_text(encoding="utf-8"))
    assert receipt["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert receipt["model_release_version"] == json.loads(manifest_path.read_text(encoding="utf-8"))["release_version"]


def test_published_dataset_links_and_archive_identity():
    access = json.loads((ROOT / "training/DATA_ACCESS.json").read_text(encoding="utf-8"))
    doi = "10.5281/zenodo.23543197"
    base = "https://zenodo.org/records/23543197/files/"
    assert access["status"] == "published"
    assert access["doi"] == access["labels_archive"]["doi"] == doi
    inputs, = access["archives"]
    assert access["download_url"] == inputs["download_url"]
    for archive in [inputs, access["labels_archive"]]:
        assert archive["download_url"] == base + archive["filename"] + "?download=1"
        assert re.fullmatch(r"[0-9a-f]{64}", archive["sha256"])
        assert archive["bytes"] > 0
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["urls"]["Dataset"] == "https://doi.org/" + doi
    training_readme = (ROOT / "training/README.md").read_text(encoding="utf-8")
    assert "deposition is pending" not in training_readme.lower()
    assert doi in training_readme


def test_preferred_citation_has_correct_author_names():
    import yaml
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    authors = citation["preferred-citation"]["authors"]
    assert [(a["family-names"], a["given-names"]) for a in authors] == [
        ("Lobo", "Samuel"), ("Najafi", "Saeed"),
        ("Shea", "Joan-Emma"), ("Shell", "M. Scott"),
    ]


def test_zenodo_release_metadata_matches_package_and_dataset():
    import yaml
    metadata = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert metadata["version"] == project["version"] == citation["version"]
    assert metadata["creators"] == [
        {"name": f'{a["family-names"]}, {a["given-names"]}'}
        for a in citation["authors"]
    ]
    assert metadata["upload_type"] == "software"
    assert metadata["license"] == citation["license"].lower()
    assert {"identifier": "10.5281/zenodo.23543197", "relation": "isSupplementTo",
            "resource_type": "dataset"} in metadata["related_identifiers"]
