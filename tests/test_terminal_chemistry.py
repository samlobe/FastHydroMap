"""Terminal atom naming, template selection, and input validation."""

import json
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import Atom, Chain, Model, PDBIO, PDBParser, Residue, Structure

from FastHydroMap.v2 import chemistry
from FastHydroMap.v2.geometry import canonical_atoms


@pytest.mark.parametrize("version", [None, "legacy", "terminal_carboxylate_v1"])
def test_incompatible_manifest_feature_contract_rejected_before_loading(tmp_path, version):
    from FastHydroMap.v2.predictor import Predictor

    manifest = {"architecture": "shared_water_embedding_v1"}
    if version is not None:
        manifest["feature_version"] = version
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Model feature version must be"):
        Predictor(weights=tmp_path)


@pytest.mark.parametrize("version", [None, "legacy"])
def test_incompatible_checkpoint_feature_contract_rejected(tmp_path, version):
    import hashlib
    import torch
    from FastHydroMap.v2.predictor import FEATURE_VERSION, Predictor, TARGETS

    checkpoint = tmp_path / "member.pt"
    metadata = {} if version is None else {"feature_version": version}
    torch.save({"metadata": metadata}, checkpoint)
    manifest = {
        "architecture": "shared_water_embedding_v1", "feature_version": FEATURE_VERSION,
        "embedding_seed": 17, "width": 16, "targets": list(TARGETS),
        "members": [{"seed": 17, "file": checkpoint.name,
                     "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Checkpoint feature version mismatch"):
        Predictor(weights=tmp_path)


def test_regression_fixture_preserves_original_atom_records():
    import hashlib

    directory = Path(__file__).parent / "data"
    provenance = json.loads((directory / "v2_chain_preparation.json").read_text())
    content = (directory / "v2_chain.pdb").read_bytes()
    assert hashlib.sha256(content).hexdigest() == provenance["completed_sha256"]
    lines = content.decode().splitlines(keepends=True)
    oxygen = [line for line in lines if line.startswith("ATOM") and line[12:16].strip() == "OXT"]
    assert len(oxygen) == 1
    original = "".join(line.replace("TER     559", "TER     558") for line in lines if line not in oxygen)
    assert hashlib.sha256(original.encode()).hexdigest() == provenance["original_sha256"]


def make_peptide(tmp_path, template_names=("NALA", "ALA", "CALA"), pdb_names=None):
    templates, _ = chemistry.templates()
    structure = Structure.Structure("peptide")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    structure.add(model)
    model.add(chain)
    states = []
    for i, template in enumerate(template_names):
        base = template[1:] if template[0] in "NC" and template[1:] in templates else template
        name = "HIS" if base in ("HID", "HIE", "HIP") else "CYS" if base == "CYX" else base
        name = pdb_names[i] if pdb_names else name
        residue = Residue.Residue((" ", i + 1, " "), name, " ")
        chain.add(residue)
        shift = np.array([i * 3.5, 0., 0.])
        backbone = {"N": [0., 0., 0.], "CA": [1.2, 1., 0.],
                    "C": [2.4, 0., 0.], "O": [2.5, -1.2, 0.], "OXT": [3.2, -.4, 0.]}
        for j, (atom_name, properties) in enumerate(templates[template].items()):
            xyz = np.asarray(backbone.get(atom_name, [1.2 + .2 * j, 2. + .3 * j, .2 * j])) + shift
            residue.add(Atom.Atom(atom_name, xyz.astype(np.float32), 0., 1., " ",
                                 atom_name.rjust(4), j + 1, element=properties["element"]))
        states.append(base if base in ("HID", "HIE", "HIP") else "")
    path = tmp_path / "peptide.pdb"
    save(structure, path)
    return structure, path, [f"A:{i + 1}" for i in range(len(template_names))], states


def save(structure, path):
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(path))


def renamed(atom, name):
    result = atom.copy()
    result.id = result.name = name
    result.fullname = name.rjust(4)
    return result


def test_preserves_two_terminal_oxygens_and_net_charge(tmp_path):
    _, path, uids, states = make_peptide(tmp_path)
    xyz, props, owners, _, _, checks, _ = chemistry.parse(path, uids, states)
    assert len(xyz) == 16
    assert [item["template"] for item in checks] == ["NALA", "ALA", "CALA"]
    np.testing.assert_allclose([item["charge"] for item in checks], [1., 0., -1.], atol=1e-12)
    np.testing.assert_allclose(props[:, 1].sum(), 0., atol=1e-12)
    assert (owners == 2).sum() == 6
    last = list(PDBParser(QUIET=True).get_structure("", path).get_residues())[-1]
    atoms, sidechain = canonical_atoms(last)
    assert "O" in atoms and "OXT" in atoms and len(sidechain) == 1
    np.testing.assert_array_equal(atoms["O"], last["O"].coord)
    np.testing.assert_array_equal(atoms["OXT"], last["OXT"].coord)


@pytest.mark.parametrize("pair", [("OT1", "OT2"), ("O1", "O2"), ("OC1", "OC2")])
def test_terminal_aliases_preserve_coordinates_and_features(tmp_path, pair):
    structure, path, uids, states = make_peptide(tmp_path)
    baseline = chemistry.parse(path, uids, states)
    last = list(structure.get_residues())[-1]
    # Preserve file order while changing the names.
    copied = [renamed(a, dict(zip(("O", "OXT"), pair)).get(a.name, a.name)) for a in last]
    for name in list(last.child_dict):
        last.detach_child(name)
    for atom in copied:
        last.add(atom)
    save(structure, path)
    aliased = chemistry.parse(path, uids, states)
    for index in range(5):
        np.testing.assert_array_equal(baseline[index], aliased[index])
    assert baseline[5] == aliased[5]


@pytest.mark.parametrize("state", ["HID", "HIE", "HIP"])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_histidine_states_at_each_chain_position(tmp_path, state, position):
    names = ["NALA", "ALA", "CALA"]
    names[position] = ("N" if position == 0 else "C" if position == 2 else "") + state
    _, path, uids, states = make_peptide(tmp_path, names)
    result = chemistry.parse(path, uids, states)
    expected = (1 if position == 0 else -1 if position == 2 else 0) + (state == "HIP")
    assert result[5][position]["template"] == names[position]
    assert result[5][position]["charge"] == pytest.approx(expected, abs=1e-12)


def test_internal_atom_parameters_unchanged_and_terminal_templates_normalized():
    raw = json.loads(Path(chemistry.__file__).with_name("chemistry.json").read_text())["templates"]
    normalized, _ = chemistry.templates()
    for name, atoms in raw.items():
        if "OC1" not in atoms:
            assert normalized[name] == atoms
        else:
            assert normalized[name]["O"] == atoms["OC1"]
            assert normalized[name]["OXT"] == atoms["OC2"]
            assert len(normalized[name]) == len(atoms)
            assert sum(v["charge"] for v in normalized[name].values()) == sum(v["charge"] for v in atoms.values())


@pytest.mark.parametrize("name", ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
                                  "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR",
                                  "TRP", "TYR", "VAL"])
def test_standard_residue_template_selection_at_all_positions(tmp_path, name):
    _, path, uids, states = make_peptide(tmp_path, ("N" + name, name, "C" + name))
    result = chemistry.parse(path, uids, states)
    assert [r["template"] for r in result[5]] == ["N" + name, name, "C" + name]
    base_charge = 1 if name in ("ARG", "LYS") else -1 if name in ("ASP", "GLU") else 0
    # The source XML's rounded internal LYS charges sum to 0.999999.
    np.testing.assert_allclose([r["charge"] for r in result[5]],
                               [base_charge + 1, base_charge, base_charge - 1], atol=2e-6)


def test_internal_coordinates_and_chemical_channels_remain_unchanged(tmp_path):
    structure, path, uids, states = make_peptide(tmp_path, ("NALA", "HIE", "CALA"))
    internal = list(structure.get_residues())[1]
    atoms, _ = canonical_atoms(internal)
    assert list(atoms) == list(internal.child_dict)
    for name, position in atoms.items():
        np.testing.assert_array_equal(position, internal[name].coord)
    result = chemistry.parse(path, uids, states)
    internal_properties = result[1][result[2] == 1]
    raw = json.loads(Path(chemistry.__file__).with_name("chemistry.json").read_text())["templates"]["HIE"]
    for index, name in enumerate(atoms):
        expected = raw[name]
        assert internal_properties[index, 1] == expected["charge"]
        assert internal_properties[index, 2] == expected["sigma"]
        assert internal_properties[index, 3] == expected["epsilon"]


def test_disulfide_templates_are_retained(tmp_path):
    structure, path, uids, states = make_peptide(tmp_path, ("NALA", "CYX", "CYX", "CALA"))
    residues = list(structure.get_residues())
    residues[1]["SG"].coord = np.array([5., 4., 0.])
    residues[2]["SG"].coord = np.array([7.05, 4., 0.])
    save(structure, path)
    result = chemistry.parse(path, uids, states)
    assert [r["template"] for r in result[5]] == ["NALA", "CYX", "CYX", "CALA"]
    assert result[1][:, 1].sum() == pytest.approx(0., abs=1e-12)


def test_missing_oxt_is_not_invented(tmp_path):
    structure, path, uids, states = make_peptide(tmp_path)
    list(structure.get_residues())[-1].detach_child("OXT")
    save(structure, path)
    with pytest.raises(ValueError, match="C-terminal OXT is missing"):
        chemistry.parse(path, uids, states)


def test_internal_oxt_is_rejected(tmp_path):
    structure, path, uids, states = make_peptide(tmp_path)
    middle = list(structure.get_residues())[1]
    middle.add(renamed(middle["O"], "OXT"))
    save(structure, path)
    with pytest.raises(ValueError, match="following peptide bond"):
        chemistry.parse(path, uids, states)


@pytest.mark.parametrize("distance", [1.8, 1.85, 2.0])
def test_stretched_peptide_bond_matches_terminal_flag_policy(tmp_path, distance):
    from FastHydroMap.io.residue_qc import collect_residue_records

    structure, path, uids, states = make_peptide(tmp_path)
    first, middle, _ = list(structure.get_residues())
    middle["N"].coord = first["C"].coord + np.array([distance, 0., 0.])
    save(structure, path)
    result = chemistry.parse(path, uids, states)
    records, _, _ = collect_residue_records(path)
    assert not records[0].is_c_term and not records[1].is_n_term
    assert [r["template"] for r in result[5]] == ["NALA", "ALA", "CALA"]


def test_peptide_gap_above_two_angstrom_is_not_joined(tmp_path):
    from FastHydroMap.io.residue_qc import collect_residue_records

    structure, path, uids, states = make_peptide(tmp_path)
    first, middle, _ = list(structure.get_residues())
    middle["N"].coord = first["C"].coord + np.array([2.01, 0., 0.])
    save(structure, path)
    records, _, _ = collect_residue_records(path)
    assert records[0].is_c_term and records[1].is_n_term
    with pytest.raises(ValueError, match="singleton/disconnected"):
        chemistry.parse(path, uids, states)


def test_singleton_requires_an_unavailable_combined_template(tmp_path):
    _, path, uids, states = make_peptide(tmp_path, ("CALA",))
    with pytest.raises(ValueError, match="combined N/C-terminal"):
        chemistry.parse(path, uids, states)


def test_duplicate_alias_is_rejected_instead_of_discarded(tmp_path):
    structure, path, uids, states = make_peptide(tmp_path)
    last = list(structure.get_residues())[-1]
    last.add(renamed(last["O"], "OT1"))
    save(structure, path)
    with pytest.raises(ValueError, match="ambiguous duplicate atom aliases"):
        chemistry.parse(path, uids, states)


@pytest.mark.parametrize("name", ["ACE", "NME", "NHE", "NH2"])
def test_caps_are_not_silently_ignored(tmp_path, name):
    structure, path, uids, states = make_peptide(tmp_path)
    cap = Residue.Residue(("H_" + name, 4, " "), name, " ")
    cap.add(renamed(list(structure.get_residues())[0]["C"], "CH3"))
    structure[0]["A"].add(cap)
    save(structure, path)
    with pytest.raises(ValueError, match="Capped peptides are unsupported"):
        chemistry.parse(path, uids, states)
