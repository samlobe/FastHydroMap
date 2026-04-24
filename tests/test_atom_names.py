from FastHydroMap.utils.atom_names import (
    backbone_alias_priority,
    canonical_backbone_atom_name,
)


def test_canonical_backbone_atom_aliases():
    assert canonical_backbone_atom_name("O", "O") == "O"
    assert canonical_backbone_atom_name("OXT", "O") == "O"
    assert canonical_backbone_atom_name("OT1", "O") == "O"
    assert canonical_backbone_atom_name("OT2", "O") == "O"
    assert canonical_backbone_atom_name("O1", "O") == "O"
    assert canonical_backbone_atom_name("OC2", "O") == "O"


def test_canonical_backbone_atom_rejects_mismatched_elements():
    assert canonical_backbone_atom_name("CA", "CA") is None
    assert canonical_backbone_atom_name("OT1", "N") is None


def test_backbone_alias_priority_prefers_canonical_names():
    assert backbone_alias_priority("O") < backbone_alias_priority("OXT")
    assert backbone_alias_priority("OXT") < backbone_alias_priority("OT2")
