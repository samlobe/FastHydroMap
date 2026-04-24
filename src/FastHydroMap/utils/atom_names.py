from __future__ import annotations


CANONICAL_BACKBONE_ATOMS = ("N", "CA", "C", "O", "CB")

_ALIAS_TO_CANONICAL = {
    "N": "N",
    "CA": "CA",
    "C": "C",
    "O": "O",
    "OXT": "O",
    "OT1": "O",
    "OT2": "O",
    "O1": "O",
    "O2": "O",
    "OC1": "O",
    "OC2": "O",
    "CB": "CB",
}

_ALIAS_PRIORITY = {
    # Prefer canonical protein names when several aliases are present.
    "N": 0,
    "CA": 0,
    "C": 0,
    "O": 0,
    "CB": 0,
    "OXT": 1,
    "OT1": 2,
    "OT2": 3,
    "O1": 4,
    "O2": 5,
    "OC1": 6,
    "OC2": 7,
}

_EXPECTED_ELEMENT = {
    "N": "N",
    "CA": "C",
    "C": "C",
    "O": "O",
    "CB": "C",
}


def canonical_backbone_atom_name(name: str, element: str | None = None) -> str | None:
    atom_name = name.strip().upper()
    canonical = _ALIAS_TO_CANONICAL.get(atom_name)
    if canonical is None:
        return None

    if element is not None and element.strip():
        elem = element.strip().upper()
        if elem != _EXPECTED_ELEMENT[canonical]:
            return None

    return canonical


def backbone_alias_priority(name: str) -> int:
    return _ALIAS_PRIORITY.get(name.strip().upper(), 9999)

