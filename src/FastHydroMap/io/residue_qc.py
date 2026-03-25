from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

from ..utils.atom_names import backbone_alias_priority, canonical_backbone_atom_name
from ..utils.constants import AA3_TO_1


@dataclass(frozen=True)
class ResidueRecord:
    chain_id: str
    resid: int
    insertion_code: str
    res_uid: str
    aa: str
    resname: str
    is_standard: bool
    is_n_term: bool
    is_c_term: bool


def residue_uid(chain_id: str, resid: int, insertion_code: str) -> str:
    chain = chain_id.strip() or "_"
    return f"{chain}:{int(resid)}{insertion_code.strip()}"


def residue_display_label(chain_id: str, resid: int, insertion_code: str) -> str:
    chain = chain_id.strip() or "_"
    return f"{chain}:{int(resid)}{insertion_code.strip()}"


def _icode_rank(icode: str) -> int:
    stripped = icode.strip()
    if not stripped:
        return 0
    return ord(stripped[0].upper()) - ord("A") + 1


def _is_sequential_residue(prev_res, next_res) -> bool:
    prev_id = prev_res.id[1]
    next_id = next_res.id[1]
    prev_icode = prev_res.id[2].strip()
    next_icode = next_res.id[2].strip()

    if next_id == prev_id:
        return _icode_rank(next_icode) == _icode_rank(prev_icode) + 1
    if next_id == prev_id + 1:
        return not next_icode
    return False


def _peptide_bond_present(prev_res, next_res, max_cn_distance: float = 2.0) -> bool:
    c_xyz = _canonical_atom_coord(prev_res, "C")
    n_xyz = _canonical_atom_coord(next_res, "N")
    if c_xyz is not None and n_xyz is not None:
        return float(np.linalg.norm(c_xyz - n_xyz)) <= max_cn_distance
    return _is_sequential_residue(prev_res, next_res)


def _canonical_atom_coord(res, canonical_name: str) -> np.ndarray | None:
    best = None
    best_priority = 9999
    for atom in res:
        elem = atom.element.strip().upper() if atom.element else ""
        if elem == "H":
            continue
        canonical = canonical_backbone_atom_name(atom.name, elem)
        if canonical != canonical_name:
            continue
        priority = backbone_alias_priority(atom.name)
        if priority < best_priority:
            best_priority = priority
            best = atom.coord.astype(np.float32)
    return best


def collect_residue_records(
    pdb_path: Path | str,
) -> tuple[list[ResidueRecord], list[str], list[str]]:
    """
    Return protein-like residue records plus user-facing warning lists.

    * Standard and nonstandard polymer residues with a CA atom are kept.
    * Heterogens, solvent, and small molecules are ignored.
    """

    struct = PDBParser(QUIET=True).get_structure("", str(pdb_path))[0]
    records: list[ResidueRecord] = []
    ignored_nonprotein: list[str] = []
    nonstandard_polymer: list[str] = []

    for chain in struct:
        chain_pairs: list[tuple[ResidueRecord, object]] = []
        for res in chain:
            hetflag, resid, insertion_code = res.id
            resname = res.resname.strip().upper()
            label = f"{residue_display_label(chain.id, resid, insertion_code)} {resname}"

            if hetflag.strip():
                if any(atom.element != "H" for atom in res):
                    ignored_nonprotein.append(label)
                continue
            if "CA" not in res:
                continue
            if not any(atom.element != "H" for atom in res):
                continue

            aa = AA3_TO_1.get(resname, "X")
            is_standard = aa != "X"
            if not is_standard:
                nonstandard_polymer.append(label)

            chain_pairs.append(
                (
                ResidueRecord(
                    chain_id=chain.id,
                    resid=resid,
                    insertion_code=insertion_code.strip(),
                    res_uid=residue_uid(chain.id, resid, insertion_code),
                    aa=aa,
                    resname=resname,
                    is_standard=is_standard,
                    is_n_term=False,
                    is_c_term=False,
                )
                ,
                res,
                )
            )

        if chain_pairs:
            chain_records = [record for record, _ in chain_pairs]
            chain_residues = [res for _, res in chain_pairs]
            chain_out: list[ResidueRecord] = []

            for i, record in enumerate(chain_records):
                has_prev = i > 0 and _peptide_bond_present(chain_residues[i - 1], chain_residues[i])
                has_next = i < len(chain_records) - 1 and _peptide_bond_present(chain_residues[i], chain_residues[i + 1])
                chain_out.append(
                    ResidueRecord(
                        **{
                            **record.__dict__,
                            "is_n_term": not has_prev,
                            "is_c_term": not has_next,
                        }
                    )
                )

            records.extend(chain_out)

    return records, sorted(set(ignored_nonprotein)), sorted(set(nonstandard_polymer))


def collect_backbone_atom_issues(pdb_path: Path | str) -> list[str]:
    """
    Report polymer residues with missing backbone atoms after alias mapping.

    Backbone aliases such as OXT/OT1/OT2 are treated as canonical O.
    """

    struct = PDBParser(QUIET=True).get_structure("", str(pdb_path))[0]
    issues: list[str] = []
    required = {"N", "CA", "C", "O"}

    for chain in struct:
        for res in chain:
            hetflag, resid, insertion_code = res.id
            if hetflag.strip():
                continue
            if not any(atom.element != "H" for atom in res):
                continue

            found: set[str] = set()
            for atom in res:
                elem = atom.element.strip().upper() if atom.element else ""
                if elem == "H":
                    continue
                canonical = canonical_backbone_atom_name(atom.name, elem)
                if canonical in required:
                    found.add(canonical)

            missing = sorted(required - found)
            if missing:
                label = residue_display_label(chain.id, resid, insertion_code)
                resname = res.resname.strip().upper()
                issues.append(f"{label} {resname}: missing {', '.join(missing)}")

    return issues
