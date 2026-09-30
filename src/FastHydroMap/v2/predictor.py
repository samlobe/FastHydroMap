"""Portable v2 CPU inference, sharing feature extraction across all four outputs."""

from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import tempfile
import warnings
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser, PDBIO, Select
from scipy.spatial import cKDTree
from . import chemistry, surface, graph
from .geometry import residue_uid_from_biopdb
from .model import SurfaceModel
from ..io.residue_qc import collect_residue_records
from ..utils.constants import AA3

WEIGHTS = Path(__file__).parents[1] / "weights" / "v2"
TARGETS = ("Fdewet", "PC1", "PC2", "PC3")


def components(path, rows):
    model = PDBParser(QUIET=True).get_structure("", str(path))[0]
    chains = list(dict.fromkeys(r["chain_id"] for r in rows))
    parents = list(range(len(chains)))
    xyz = {
        c: np.array(
            [
                a.coord
                for r in model[c]
                if r.id[0] == " "
                for a in r
                if a.element not in ("H", "D")
            ]
        )
        for c in chains
    }

    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i

    for i, chain in enumerate(chains):
        tree = cKDTree(xyz[chain])
        for j in range(i):
            if np.any(tree.query(xyz[chains[j]], distance_upper_bound=18.0)[0] < 18.0):
                parents[root(i)] = root(j)
    return [
        [c for j, c in enumerate(chains) if root(j) == k]
        for k in sorted({root(j) for j in range(len(chains))})
    ]


class Predictor:
    """Load once and reuse. Input must contain complete standard heavy atoms.

    Histidine states map residue IDs (e.g. A:42) to HID/HIE/HIP. Unspecified
    ordinary HIS defaults to HIE with a warning; this is not pKa estimation.
    """

    def __init__(self, quantities=TARGETS, threads=4):
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        if isinstance(quantities, str):
            quantities = [quantities]
        lookup = {t.lower(): t for t in TARGETS}
        try:
            self.quantities = tuple(
                dict.fromkeys(lookup[q.lower()] for q in quantities)
            )
        except KeyError as exc:
            raise ValueError("Choose Fdewet, PC1, PC2 or PC3") from exc
        if not self.quantities:
            raise ValueError("Select at least one quantity")
        self.threads = threads
        self.manifest = json.loads((WEIGHTS / "manifest.json").read_text())
        self.models = {}
        for target in self.quantities:
            spec = self.manifest["models"][target]
            models = []
            for member in spec["members"]:
                path = WEIGHTS / member["file"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != member["sha256"]:
                    raise ValueError(f"Checkpoint checksum mismatch: {path.name}")
                model = SurfaceModel()
                model.load_state_dict(
                    torch.load(path, map_location="cpu", weights_only=True)
                )
                models.append(model.eval())
            self.models[target] = (models, spec["normalization"])

    @torch.inference_mode()
    def _component(self, path, rows, parts):
        uids = [r["res_uid"] for r in rows]
        states = [r["histidine_state"] for r in rows]
        try:
            parsed = chemistry.parse(path, uids, states)
        except (AssertionError, KeyError, ValueError) as exc:
            raise ValueError(
                f"Incomplete or unsupported heavy-atom chemistry: {exc}. Prepare the structure before v2 inference (see docs/V2.md)."
            ) from exc
        if len(parsed[0]) < 8:
            raise ValueError(
                "v2 atom features require at least eight protein heavy atoms"
            )
        core = graph.core(path, rows)
        raw = cloud = atoms = None
        for chain in dict.fromkeys(r["chain_id"] for r in rows):
            ix = [i for i, r in enumerate(rows) if r["chain_id"] == chain]
            rr, cc, aa, _ = surface.build(
                path,
                uids,
                states,
                parsed=parsed,
                targets=ix,
                anchor_index=ix[0],
                optimized=True,
                workers=self.threads,
                residue_workers=self.threads,
            )
            if raw is None:
                raw = {k: np.zeros_like(v) for k, v in rr.items()}
                cloud = {k: np.zeros_like(v) for k, v in cc.items()}
                atoms = np.zeros_like(aa)
            for k in raw:
                raw[k][ix] = rr[k][ix]
            for k in cloud:
                cloud[k][ix] = cc[k][ix]
            atoms[ix] = aa[ix]
        output = {}
        for target, (models, stats) in self.models.items():
            g = graph.attach(core, raw, cloud, atoms, stats)
            values = [m(g, return_parts=True) for m in models]
            means = {
                k: torch.stack([v[k] for v in values]).mean(0).numpy()
                for k in values[0]
            }
            if "pc_target" in stats:
                s = stats["pc_target"]
                for k in means:
                    means[k] = means[k] * s["std"] + (
                        s["mean"] if k != "context" else 0.0
                    )
            output[target] = means["total"]
            if parts:
                output[target + "_intrinsic"] = means["intrinsic"]
                output[target + "_context"] = means["context"]
        if not all(np.isfinite(v).all() for v in output.values()):
            raise ValueError("Nonfinite model output")
        return output

    def predict(self, pdb, *, histidine_states=None, parts=False):
        """Return a DataFrame with stable residue IDs and unrounded predictions."""
        path = Path(pdb)
        structure = PDBParser(QUIET=True).get_structure("", str(path))
        if len(structure) != 1:
            raise ValueError("Provide one PDB model at a time")
        if not all(np.isfinite(a.coord).all() for a in structure.get_atoms()):
            raise ValueError("PDB coordinates must be finite")
        if any(a.is_disordered() for a in structure.get_atoms()):
            raise ValueError(
                "Resolve alternate atom locations before v2 prediction (or use --prepare)"
            )
        if any(
            r.id[0] == " " and r.resname in AA3 and "CA" not in r
            for r in structure.get_residues()
        ):
            raise ValueError(
                "A protein residue is missing its CA atom; prepare the structure first"
            )
        named_states = {}
        for residue in structure.get_residues():
            if residue.resname in ("HID", "HIE", "HIP"):
                named_states[residue_uid_from_biopdb(residue)] = residue.resname
                residue.resname = "HIS"
        supplied = dict(histidine_states or {})
        if any(v not in ("HID", "HIE", "HIP") for v in supplied.values()):
            raise ValueError("Histidine states must be HID, HIE or HIP")
        with tempfile.TemporaryDirectory(prefix="fasthydromap_v2_") as tmp:
            if named_states:
                path = Path(tmp) / "histidine_names.pdb"
                writer = PDBIO()
                writer.set_structure(structure)
                writer.save(str(path))
            records, ignored, unsupported = collect_residue_records(path)
            if not records or unsupported:
                raise ValueError(
                    f"Standard protein residues required; unsupported: {unsupported}"
                )
            if len({r.res_uid for r in records}) != len(records):
                raise ValueError(
                    "Residue identifiers must be unique, including chain and insertion code"
                )
            if ignored:
                warnings.warn(
                    "Nonprotein atoms are ignored; ligand, water and membrane context is not modeled",
                    stacklevel=2,
                )
            histidines = {r.res_uid for r in records if r.aa == "H"}
            if set(supplied) - histidines:
                raise ValueError(
                    "Histidine assignments contain unknown or non-histidine residue IDs"
                )
            assigned = {**named_states, **supplied}
            if histidines - assigned.keys():
                warnings.warn(
                    "Unassigned HIS residues use HIE; supply histidine states to specify protonation",
                    stacklevel=2,
                )
            rows = []
            for rec in records:
                state = assigned.get(rec.res_uid, "HIE") if rec.aa == "H" else ""
                rows.append(
                    dict(
                        asdict(rec),
                        histidine_state=state,
                        histidine_charged=int(state == "HIP"),
                    )
                )
            groups = components(path, rows)
            if len(groups) > 1:
                warnings.warn(
                    "Disconnected proteins are predicted independently", stacklevel=2
                )
            previous_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(self.threads)
                outputs = {}
                for j, chains in enumerate(groups):
                    indices = [i for i, r in enumerate(rows) if r["chain_id"] in chains]
                    current = path
                    if len(groups) > 1:

                        class Keep(Select):
                            def accept_chain(self, chain):
                                return chain.id in chains

                        current = Path(tmp) / f"component{j}.pdb"
                        writer = PDBIO()
                        writer.set_structure(structure[0])
                        writer.save(str(current), Keep())
                    out = self._component(current, [rows[i] for i in indices], parts)
                    for key, values in out.items():
                        outputs.setdefault(key, np.zeros(len(rows), np.float32))[
                            indices
                        ] = values
            finally:
                torch.set_num_threads(previous_threads)
        frame = pd.DataFrame(rows)[
            [
                "res_uid",
                "chain_id",
                "resid",
                "insertion_code",
                "aa",
                "histidine_state",
                "is_n_term",
                "is_c_term",
            ]
        ]
        for key, values in outputs.items():
            frame[key] = values
        frame.attrs["model"] = "v2"
        frame.attrs["input_sha256"] = hashlib.sha256(Path(pdb).read_bytes()).hexdigest()
        return frame
