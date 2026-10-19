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
from .model import SharedWaterModel
from ..io.residue_qc import collect_residue_records
from ..utils.constants import AA3

WEIGHTS = Path(__file__).parents[1] / "weights" / "v2"
TARGETS = ("Fdewet", "PC1", "PC2", "PC3")
FEATURE_VERSION = "terminal_carboxylate_v2"
TERMINAL_CAUTION_NOTE = (
    "Terminal residues were excluded from fitting targets; their predictions are extrapolations."
)


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

    def __init__(self, quantities=TARGETS, threads=4, weights=None):
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
        self.weights = Path(weights) if weights is not None else WEIGHTS
        self.manifest = json.loads((self.weights / "manifest.json").read_text())
        self.shared = self.manifest.get("architecture") == "shared_water_embedding_v1"
        self.compact = True
        if not self.shared:
            raise ValueError("This predictor requires shared-model weights; use --model v0 for the previous published model")
        if self.manifest.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"Model feature version must be {FEATURE_VERSION!r}; these weights are "
                "incompatible with the current terminal-chemistry features."
            )
        if self.shared:
            self.members = []
            seeds = [m['seed'] for m in self.manifest['members']]
            if len(seeds) != len(set(seeds)) or self.manifest['embedding_seed'] not in seeds:
                raise ValueError('Unique model seeds and an official embedding member are required')
            if self.manifest['targets'] != list(TARGETS) or self.manifest['width'] != 16:
                raise ValueError('Unsupported shared output order or embedding width')
            for member in self.manifest['members']:
                path = self.weights / member['file']
                if hashlib.sha256(path.read_bytes()).hexdigest() != member['sha256']:
                    raise ValueError(f'Checkpoint checksum mismatch: {path.name}')
                payload = torch.load(path, map_location='cpu', weights_only=True)
                meta = payload['metadata']
                if meta.get("feature_version") != FEATURE_VERSION:
                    raise ValueError(f"Checkpoint feature version mismatch: {path.name}")
                for key in ('feature_normalization', 'target_normalization'):
                    if meta[key] != self.manifest[key]:
                        raise ValueError(f'Checkpoint/manifest mismatch: {key}')
                if meta['seed'] != member['seed'] or meta['architecture'] != 'shared_water_embedding_v1':
                    raise ValueError('Checkpoint identity mismatch')
                if meta['fit_proteins'] != self.manifest['fit_proteins']:
                    raise ValueError('Checkpoint training population mismatch')
                network = SharedWaterModel()
                network.load_state_dict(payload['state_dict'])
                self.members.append((member, network.eval()))
            if self.manifest.get('status') != 'released':
                warnings.warn('LOCAL PUBLICATION CANDIDATE: these weights are not a published release. '
                    'Check model_manifest_sha256 when recording predictions.', stacklevel=2)
            return

    @torch.inference_mode()
    def _component(self, path, rows, parts, embeddings=False):
        uids = [r["res_uid"] for r in rows]
        states = [r["histidine_state"] for r in rows]
        try:
            parsed = chemistry.parse(path, uids, states)
        except (AssertionError, KeyError, ValueError) as exc:
            raise ValueError(
                f"Incomplete or unsupported heavy-atom chemistry: {exc}. "
                "For missing atoms, use --prepare on a single structure and inspect the "
                "prepared PDB; trajectory topology must be complete (see docs/MODEL.md)."
            ) from exc
        if len(parsed[0]) < 8:
            raise ValueError(
                "Atom features require at least eight protein heavy atoms"
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
                compact=self.compact,
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
        if getattr(self, 'shared', False):
            g = SharedWaterModel.inputs(graph.attach(core, raw, cloud, atoms, self.manifest['feature_normalization']))
            norm = self.manifest['target_normalization']
            mean, std = torch.tensor(norm['mean']), torch.tensor(norm['std'])
            predictions = []
            for member, network in self.members:
                value, h = network(g, return_embeddings=True)
                predictions.append(value * std + mean)
                if embeddings and member['seed'] == self.manifest['embedding_seed']:
                    output.update({f'embedding_{j}': h[:, j].numpy() for j in range(16)})
            averaged = torch.stack(predictions).mean(0).numpy()
            output.update({target: averaged[:, TARGETS.index(target)] for target in self.quantities})
        if not all(np.isfinite(v).all() for v in output.values()):
            raise ValueError("Nonfinite model output")
        # MDTraj returns nm²; only this user-facing display column is in Å².
        output["sasa_A2"] = core.raw_sasa[:, 0].numpy() * 100.0
        output["probe_count"] = cloud["point_mask"].sum(1).astype(np.float32)
        return output

    def predict(self, pdb, *, histidine_states=None, parts=False, embeddings=False):
        """Return a DataFrame with stable residue IDs and unrounded predictions."""
        if parts and self.compact:
            raise ValueError("The selected v2 model has no intrinsic/context decomposition. Omit --parts.")
        if embeddings and not getattr(self, 'shared', False):
            raise ValueError('Embedding export requires the shared water model')
        path = Path(pdb)
        structure = PDBParser(QUIET=True).get_structure("", str(path))
        if len(structure) != 1:
            raise ValueError("Provide one PDB model at a time")
        if not all(np.isfinite(a.coord).all() for a in structure.get_atoms()):
            raise ValueError("PDB coordinates must be finite")
        if any(a.is_disordered() for a in structure.get_atoms()):
            raise ValueError(
                "Resolve alternate atom locations before prediction (or use --prepare)"
            )
        standard_names = set(AA3) | {"HID", "HIE", "HIP"}
        unsupported = [
            f"{residue_uid_from_biopdb(r)} {r.resname}"
            for r in structure.get_residues()
            if (r.id[0] == " " and r.resname not in standard_names)
            or r.resname in {"ACE", "NME", "NHE", "NH2"}
        ]
        if unsupported:
            raise ValueError(
                "Unsupported polymer residues or caps: " + ", ".join(unsupported)
                + ". Supply standard, uncapped protein residues."
            )
        if any(
            r.id[0] == " " and r.resname in standard_names and "CA" not in r
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
                    if embeddings:
                        out = self._component(current, [rows[i] for i in indices], parts, embeddings=True)
                    else:
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
        frame["probe_count"] = frame.probe_count.astype(int)
        frame["terminal_caution"] = frame.is_n_term | frame.is_c_term
        frame["zero_sasa_caution"] = frame.sasa_A2 <= 0
        frame["no_probe_caution"] = frame.probe_count == 0
        frame["assembly_caution"] = frame.chain_id.nunique() > 1
        frame["hid_unvalidated_caution"] = frame.histidine_state.eq("HID")
        frame["applicability_note"] = "Structural flags only; MD hydration support, replica repeatability and prediction confidence are unknown."
        frame.attrs["model"] = "latest"
        frame.attrs["terminal_caution_note"] = TERMINAL_CAUTION_NOTE
        frame.attrs["feature_version"] = FEATURE_VERSION
        frame.attrs["input_sha256"] = hashlib.sha256(Path(pdb).read_bytes()).hexdigest()
        if getattr(self, 'shared', False):
            official = next(m for m in self.manifest['members'] if m['seed'] == self.manifest['embedding_seed'])
            frame.attrs.update(architecture=self.manifest['architecture'],
                model_status=self.manifest['status'],
                model_manifest_sha256=hashlib.sha256((self.weights/'manifest.json').read_bytes()).hexdigest(),
                embedding_seed=official['seed'] if embeddings else None,
                embedding_checkpoint_sha256=official['sha256'] if embeddings else None,
                embedding_width=16 if embeddings else None,
                embedding_representation='post_message_passing_pre_prediction_heads' if embeddings else None,
                prediction_ensemble_seeds=[m['seed'] for m in self.manifest['members']])
        return frame
