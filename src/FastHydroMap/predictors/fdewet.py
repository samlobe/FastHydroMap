# FastHydroMap/predictors/fdewet.py
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..featurize.graph import build_graph
from ..featurize.sasa import (
    SASA_COMPONENT_NAMES,
    compute_relative_sasa_features,
    load_sasa_feature_stats,
    residue_sasa_components,
)
from ..io.pdb import load_traj
from ..io.residue_qc import (
    collect_backbone_atom_issues,
    collect_residue_records,
    residue_display_label,
)
from ..models.mpnn import FdewetMPNN
from ..utils.atom_names import backbone_alias_priority, canonical_backbone_atom_name

WEIGHT_DIR = Path(__file__).parents[1] / "weights"

def _display_residue_label(
    chain_id: str,
    resid: int,
    insertion_code: str,
    *,
    include_chain: bool,
) -> str:
    base = f"{resid}{insertion_code.strip()}"
    if include_chain:
        chain = chain_id.strip() or "_"
        return f"{chain}:{base}"
    return base


class FdewetPredictor:
    def __init__(
        self,
        k_nn: int = 12,
        n_rbf: int = 3,
        rbf_min: float = 2.0,
        rbf_max: float = 14.0,
        rbf_sigma: float = 4.0,
        mpnn_pt: Path = WEIGHT_DIR / "mpnn_latest.pt",
        sasa_stats_npz: Path = WEIGHT_DIR / "sasa_feature_stats.npz",
        device: str | torch.device | None = None,
    ):
        self.k = k_nn
        self.n_rbf = n_rbf
        self.rbf_min = rbf_min
        self.rbf_max = rbf_max
        self.rbf_sigma = rbf_sigma
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.mpnn = FdewetMPNN().to(self.device)
        self.mpnn.load_state_dict(torch.load(mpnn_pt, map_location=self.device))
        self.mpnn.eval()
        self.sasa_stats = load_sasa_feature_stats(sasa_stats_npz)

    @staticmethod
    def _preview(items: list[str], max_items: int = 20) -> str:
        if len(items) <= max_items:
            return ", ".join(items)
        return ", ".join(items[:max_items]) + f", ... (+{len(items) - max_items} more)"

    def _warn_input_qc(
        self,
        ignored_nonprotein: list[str],
        nonstandard_polymer: list[str],
        backbone_issues: list[str],
    ) -> None:
        if ignored_nonprotein:
            warnings.warn(
                "ignoring non-protein residues: " + self._preview(ignored_nonprotein),
                stacklevel=2,
            )
        if nonstandard_polymer:
            warnings.warn(
                "nonstandard polymer residues will be treated as unknown residue identity; "
                "predictions may be unreliable for: "
                + self._preview(nonstandard_polymer),
                stacklevel=2,
            )
        if backbone_issues:
            warnings.warn(
                "polymer residues with incomplete backbone atoms were detected; "
                "affected residues may be skipped: " + self._preview(backbone_issues),
                stacklevel=2,
            )

    @staticmethod
    def _trajectory_residue_indices(topology) -> list[int]:
        keep: list[int] = []
        for idx, residue in enumerate(topology.residues):
            has_n = False
            has_ca = False
            has_c = False
            for atom in residue.atoms:
                elem = atom.element.symbol if atom.element is not None else ""
                canonical = canonical_backbone_atom_name(atom.name, elem)
                if canonical == "N":
                    has_n = True
                elif canonical == "CA":
                    has_ca = True
                elif canonical == "C":
                    has_c = True
            if has_n and has_ca and has_c:
                keep.append(idx)
        return keep

    @staticmethod
    def _populate_frame_atom_coords(
        frame,
        topology,
        traj_residue_indices: list[int],
        res2atom: dict[str, np.ndarray],
    ) -> None:
        residues = list(topology.residues)
        for out_idx, top_idx in enumerate(traj_residue_indices):
            residue = residues[top_idx]
            best_priority = {name: 9999 for name in res2atom}
            for atom in residue.atoms:
                elem = atom.element.symbol if atom.element is not None else ""
                canonical = canonical_backbone_atom_name(atom.name, elem)
                if canonical not in res2atom:
                    continue
                priority = backbone_alias_priority(atom.name)
                if priority < best_priority[canonical]:
                    best_priority[canonical] = priority
                    res2atom[canonical][out_idx] = frame.xyz[0, atom.index] * 10.0

    def _feature_frame(self, traj_frame, residue_records):
        components = residue_sasa_components(traj_frame, strict=True)
        total_sasa = components.sum(axis=1)
        aa_codes = [record.aa for record in residue_records]
        rel = compute_relative_sasa_features(
            aa_codes,
            total_sasa,
            components[:, SASA_COMPONENT_NAMES.index("sc_C")],
            components[:, SASA_COMPONENT_NAMES.index("sc_NOS")],
            self.sasa_stats,
        )

        absurd = np.where(
            (np.abs(rel["sasa_rel"]) > 6.0)
            | (np.abs(rel["sc_C_rel"]) > 6.0)
            | (np.abs(rel["sc_NOS_rel"]) > 6.0)
        )[0]
        if absurd.size:
            labels = [
                residue_display_label(
                    residue_records[i].chain_id,
                    residue_records[i].resid,
                    residue_records[i].insertion_code,
                )
                for i in absurd[:10]
            ]
            warnings.warn(
                "unusually large heavy-atom SASA features for residues: " + ", ".join(labels),
                stacklevel=2,
            )

        data = {
            "res_uid": [record.res_uid for record in residue_records],
            "Fdewet_pred": np.zeros(len(residue_records), np.float32),
            "is_n_term": np.array([record.is_n_term for record in residue_records], dtype=np.bool_),
            "is_c_term": np.array([record.is_c_term for record in residue_records], dtype=np.bool_),
            "sasa_rel": rel["sasa_rel"],
            "sc_C_rel": rel["sc_C_rel"],
            "sc_NOS_rel": rel["sc_NOS_rel"],
        }
        for comp_idx, comp_name in enumerate(SASA_COMPONENT_NAMES):
            data[f"{comp_name}_sasa"] = components[:, comp_idx]
        data["sasa"] = total_sasa
        return pd.DataFrame(data).set_index("res_uid")

    def __call__(
        self,
        pdb_path: Path,
        dcd_path: Path | None = None,
        chain_id: str | None = None,
        return_parts: bool = False,
    ):
        traj = load_traj(pdb_path, dcd_path)
        top = traj.topology

        residue_records, ignored_nonprotein, nonstandard_polymer = collect_residue_records(pdb_path)
        if not residue_records:
            raise ValueError(
                "no protein-like residues detected (requires polymer residues with backbone atoms)"
            )
        backbone_issues = collect_backbone_atom_issues(pdb_path)
        self._warn_input_qc(ignored_nonprotein, nonstandard_polymer, backbone_issues)
        traj_residue_indices = self._trajectory_residue_indices(top)
        if len(residue_records) != len(traj_residue_indices):
            raise ValueError(
                f"residue count mismatch between PDB parsing ({len(residue_records)}) "
                f"and trajectory protein-like residues ({len(traj_residue_indices)})"
            )

        include_chain = len({r.chain_id for r in residue_records}) > 1
        res_ids = [
            _display_residue_label(
                r.chain_id, r.resid, r.insertion_code, include_chain=include_chain
            )
            for r in residue_records
        ]

        if traj.n_frames == 1:
            df_features = self._feature_frame(traj, residue_records)
            graph = build_graph(
                pdb_path,
                df_features,
                atom_coords=None,
                k_nn=self.k,
                n_rbf=self.n_rbf,
                rbf_min=self.rbf_min,
                rbf_max=self.rbf_max,
                rbf_sigma=self.rbf_sigma,
                pid_ix=0,
            ).to(self.device)
            with torch.inference_mode():
                out = self.mpnn(graph, return_parts=return_parts)
            if return_parts:
                parts = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in out.items()}
                return parts, res_ids
            return out.detach().cpu().numpy().astype(np.float32), res_ids

        total_rows: list[np.ndarray] = []
        intrinsic_rows: list[np.ndarray] = []
        context_rows: list[np.ndarray] = []

        res2atom = {
            name: np.full((len(res_ids), 3), np.nan, np.float32)
            for name in ("N", "CA", "C", "O", "CB")
        }

        with torch.inference_mode():
            for frame_idx in tqdm(range(traj.n_frames)):
                fr = traj.slice(frame_idx)
                df_features = self._feature_frame(fr, residue_records)

                for arr in res2atom.values():
                    arr.fill(np.nan)

                self._populate_frame_atom_coords(fr, top, traj_residue_indices, res2atom)

                graph = build_graph(
                    pdb_path,
                    df_features,
                    atom_coords=res2atom,
                    k_nn=self.k,
                    n_rbf=self.n_rbf,
                    rbf_min=self.rbf_min,
                    rbf_max=self.rbf_max,
                    rbf_sigma=self.rbf_sigma,
                    pid_ix=0,
                ).to(self.device)

                out = self.mpnn(graph, return_parts=return_parts)
                if return_parts:
                    total_rows.append(out["total"].detach().cpu().numpy().astype(np.float32))
                    intrinsic_rows.append(out["intrinsic"].detach().cpu().numpy().astype(np.float32))
                    context_rows.append(out["context"].detach().cpu().numpy().astype(np.float32))
                else:
                    total_rows.append(out.detach().cpu().numpy().astype(np.float32))

        if return_parts:
            return {
                "total": np.stack(total_rows, axis=0),
                "intrinsic": np.stack(intrinsic_rows, axis=0),
                "context": np.stack(context_rows, axis=0),
            }, res_ids

        return np.stack(total_rows, axis=0), res_ids
