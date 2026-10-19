"""Small tensor container and structural features; no PyTorch Geometric required."""

from types import SimpleNamespace
import numpy as np
import torch
from Bio.PDB import PDBParser
from . import geometry
from .layers import radial_basis
from .sasa import compute_sasa
from ..featurize.sasa import compute_relative_sasa_features


class Graph(SimpleNamespace):
    def clone(self):
        return Graph(
            **{
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in vars(self).items()
            }
        )


def core(path, rows):
    sasa = compute_sasa(path, len(rows))
    codes = np.array(["ACDEFGHIKLMNPQRSTVWY".index(r["aa"]) for r in rows])
    x = np.column_stack(
        (
            np.eye(20)[codes],
            sasa,
            np.zeros((len(rows), 3)),
            [r["is_n_term"] for r in rows],
            [r["is_c_term"] for r in rows],
            [r["histidine_charged"] for r in rows],
        )
    ).astype(np.float32)
    model = PDBParser(QUIET=True).get_structure("", str(path))[0]
    lookup = {}
    for residue in model.get_residues():
        if residue.id[0] == " " and "CA" in residue:
            geometry.select_altloc(residue)
            lookup[geometry.residue_uid_from_biopdb(residue)] = residue["CA"].coord
    uids = [r["res_uid"] for r in rows]
    xyz = np.asarray([lookup[u] for u in uids], np.float32)
    fn, fv, fc = geometry.node_features(
        uids, geometry.parse_geometry(path), "functional"
    )
    src, dst, rank = geometry.inbound_neighbors(xyz, 24)
    # Compute the cutoff in float64, matching the frozen training builder.
    capped = np.linalg.norm(xyz[src].astype(float) - xyz[dst], axis=1) <= 12.0
    functional = geometry.edge_features(src, dst, xyz, fn, fv, fc)
    distance = torch.tensor(np.linalg.norm(xyz[src] - xyz[dst], axis=1))
    return Graph(
        x=torch.tensor(x),
        pos=torch.tensor(xyz),
        aa_code=torch.tensor(codes),
        raw_sasa=torch.tensor(
            np.column_stack((sasa.sum(1), sasa[:, 5], sasa[:, 6])).astype(np.float32)
        ),
        pf_edge_index=torch.tensor(np.stack((src, dst)), dtype=torch.long),
        pf_edge_distance=distance,
        pf_edge_rank=torch.tensor(rank),
        pf_functional_edge=functional,
        graph_edge_index=torch.tensor(np.stack((src[capped], dst[capped])), dtype=torch.long),
        graph_edge_attr=torch.cat((radial_basis(distance[capped], 8), functional[capped]), -1),
    )


def attach(g, raw, cloud, atoms, stats):
    g = g.clone()
    sasa = g.raw_sasa.numpy()
    aa = np.array(["ACDEFGHIKLMNPQRSTVWY"[int(i)] for i in g.aa_code])
    rel = compute_relative_sasa_features(
        aa, sasa[:, 0], sasa[:, 1], sasa[:, 2], stats["sasa"]
    )
    g.x[:, 27:30] = torch.tensor(
        np.column_stack([rel[k] for k in ("sasa_rel", "sc_C_rel", "sc_NOS_rel")])
    )
    for key, value in raw.items():
        s = stats["rich"][key]
        setattr(
            g,
            "rich_" + key,
            torch.tensor(
                np.clip((value - np.array(s["mean"])) / np.array(s["std"]), -8, 8),
                dtype=torch.float32,
            ),
        )
    s = stats["point"]
    g.point_mask = torch.tensor(cloud["point_mask"])
    g.point = (
        torch.tensor(
            np.clip((cloud["point"] - np.array(s["mean"])) / np.array(s["std"]), -8, 8),
            dtype=torch.float32,
        )
        * g.point_mask[:, :, None]
    )
    g.probe_atom = torch.tensor(atoms)
    return g
