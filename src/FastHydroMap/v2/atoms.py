import numpy as np
from scipy.spatial import cKDTree

NAMES = [
    "radius",
    "charge",
    "sigma",
    "sqrt_epsilon",
    "donor",
    "acceptor",
    "C",
    "N",
    "O",
    "S",
    "aliphatic",
    "aromatic",
    "backbone",
    *[f"distance_rbf_{r}" for r in (2, 3, 4, 5, 6, 8)],
    "normal_alignment",
    "same_residue",
]


def arrays(xyz, props, owner, frames, centers, points, normals, mask):
    f = np.asarray(frames[0], float)
    c = (np.asarray(centers, float) - centers[0]) @ f.T
    x = (np.asarray(xyz, float) - centers[0]) @ f.T
    probes = points + c[:, None, :]
    d, idx = cKDTree(x).query(probes.reshape(-1, 3), k=8)
    p = props[idx].copy()
    p[:, :, 0] /= 2
    p[:, :, 2] /= 4
    p[:, :, 3] = np.sqrt(p[:, :, 3])
    disp = probes.reshape(-1, 3)[:, None, :] - x[idx]
    unit = disp / np.maximum(d[:, :, None], 1e-8)
    align = (unit * normals.reshape(-1, 3)[:, None, :]).sum(2)
    same = owner[idx] == np.repeat(np.arange(len(centers)), 24)[:, None]
    rbf = np.exp(-0.5 * ((d[:, :, None] - np.array([2, 3, 4, 5, 6, 8])) / 1.0) ** 2)
    out = np.concatenate((p, rbf, align[:, :, None], same[:, :, None]), 2).reshape(
        len(centers), 24, 8, len(NAMES)
    )
    out *= mask[:, :, None, None]
    assert np.isfinite(out).all()
    return out.astype(np.float32)
