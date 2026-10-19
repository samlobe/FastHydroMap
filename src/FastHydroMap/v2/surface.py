"""Fused extraction for frozen full_surface models; no weights changed.

Reuse the common solvent grid, residue shells and angular probes. Only build
the 14 point channels actually consumed by full_surface; unused channels and
point-convolution edges are omitted. spacing/directions changes are OPTIONAL
approximations; default settings preserve the original feature definitions.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
from scipy.signal import fftconvolve
from . import chemistry as rich
from . import atoms as af


def lattice_ids(coords, lower, shape, radius=4.25):
    """Enumerate the same sorted lattice IDs as the former full-grid KD tree."""
    offset = np.stack(
        np.meshgrid(*([np.arange(-5, 6)] * 3), indexing="ij"), -1
    ).reshape(-1, 3)
    q = np.floor(coords - lower).astype(int)[:, None, :] + offset[None, :, :]
    valid = ((q >= 0) & (q < np.asarray(shape))).all(-1)
    valid &= np.square(q + lower - coords[:, None, :]).sum(-1) <= radius**2
    q = q[valid]
    return np.unique(np.ravel_multi_index(q.T, shape))


def build(
    path,
    uids,
    states,
    spacing=1.0,
    direction_count=32,
    workers=4,
    residue_workers=4,
    anchor_index=0,
    targets=None,
    parsed=None,
    optimized=False,
    compact=False,
):
    if spacing != 1.0:
        raise ValueError("The selected model requires 1-A grid spacing")
    xyz, props, owner, frames, centers, checks, water = (
        rich.parse(path, uids, states) if parsed is None else parsed
    )
    origin = np.asarray(centers[anchor_index], float)
    frame = np.asarray(frames[anchor_index], float)
    x = (np.asarray(xyz, float) - origin) @ frame.T
    c = (np.asarray(centers, float) - origin) @ frame.T
    tree = cKDTree(x)
    radii = props[:, 0]
    axes = [
        np.arange(lo, hi + spacing * 0.5, spacing)
        for lo, hi in zip(np.floor(x.min(0) - 6), np.ceil(x.max(0) + 6))
    ]
    shape = tuple(map(len, axes))
    if np.prod(shape) >= 2500000:
        raise ValueError(
            "Structure exceeds the v2 grid size limit; supply a smaller connected assembly"
        )
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    if optimized:
        # Nearest distance bounds clearance using the min/max atom radii.
        # Only the narrow ambiguous band needs the original eight-neighbor rule.
        d, _ = tree.query(
            points, workers=workers, distance_upper_bound=radii.max() + 1.4 + 1e-10
        )
        free = d > radii.max() + 1.4
        ambiguous = (d > radii.min() + 1.4) & (~free)
        free[ambiguous] = rich.clearance(points[ambiguous], tree, radii)[0] > 1.4
        clear = nearest = None
    else:
        clear = []
        nearest = []
        for begin in range(0, len(points), 50000):
            d, idx = tree.query(
                points[begin : begin + 50000], k=min(8, len(radii)), workers=workers
            )
            clear.append(np.min(d - radii[idx], axis=-1))
            nearest.append(idx[:, 0])
        clear = np.concatenate(clear)
        nearest = np.concatenate(nearest)
        free = clear > 1.4
    mask = free.reshape(shape)
    boundary = np.zeros(shape, bool)
    for dim in range(3):
        sl = [slice(None)] * 3
        sl[dim] = 0
        boundary[tuple(sl)] = True
        sl[dim] = -1
        boundary[tuple(sl)] = True
    connected = ndimage.binary_propagation(boundary & mask, mask=mask).ravel()
    lattice_counts = {}
    for radius in (2.0, 3.5):
        reach = int(np.ceil(radius))
        offset = np.arange(-reach, reach + 1)
        dx, dy, dz = np.meshgrid(offset, offset, offset, indexing="ij")
        kernel = (dx * dx + dy * dy + dz * dz <= radius * radius).astype(np.uint16)
        # Integer occupancy counts on a 1-A lattice. Rounding removes FFT
        # roundoff; the exact result is an integer bounded by kernel.sum().
        count = fftconvolve(mask, kernel, mode="same")
        assert np.max(np.abs(count - np.rint(count))) < 1e-5
        lattice_counts[radius] = np.rint(count).astype(np.uint16).ravel()
    grid = None if optimized else cKDTree(points)
    k = np.arange(direction_count)
    z = 1 - 2 * (k + 0.5) / direction_count
    theta = k * np.pi * (3 - np.sqrt(5))
    directions = np.column_stack(
        (np.sqrt(1 - z * z) * np.cos(theta), np.sqrt(1 - z * z) * np.sin(theta), z)
    )
    ii, jj = np.triu_indices(direction_count, 1)
    cosines = (directions @ directions.T)[ii, jj]
    angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
    basis = np.stack(
        (
            cosines,
            (3 * cosines**2 - 1) / 2,
            (5 * cosines**3 - 3 * cosines) / 2,
            (angles >= 40) & (angles < 50),
            (angles >= 100) & (angles < 120),
            (angles >= 140) & (angles < 150),
        ),
        1,
    )
    n = len(c)
    out = {
        name: np.zeros((n, len(names)), np.float32)
        for name, names in rich.NAMES.items()
    }
    cloud = dict(
        point=np.zeros((n, 24, 32), np.float32),
        point_mask=np.zeros((n, 24), bool),
        point_pos=np.zeros((n, 24, 3), np.float32),
        point_normal=np.zeros((n, 24, 3), np.float32),
    )

    def residue(i):
        own = np.where(owner == i)[0]
        own_tree = cKDTree(x[own])
        near = tree.query_ball_point(c[i], 18)
        env = x[near]
        pr = props[near]
        dist = own_tree.query(env)[0]
        categories = np.stack(
            (
                pr[:, 10],
                pr[:, 11],
                pr[:, 4],
                pr[:, 5],
                pr[:, 9],
                pr[:, 1],
                np.abs(pr[:, 1]),
                pr[:, 12],
                np.maximum(pr[:, 1], 0),
                np.maximum(-pr[:, 1], 0),
            ),
            1,
        )
        radial = np.exp(-0.5 * ((dist[:, None] - np.array([3, 5, 8, 12])) / 1.5) ** 2)
        out["atom_env"][i] = (radial.T @ categories).ravel()
        ids = (
            lattice_ids(x[own], points[0], shape)
            if optimized
            else np.unique(
                np.concatenate(
                    [np.asarray(v, int) for v in grid.query_ball_point(x[own], 4.25)]
                )
            )
        )
        ids = ids[free[ids]]
        if not len(ids):
            out["cavity"][i, -1] = 1
            out["angular"][i, -1] = 1
            out["field"][i, -4] = props[own, 1].sum()
            return
        probes = points[ids]
        own_dist = own_tree.query(probes)[0]
        shifted = probes - probes.mean(0)
        cov = shifted.T @ shifted / max(len(probes), 1)
        eig = np.maximum(np.linalg.eigvalsh(cov), 0)
        trace = max(eig.sum(), 1e-6)
        degree2 = lattice_counts[2.0][ids] * spacing**3 / (4 * np.pi * 2**3 / 3)
        degree35 = lattice_counts[3.5][ids] * spacing**3 / (4 * np.pi * 3.5**3 / 3)
        if optimized:
            clr, near_id = rich.clearance(probes, tree, radii)
        else:
            clr, near_id = clear[ids], nearest[ids]
        cats = props[near_id, 6:10].mean(0)
        out["cavity"][i] = np.r_[
            np.log1p(len(ids) * spacing**3),
            connected[ids].mean(),
            np.log1p((~connected[ids]).sum() * spacing**3),
            clr.mean(),
            clr.std(),
            np.quantile(clr, [0.1, 0.9]),
            own_dist.mean(),
            own_dist.std(),
            (own_dist < 3.5).mean(),
            eig / trace,
            np.sqrt(trace),
            np.linalg.norm(probes.mean(0) - c[i]),
            degree2.mean(),
            degree2.std(),
            degree35.mean(),
            degree35.std(),
            cats,
            0,
        ]
        chosen_index = np.linspace(0, len(probes) - 1, min(24, len(probes)), dtype=int)
        chosen = probes[chosen_index]
        m = len(chosen)
        allowed = (
            rich.clearance(
                (chosen[:, None, :] + 2.8 * directions[None, :, :]).reshape(-1, 3),
                tree,
                radii,
            )[0].reshape(-1, direction_count)
            > 1.4
        )
        pair = allowed[:, ii] & allowed[:, jj]
        counts = pair.sum(1)
        hist = (pair @ basis) / np.maximum(counts[:, None], 1)
        anisotropy = np.linalg.norm(allowed @ directions, axis=1) / np.maximum(
            allowed.sum(1), 1
        )
        out["angular"][i] = np.r_[
            allowed.mean(1).mean(),
            allowed.mean(1).std(),
            anisotropy.mean(),
            anisotropy.std(),
            hist.mean(0),
            np.std(hist[:, 3] + hist[:, 5]),
            (counts == 0).mean(),
        ]
        delta = chosen[:, None, :] - env[None, :, :]
        r = np.linalg.norm(delta, axis=-1).clip(0.5)
        cutoff = r < 12
        screen = np.exp(-r / 6) * cutoff
        q = pr[:, 1]
        pot = (screen * q / r).sum(1)
        ef = (delta * (screen * q / r**3)[:, :, None]).sum(1)
        efnorm = np.linalg.norm(ef, axis=1)
        sigma = (pr[:, 2] + water[0]) / 2
        eps = np.sqrt(pr[:, 3] * water[1])
        sr6 = (sigma[None, :] / r) ** 6
        lj = (4 * eps[None, :] * (sr6**2 - sr6) * cutoff).sum(1)
        local = np.exp(-0.5 * ((r - 3.0) / 0.8) ** 2)
        donor = local @ pr[:, 4]
        acceptor = local @ pr[:, 5]
        outward = chosen - c[i]
        outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-8)
        alignment = (ef * outward).sum(1)
        values = np.stack(
            (pot, efnorm, lj, donor, acceptor, donor - acceptor, alignment), 1
        )
        out["field"][i] = np.r_[
            rich.summarize(values),
            props[own, 1].sum(),
            q.sum(),
            np.abs(q).sum(),
            np.mean(donor * acceptor),
        ]
        d, ix = tree.query(chosen, k=8)
        nb = ix[np.arange(m), np.argmin(d - props[ix, 0], axis=1)]
        normal = chosen - x[nb]
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        cloud["point"][i, :m, :14] = np.column_stack(
            (
                clr[chosen_index],
                own_dist[chosen_index],
                connected[ids[chosen_index]],
                degree2[chosen_index],
                degree35[chosen_index],
                (normal * outward).sum(1),
                allowed.mean(1),
                anisotropy,
                hist,
            )
        )
        cloud["point_mask"][i, :m] = True
        cloud["point_pos"][i, :m] = chosen - c[i]
        cloud["point_normal"][i, :m] = normal

    def residue_compact(i):
        # Same consumed features and point ordering; omit unused residue banks.
        own = np.where(owner == i)[0]
        ids = lattice_ids(x[own], points[0], shape)
        ids = ids[free[ids]]
        out["cavity"][i, 0] = np.log1p(len(ids))
        if not len(ids):
            out["cavity"][i, -1] = 1
            return
        ids = ids[np.linspace(0, len(ids) - 1, min(24, len(ids)), dtype=int)]
        chosen = points[ids]
        m = len(chosen)
        clr, _ = rich.clearance(chosen, tree, radii)
        own_dist = cKDTree(x[own]).query(chosen)[0]
        d, ix = tree.query(chosen, k=8)
        nb = ix[np.arange(m), np.argmin(d - props[ix, 0], axis=1)]
        normal = chosen - x[nb]
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        outward = chosen - c[i]
        outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-8)
        allowed = rich.clearance(
            (chosen[:, None, :] + 2.8 * directions).reshape(-1, 3), tree, radii
        )[0].reshape(m, direction_count) > 1.4
        pair = allowed[:, ii] & allowed[:, jj]
        hist = (pair @ basis) / np.maximum(pair.sum(1)[:, None], 1)
        anis = np.linalg.norm(allowed @ directions, axis=1) / np.maximum(allowed.sum(1), 1)
        cloud["point"][i, :m, :14] = np.column_stack((
            clr, own_dist, connected[ids],
            lattice_counts[2.0][ids] / (4 * np.pi * 2**3 / 3),
            lattice_counts[3.5][ids] / (4 * np.pi * 3.5**3 / 3),
            (normal * outward).sum(1), allowed.mean(1), anis, hist))
        cloud["point_mask"][i, :m] = True
        cloud["point_pos"][i, :m] = chosen - c[i]
        cloud["point_normal"][i, :m] = normal

    targets = range(n) if targets is None else targets
    residue_builder = residue_compact if compact else residue
    if residue_workers == 1:
        for i in targets:
            residue_builder(i)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=residue_workers) as pool:
            list(pool.map(residue_builder, targets))
    # Atom encoder helper uses the first frame as its reference.
    af_frames = np.asarray(frames).copy()
    af_frames[0] = frame
    if optimized:
        # Reuse this chain-frame atom tree; query active probes only.
        active = np.flatnonzero(cloud["point_mask"].ravel())
        rid = active // 24
        probes = cloud["point_pos"].reshape(-1, 3)[active] + c[rid]
        d, idx = tree.query(probes, k=8)
        p = props[idx].copy()
        p[:, :, 0] /= 2
        p[:, :, 2] /= 4
        p[:, :, 3] = np.sqrt(p[:, :, 3])
        unit = (probes[:, None, :] - x[idx]) / np.maximum(d[:, :, None], 1e-8)
        align = (unit * cloud["point_normal"].reshape(-1, 3)[active, None, :]).sum(2)
        same = owner[idx] == rid[:, None]
        rbf = np.exp(-0.5 * ((d[:, :, None] - np.array([2, 3, 4, 5, 6, 8])) / 1.0) ** 2)
        atoms = np.zeros((n * 24, 8, len(af.NAMES)), np.float32)
        atoms[active] = np.concatenate((p, rbf, align[:, :, None], same[:, :, None]), 2)
        atoms = atoms.reshape(n, 24, 8, len(af.NAMES))
    else:
        atoms = af.arrays(
            xyz,
            props,
            owner,
            af_frames,
            centers,
            cloud["point_pos"],
            cloud["point_normal"],
            cloud["point_mask"],
        )
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError("Nonfinite surface descriptors; inspect input coordinates")
    return (
        out,
        cloud,
        atoms,
        dict(grid_points=len(points), spacing=spacing, directions=direction_count),
    )
