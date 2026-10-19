"""Bounded-memory trajectory export to an ordinary, pickle-free NumPy archive."""

from pathlib import Path
import shutil
import zipfile

import numpy as np


class TrajectoryArchive:
    """Spool frames to temporary disk; serialize compressed NPY members at finish.

    The caller owns the temporary directory and publishes the archive only after
    successful completion. Memory is proportional to one frame, not frame count.
    """

    def __init__(self, directory, targets, embeddings=False):
        self.directory = Path(directory)
        self.targets = list(targets)
        self.embeddings = embeddings
        self.count = 0
        self.ids = None
        self.fields = {}

    def append(self, result, time_ps):
        ids = result.res_uid.astype(str).tolist()
        if self.ids is not None and ids != self.ids:
            raise ValueError("Residue identities changed between frames")
        self.ids = ids
        arrays = {
            "scores": np.asarray(result[self.targets], dtype="<f4"),
            "frame": np.asarray(self.count, dtype="<i8"),
            "time_ps": np.asarray(time_ps, dtype="<f8"),
        }
        if self.embeddings:
            arrays["embedding"] = np.asarray(
                result[[f"embedding_{j}" for j in range(16)]], dtype="<f4"
            )
        for name, array in arrays.items():
            spec = (array.dtype, array.shape)
            if name in self.fields and self.fields[name] != spec:
                raise ValueError(f"Trajectory {name} shape changed between frames")
            self.fields[name] = spec
            with (self.directory / (name + ".raw")).open("ab") as handle:
                handle.write(array.tobytes(order="C"))
        self.count += 1

    def finish(self, destination, metadata=None):
        if not self.count:
            raise ValueError("Trajectory contains no frames")
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED,
                             allowZip64=True) as archive:
            for name, (dtype, shape) in self.fields.items():
                with archive.open(name + ".npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array_header_2_0(member, {
                        "descr": np.lib.format.dtype_to_descr(dtype),
                        "fortran_order": False,
                        "shape": (self.count, *shape),
                    })
                    with (self.directory / (name + ".raw")).open("rb") as raw:
                        shutil.copyfileobj(raw, member, length=1024 * 1024)
            small = {"res_uid": np.asarray(self.ids, dtype=str),
                     "targets": np.asarray(self.targets, dtype=str),
                     **(metadata or {})}
            for name, value in small.items():
                with archive.open(name + ".npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, np.asarray(value), allow_pickle=False)
