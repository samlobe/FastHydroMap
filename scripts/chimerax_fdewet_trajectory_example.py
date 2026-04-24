"""
Teaching example: color a trajectory with per-residue FastHydroMap values in ChimeraX.

Typical workflow:

1. Run FastHydroMap on your trajectory:

       fasthydromap predict-trajectory your_topology.pdb your_traj.dcd -o outputs/mytraj --parts

2. Edit the user settings below:
   - PDB_PATH
   - TRAJECTORY_PATH
   - CSV_PATH
   - optionally MOVIE_OUTPUT

3. Run from ChimeraX (or command line):
       chimerax --script chimerax_fdewet_trajectory_example.py

Notes:
- Use the ``*_total.csv`` file for total Fdewet values.
- If you ran ``--parts``, you can swap to ``*_intrinsic.csv`` or
  ``*_context.csv`` to visualize those components instead.
- This script is intentionally simple and comment-heavy so a future researcher
  can edit it without needing to know much ChimeraX scripting beforehand.
"""

from __future__ import annotations

import csv
import math
import pathlib

from chimerax.core.commands import run


# ---------------------------------------------------------------------------
# User-editable settings
# ---------------------------------------------------------------------------
# Point these three paths at your own files.
WORKING_DIR = pathlib.Path("~/Research/FastHydroMap_development/FastHydroMap").expanduser()
PDB_PATH = WORKING_DIR / "examples" / "proteinG.pdb"
TRAJECTORY_PATH = WORKING_DIR / "examples" / "proteinG_short.dcd"
CSV_PATH = WORKING_DIR / "outputs" / "proteinG_fdewet_total.csv"

# Optional movie output. Set to None if you only want an interactive preview.
MOVIE_OUTPUT = None
# Example:
# MOVIE_OUTPUT = pathlib.Path.home() / "Desktop" / "proteinG_fdewet.mp4"

# Display settings you may want to tweak.
MODEL_ID = "#1"
PALETTE = "lipophilicity"
COLOR_RANGE = (4.0, 6.5)   # (low, high)
FRAME_START = 0
FRAME_STOP = None          # None = go to the last CSV frame
FRAME_STEP = 1
WAIT_FRAMES = 1
SUPERSAMPLE = 1            # movie antialiasing quality
RUN_DSSP_EVERY = 10        # set to 0 to skip DSSP updates
CLOSE_EXISTING_MODELS = True
BACKGROUND_COLOR = "white"


def load_fdewet_table(path: pathlib.Path):
    """
    Read a FastHydroMap trajectory CSV.

    Expected format:
        frame,<residue label 1>,<residue label 2>,...

    Returns a list of ``(frame_index, {residue_label: value})`` rows.
    """
    rows = []
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if not header or header[0] != "frame":
            raise ValueError(
                f"{path} does not look like a FastHydroMap trajectory CSV "
                "(expected first column to be 'frame')."
            )
        residue_labels = header[1:]
        for line in reader:
            frame_idx = int(line[0])
            values = [float(v) for v in line[1:]]
            rows.append((frame_idx, dict(zip(residue_labels, values))))
    if not rows:
        raise ValueError(f"{path} had no frame rows.")
    return rows


def residue_label_to_spec(model_id: str, label: str) -> str:
    """
    Convert a FastHydroMap CSV residue label into a ChimeraX atom spec.

    Common cases:
    - "53"    -> "#1:53"
    - "53A"   -> "#1:53A"
    - "A:53"  -> "#1/A:53"
    - "A:53A" -> "#1/A:53A"

    If your residue labels use a different style, this is the helper to edit.
    """
    if ":" in label:
        chain_id, residue_id = label.split(":", 1)
        return f"{model_id}/{chain_id}:{residue_id}"
    return f"{model_id}:{label}"


def apply_frame_values(model_id: str, residue_values: dict[str, float]) -> None:
    for residue_label, value in residue_values.items():
        if not math.isfinite(value):
            continue
        residue_spec = residue_label_to_spec(model_id, residue_label)
        run(session, f"setattr {residue_spec} a bfactor {value:.2f}")


def main() -> None:
    if CLOSE_EXISTING_MODELS:
        run(session, "close all")

    for path in (PDB_PATH, TRAJECTORY_PATH, CSV_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    fdewet_rows = load_fdewet_table(CSV_PATH)

    # Open topology and attach the trajectory. With CLOSE_EXISTING_MODELS=True,
    # this will normally be model #1.
    run(session, f"open {PDB_PATH}")
    run(session, f"open {TRAJECTORY_PATH} structureModel {MODEL_ID}")

    # Basic, readable defaults. Feel free to edit or delete these.
    run(session, "cartoon")
    run(session, "hide atoms")
    run(session, f"set bgColor {BACKGROUND_COLOR}")
    run(session, f"color {MODEL_ID} light gray")

    lo, hi = COLOR_RANGE
    frame_stop = FRAME_STOP if FRAME_STOP is not None else fdewet_rows[-1][0]
    rows_to_show = [
        (frame_idx, residue_values)
        for frame_idx, residue_values in fdewet_rows
        if FRAME_START <= frame_idx <= frame_stop and (frame_idx - FRAME_START) % FRAME_STEP == 0
    ]

    if MOVIE_OUTPUT is not None:
        run(session, "movie stop")
        run(session, "movie reset")
        run(session, f"movie record supersample {SUPERSAMPLE}")

    for i, (frame_idx, residue_values) in enumerate(rows_to_show):
        # ChimeraX coordsets are 1-based; FastHydroMap CSV frame numbers are 0-based.
        run(session, f"coordset {MODEL_ID} {frame_idx + 1}")
        apply_frame_values(MODEL_ID, residue_values)
        run(session, f"color bfactor palette {PALETTE} range {hi},{lo}")

        if RUN_DSSP_EVERY and i % RUN_DSSP_EVERY == 0:
            run(session, f"dssp {MODEL_ID} report false")

        run(session, f"wait {WAIT_FRAMES}")

    if MOVIE_OUTPUT is not None:
        run(session, "movie stop")
        run(session, f"movie encode {MOVIE_OUTPUT} bitrate 4000")
        print(f"\nMovie written to: {MOVIE_OUTPUT}\n")
    else:
        print(
            "\nFinished previewing trajectory coloring.\n"
            "Tip: set MOVIE_OUTPUT to a .mp4 path if you want to save a movie.\n"
        )


main()
