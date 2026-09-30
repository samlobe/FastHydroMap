"""Command-line integration for the v2 preview."""

import json
import hashlib
from pathlib import Path
import tempfile
import warnings
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO
from ..io.pdb import write_bfactor
from .geometry import residue_uid_from_biopdb
from .predictor import Predictor, TARGETS, WEIGHTS


def prepare(source, destination):
    """Add missing atoms, not residues, and retain chain/residue identifiers."""
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    fixer = PDBFixer(filename=str(source))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    if fixer.nonstandardResidues:
        raise ValueError(
            "Preparation does not convert nonstandard chemistry; supply a standard protein PDB"
        )
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=17)
    with Path(destination).open("w") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    warnings.warn(
        "Prepared heavy atoms were written alongside the predictions; inspect any rebuilt regions",
        stacklevel=2,
    )


def run(args):
    targets = TARGETS if args.quantity == "all" else [args.quantity]
    predictor = Predictor(targets, threads=args.threads)
    states = None
    if args.histidine_states:
        table = pd.read_csv(args.histidine_states, keep_default_na=False)
        if (
            not {"res_uid", "histidine_state"}.issubset(table.columns)
            or table.res_uid.duplicated().any()
        ):
            raise ValueError(
                "Histidine CSV needs unique res_uid and histidine_state columns"
            )
        states = dict(zip(table.res_uid, table.histidine_state))
    trajectory = args.dcd is not None
    if trajectory and args.prepare:
        raise ValueError(
            "Do not repair a trajectory topology during prediction; atom counts and order must already match"
        )
    root = args.outroot or args.pdb.with_name(
        args.pdb.stem + "_" + args.quantity + ("_traj" if trajectory else "")
    )
    root = Path(root)
    outputs = [Path(str(root) + ".csv"), Path(str(root) + ".json")]
    if not trajectory:
        outputs += [
            Path(
                str(root) + ("_" + t.lower() if args.quantity == "all" else "") + ".pdb"
            )
            for t in predictor.quantities
        ]
    if args.prepare:
        outputs.append(Path(str(root) + "_prepared.pdb"))
    if trajectory and args.parts:
        outputs += [
            Path(str(root) + "_" + part + ".csv") for part in ("intrinsic", "context")
        ]
    # Never overwrite an input, even with --overwrite.
    inputs = [args.pdb, args.dcd, args.histidine_states]
    if any(
        o.resolve() == Path(i).resolve()
        for o in outputs
        for i in inputs
        if i is not None
    ):
        raise ValueError("Output paths must not replace input files")
    if not args.overwrite and any(p.exists() for p in outputs):
        raise ValueError(
            "Output already exists; choose another --outroot or use --overwrite"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    source = args.pdb
    if args.prepare:
        # Preserve explicit histidine labels even if PDBFixer normalizes names.
        structure = PDBParser(QUIET=True).get_structure("", str(source))
        named = {
            residue_uid_from_biopdb(r): r.resname
            for r in structure.get_residues()
            if r.resname in ("HID", "HIE", "HIP")
        }
        states = {**named, **(states or {})}
        source = Path(str(root) + "_prepared.pdb")
        prepare(args.pdb, source)
    if not trajectory:
        result = predictor.predict(source, histidine_states=states, parts=args.parts)
        result.to_csv(outputs[0], index=False)
        for target in predictor.quantities:
            dest = Path(
                str(root)
                + ("_" + target.lower() if args.quantity == "all" else "")
                + ".pdb"
            )
            write_bfactor(source, result[target].to_numpy(), dest)
        count = 1
    else:
        import mdtraj as md

        structure = PDBParser(QUIET=True).get_structure("", str(source))
        if len(structure) != 1:
            raise ValueError("Trajectory topology must have a single PDB model")
        atoms = list(structure[0].get_atoms())
        count = 0
        expected_ids = None
        with tempfile.TemporaryDirectory(prefix="fasthydromap_frames_") as tmp:
            current = Path(tmp) / "frame.pdb"
            for block in md.iterload(str(args.dcd), top=str(source), chunk=1):
                if block.n_atoms != len(atoms):
                    raise ValueError(
                        "Trajectory topology atom order/count is ambiguous; remove alternate locations first"
                    )
                for atom, xyz in zip(atoms, block.xyz[0] * 10):
                    atom.coord = xyz
                writer = PDBIO()
                writer.set_structure(structure)
                writer.save(str(current))
                result = predictor.predict(
                    current, histidine_states=states, parts=args.parts
                )
                ids = result.res_uid.tolist()
                if expected_ids is not None and ids != expected_ids:
                    raise ValueError("Residue identities changed between frames")
                expected_ids = ids
                for part in (
                    ["total", "intrinsic", "context"] if args.parts else ["total"]
                ):
                    values = {"frame": count}
                    for target in predictor.quantities:
                        key = target if part == "total" else target + "_" + part
                        values.update(
                            {
                                (
                                    uid
                                    if len(predictor.quantities) == 1
                                    else target + ":" + uid
                                ): float(v)
                                for uid, v in zip(ids, result[key])
                            }
                        )
                    dest = Path(
                        str(root) + ("" if part == "total" else "_" + part) + ".csv"
                    )
                    pd.DataFrame([values]).to_csv(
                        dest,
                        index=False,
                        mode="w" if count == 0 else "a",
                        header=count == 0,
                    )
                count += 1
        if count == 0:
            raise ValueError("Trajectory contains no frames")
    provenance = dict(
        model="v2",
        model_manifest_sha256=hashlib.sha256(
            (WEIGHTS / "manifest.json").read_bytes()
        ).hexdigest(),
        threads=args.threads,
        status="preview",
        quantities=predictor.quantities,
        frames=count,
        input=str(args.pdb),
        prepared=args.prepare,
        input_sha256=hashlib.sha256(Path(args.pdb).read_bytes()).hexdigest(),
        analyzed_structure_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        trajectory=str(args.dcd) if trajectory else None,
        histidine_states=dict(
            zip(
                result.res_uid[result.aa.eq("H")],
                result.histidine_state[result.aa.eq("H")],
            )
        ),
        checkpoints={
            t: predictor.manifest["models"][t]["members"] for t in predictor.quantities
        },
        limitations="Single-chain training; assemblies exploratory; buried/terminal residues and coordinate quantization require caution. See docs/V2.md.",
    )
    outputs[1].write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"FastHydroMap v2: wrote {count} frame(s) to {root}.csv")
