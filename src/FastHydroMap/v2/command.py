"""Command-line outputs for the current bundled model."""

import json
import hashlib
from pathlib import Path
import tempfile
import warnings
from copy import copy
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO
from ..io.pdb import write_bfactor
from .geometry import residue_uid_from_biopdb
from .predictor import Predictor, TARGETS, WEIGHTS
from .display import PC_DISPLAY_RANGES
from .trajectory import TrajectoryArchive


def output_paths(root, trajectory, csv=False, prepare=False):
    """The complete export contract, shared by preflight and staged publishing."""
    suffixes = [".npz", ".json"] if trajectory else [
        ".csv", ".json", "_embeddings.npz",
        "_fdewet.pdb", "_pc1.pdb", "_pc2.pdb", "_pc3.pdb",
    ]
    if trajectory and csv:
        suffixes.append("_total.csv")
    if prepare:
        suffixes.append("_prepared.pdb")
    return [Path(str(root) + suffix) for suffix in suffixes]


def residue_labels(result):
    """Preserve the legacy display column; res_uid always retains chain identity."""
    if result.chain_id.nunique() > 1:
        return result.res_uid.astype(str).tolist()
    return [f"{int(r)}{i}" for r, i in zip(result.resid, result.insertion_code)]


def prepare(source, destination):
    """Add missing atoms, not residues, and retain chain/residue identifiers."""
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    structure = PDBParser(QUIET=True).get_structure("", str(source))
    if len(structure) != 1:
        raise ValueError("Provide one PDB model at a time, including with --prepare")
    if any(r.resname in {"ACE", "NME", "NHE", "NH2"} for r in structure.get_residues()):
        raise ValueError("Preparation does not remove peptide caps; supply an uncapped protein structure")
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
    """Validate destinations, then stage everything before replacing outputs."""
    trajectory = args.dcd is not None
    if trajectory and args.prepare:
        raise ValueError(
            "--prepare is only available for a single structure, not trajectories. "
            "Use a complete topology whose atom count and order match the trajectory."
        )
    if args.quantity != "all":
        raise ValueError("The latest model always exports all four targets; omit --quantity")
    basename = Path(args.dcd if trajectory else args.pdb)
    root = Path(args.outroot or basename.with_name(
        basename.stem + ("_maps_traj" if trajectory else "_maps")))
    outputs = output_paths(root, trajectory, getattr(args, "csv", False), args.prepare)
    inputs = [args.pdb, args.dcd, args.histidine_states]
    if any(o.resolve() == Path(i).resolve() for o in outputs for i in inputs if i is not None):
        raise ValueError("Output paths must not replace input files")
    if any(p.exists() and not p.is_file() for p in outputs):
        raise ValueError("Output destination is not a regular file")
    if not args.overwrite and any(p.exists() for p in outputs):
        raise ValueError("Output already exists; choose another --outroot or use --overwrite")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fasthydromap_", dir=root.parent) as tmp:
        staged = copy(args)
        staged.outroot = Path(tmp) / root.name
        _run(staged)
        staged_outputs = output_paths(staged.outroot, trajectory, getattr(args, "csv", False), args.prepare)
        # Provenance is the last published file. No outputs change if inference fails.
        pairs = list(zip(staged_outputs, outputs))
        for source, destination in sorted(pairs, key=lambda pair: pair[1].suffix == ".json"):
            source.replace(destination)
    print(f"FastHydroMap: wrote {'trajectory archive' if trajectory else 'all four maps and embeddings'} to {outputs[0]}")


def _run(args):
    trajectory = args.dcd is not None
    embeddings = not trajectory or getattr(args, 'embeddings', False)
    predictor = Predictor(TARGETS, threads=args.threads)
    if args.parts and predictor.compact:
        raise ValueError("The latest model has no intrinsic/context decomposition; omit --parts or use --model v0.")
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
    root = Path(args.outroot)
    csv_path = Path(str(root) + ("_total" if trajectory else "") + ".csv")
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
        result = predictor.predict(source, histidine_states=states, parts=args.parts, embeddings=embeddings)
        columns = [f'embedding_{j}' for j in range(16)] if embeddings else []
        table = result.drop(columns=columns).copy()
        table.insert(0, "residue", residue_labels(result))
        first = ["residue", *predictor.quantities]
        table[first + [c for c in table if c not in first]].to_csv(csv_path, index=False)
        if embeddings:
            np.savez_compressed(Path(str(root) + '_embeddings.npz'),
                embedding=result[columns].to_numpy(np.float32),res_uid=result.res_uid.to_numpy(str),
                checkpoint_sha256=np.array(result.attrs['embedding_checkpoint_sha256']),
                manifest_sha256=np.array(result.attrs['model_manifest_sha256']),
                seed=np.array(result.attrs['embedding_seed']),
                representation=np.array(result.attrs['embedding_representation']),
                model_status=np.array(result.attrs['model_status']))
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
        if any(a.is_disordered() for a in atoms):
            raise ValueError("Resolve alternate atom locations in the trajectory topology first")
        count = 0
        with tempfile.TemporaryDirectory(prefix="frames_", dir=root.parent) as tmp:
            current = Path(tmp) / "frame.pdb"
            archive = TrajectoryArchive(tmp, predictor.quantities, embeddings)
            for block in md.iterload(str(args.dcd), top=str(source), chunk=1):
                if block.n_atoms != len(atoms):
                    raise ValueError(
                        "Trajectory topology atom order/count is ambiguous; remove alternate locations first"
                    )
                if not np.isfinite(block.xyz).all():
                    raise ValueError(f"Trajectory frame {count} contains nonfinite coordinates")
                for atom, xyz in zip(atoms, block.xyz[0] * 10):
                    atom.coord = xyz
                writer = PDBIO()
                writer.set_structure(structure)
                writer.save(str(current))
                result = predictor.predict(
                    current, histidine_states=states, embeddings=embeddings
                )
                archive.append(result, float(block.time[0]))
                if getattr(args, "csv", False):
                    values = {"frame": count}
                    for target in predictor.quantities:
                        values.update({target + ":" + uid: float(v)
                                       for uid, v in zip(residue_labels(result), result[target])})
                    pd.DataFrame([values]).to_csv(
                        csv_path,
                        index=False,
                        mode="w" if count == 0 else "a",
                        header=count == 0,
                    )
                count += 1
            if count == 0:
                raise ValueError("Trajectory contains no frames")
            metadata = {"manifest_sha256": result.attrs["model_manifest_sha256"]}
            if embeddings:
                metadata.update({k: v for k, v in result.attrs.items() if k.startswith("embedding_")})
            archive.finish(Path(str(root) + ".npz"), metadata)
    provenance = dict(
        model="latest",
        model_alias=getattr(args, "model", "latest"),
        architecture=predictor.manifest.get("architecture", "legacy_full_surface"),
        feature_version=result.attrs["feature_version"],
        training_stage=predictor.manifest.get("training_stage"),
        fit_proteins=predictor.manifest.get("fit_proteins"),
        model_manifest_sha256=hashlib.sha256(
            (WEIGHTS / "manifest.json").read_bytes()
        ).hexdigest(),
        threads=args.threads,
        status=predictor.manifest.get('status', 'preview'),
        embeddings_exported=embeddings,
        embedding_metadata={k:v for k,v in result.attrs.items() if k.startswith('embedding_')} if embeddings else None,
        quantities=predictor.quantities,
        display_ranges=PC_DISPLAY_RANGES,
        numerical_outputs="raw; display ranges saturate colors only",
        residue_ids={label: uid for label, uid in zip(residue_labels(result), result.res_uid)},
        frames=count,
        input=str(args.pdb),
        prepared=args.prepare,
        input_sha256=hashlib.sha256(Path(args.pdb).read_bytes()).hexdigest(),
        analyzed_structure_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        trajectory=str(args.dcd) if trajectory else None,
        trajectory_format={
            "schema_version": 1, "scores_axes": ["frame", "residue", "target"],
            "frame_index": "zero-based", "time_ps": "MDTraj-reported; may be inferred from format defaults",
            "embedding_axes": ["frame", "residue", "feature"] if embeddings else None,
            "csv_exported": getattr(args, "csv", False),
        } if trajectory else None,
        histidine_states=dict(
            zip(
                result.res_uid[result.aa.eq("H")],
                result.histidine_state[result.aa.eq("H")],
            )
        ),
        checkpoints=predictor.manifest['members'] if predictor.shared else {
            t: predictor.manifest["models"][t]["members"] for t in predictor.quantities
        },
        limitations="Single-chain training; assemblies exploratory; structural cautions are not confidence estimates. See docs/MODEL.md.",
        terminal_caution_note=result.attrs["terminal_caution_note"],
    )
    Path(str(root) + ".json").write_text(json.dumps(provenance, indent=2) + "\n")
