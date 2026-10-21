# Releasing FastHydroMap

The Python package and ChimeraX bundle have separate version numbers and release
workflows. Publish and verify a compatible Python backend before releasing a
bundle that requires it.

## Python package

1. Update the package version, changelog and citation metadata.
2. Run the CPU test suite and supported-platform CI checks.
3. Build the wheel and source distribution with `scripts/build_release.sh`.
4. Install the exact wheel outside the checkout with
   `scripts/test_wheel_install.sh /path/to/wheel.whl`. Check structure maps,
   embeddings, preparation, trajectories and legacy inference.
5. Review the PR, wait for CI, merge, and tag the reviewed commit. Never move a
   published tag or replace files in an existing PyPI release.
6. Upload the checked distributions to PyPI, then download and verify them.
7. Publish a GitHub Release for the same tag and attach the wheel and source
   distribution. A Git tag alone does not trigger Zenodo archiving.

The Zenodo GitHub integration must be enabled for `samlobe/FastHydroMap`.
Check the resulting software record and DOI after publishing the GitHub Release.
Keep previous software versions available.
Zenodo uses `.zenodo.json` for archive metadata, including the dataset link;
keep its version and creators consistent with `CITATION.cff`.

## Model identity

The shared model has a 16-dimensional residue encoder, four prediction heads,
and three ensemble members fitted on 800 proteins. Validation and test proteins
are excluded from fitting and normalization.

A documentation or packaging patch does not require retraining or rewriting
checkpoints. Version 2.0.1 retains the 2.0.0 model files and manifest byte for
byte. The model's recorded release version therefore remains 2.0.0, while the
installed Python package reports 2.0.1. Checkpoint and manifest checksums identify
the numerical model independently of the package version.

## ChimeraX bundle

Follow the bundle repository's `RELEASING.md`. Build and install the actual wheel,
restart ChimeraX, and exercise the public installation workflow. Check all four
maps, cached recoloring, embedding export, histidine assignments, missing-atom
errors and optional preparation. Confirm that existing open structures are not
modified and that failed backend updates preserve a working installation.

The portable application smoke script is `tests/chimerax_smoke.py` in the bundle
repository. Linux application tests and platform-mocked unit tests do not
establish native macOS or Windows application compatibility.

After uploading, verify the public wheel checksum and installation from the
Toolshed. Check its machine-readable install catalog as well as its website:
the catalog may not immediately advertise a newly downloadable version.

## Dataset deposition

Keep large simulation and training archives out of Git. The repository contains
`training/proteins.csv`, the fitting protocol, and `training/DATA_ACCESS.json`.
Deposit the labels/structures and exact precomputed fitting inputs as a separate
Zenodo dataset record, with confirmed creators, license and third-party notices.
These labels come from HydroMap simulations, not FastHydroMap inference.

Before publication, verify:

- All 1,000 proteins and their final 800/100/100 split assignments.
- Source and normalized chain IDs, structure hashes, and histidine assignments.
- Three replicas per protein and reproducible mean, sample SD and SEM values.
- Complete archive contents and checksums, with no full trajectories included.
- Rights and attribution for original data, PDB structures and third-party material.

Reserve the dataset DOI in a Zenodo draft, upload the verified archives, review
the metadata and publish. Fill the DOI and download URLs in
`training/DATA_ACCESS.json`; do not claim public data availability before the
record is actually published. Link the software and dataset records and update
the paper's software/data availability statements.
