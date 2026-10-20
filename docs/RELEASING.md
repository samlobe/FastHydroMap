# Release checklist

Version 2.0.0 is merged and tagged. PyPI and Toolshed publication are separate
steps using checked artifacts. A packaging-only follow-up adds upstream notices
and PyPI installation text without changing the tagged model code or weights.
Record its source commit and artifact checksums with the upload verification.

## Current model

- Shared 16-dimensional residue encoder, four heads, three ensemble members.
- Final refits use all 800 original training proteins with fixed schedules.
- Both C-terminal oxygens and terminal chemistry are represented.
- The feature-version contract rejects incompatible preview weights.
- Original validation/test cohorts are excluded from fitting and normalization.

The historical 500-fit / 100-tune / 200-audit roles remain unchanged for
development analyses. Corrected-feature ablations and learning curves completed
separately; they do not block the software release. Do not substitute older
600-protein results for current-model results in the paper.

## Software release gates

- [x] Review the intended source diff, README, model notes, and examples.
- [x] Pass the CPU suite and independent feature/readout parity checks.
- [x] Build wheel and source distribution; check metadata and packaged files.
- [x] Install the exact wheel outside the checkout and test structures,
  embeddings, trajectories, preparation, and legacy inference.
- [x] Run the supported-platform CI matrix or clearly state unverified platforms.
- [x] Verify the training protocol and retained input-archive checksums.
- [x] Review dependency, force-field, and data attribution; include the
  [parameter-data notice](../THIRD_PARTY_NOTICES.md) in release artifacts.
- [x] Choose the release version and update citation/changelog/status.
- [x] Review and approve the PR before merging and tagging.

The upload candidate uses the PyPI quick start; retain the Toolshed-only notice
until the compatible bundle is published. The manifest is already marked as
released at the approved freeze. Do not alter numerical weights or move the
existing release tag when completing publication metadata.

The previous published checkpoints and releases remain available. Test
`--model v0`; never replace or delete an old PyPI release.

## ChimeraX

The corrected 800-protein backend requires a fresh bundle check; the earlier
600-protein review is historical evidence only.

- [x] Pass focused bundle tests and build the exact candidate.
- [x] Use an isolated supported ChimeraX installation/profile.
- [x] Compare all four maps and embedding exports with the Python CLI.
- [x] Check missing OXT, preparation, and rejection of incompatible backends.
- [x] Check cache reuse/invalidation and internal-coordinate behavior.
- [x] Check upgrades, failed-update retention, and installation instructions.

The October 19 cross-platform run passed all 14 CI jobs, including the Windows
newline correction. The local suite contains 165 tests; the earlier installed
candidate passed 164 tests before that checkout regression was added.
The 30-protein feature/readout parity check passed. ChimeraX 1.12 on Linux
passed 66 focused tests and seven live GUI checks; that GUI check does not
qualify ChimeraX on macOS or Windows.

Publish and verify the Python backend before submitting the compatible bundle
to the Toolshed. A passing Python wheel does not validate the GUI bridge.
See the bundle repository's `RELEASING.md` for submission and rollback.

## Paper and data follow-up

- [x] Complete the scheduled ablations and learning-curve summary tables.
- [ ] Report current-model held-out and coordinate-sensitivity results with
  their actual evaluation history.
- [ ] Deposit approved reproducibility/data archives and fill access URLs/DOIs.
- [ ] Update manuscript/SI methods, figures, and software/data citations.

Data deposition is pending; inference uses the packaged weights and does not
require the training archive. The verified local archive is described in
[DATA_ACCESS.json](../training/DATA_ACCESS.json).
