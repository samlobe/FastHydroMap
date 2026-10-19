# Release checklist

The current checkout is an unreleased candidate. Public submission follows
review of the exact source commit, model manifest, wheel, and ChimeraX bundle.

## Current model

- Shared 16-dimensional residue encoder, four heads, three ensemble members.
- Final refits use all 800 original training proteins with fixed schedules.
- Both C-terminal oxygens and terminal chemistry are represented.
- The feature-version contract rejects incompatible preview weights.
- Original validation/test cohorts are excluded from fitting and normalization.

The historical 500-fit / 100-tune / 200-audit roles remain unchanged for
development analyses. Corrected-feature ablations and learning curves run
separately; they do not block the software release. Do not substitute older
600-protein results for current-model results in the paper.

## Software release gates

- [ ] Review the intended source diff, README, model notes, and examples.
- [x] Pass the CPU suite and independent feature/readout parity checks.
- [x] Build wheel and source distribution; check metadata and packaged files.
- [x] Install the exact wheel outside the checkout and test structures,
  embeddings, trajectories, preparation, and legacy inference.
- [ ] Run the supported-platform CI matrix or clearly state unverified platforms.
- [x] Verify the training protocol and retained input-archive checksums.
- [ ] Complete dependency, force-field, and data attribution review.
- [ ] Choose the release version and update citation/changelog/status.
- [ ] Review and approve the PR before merging and tagging.

Keep the unpublished README notice until the new package is available on PyPI.
For the release, switch the quick start to PyPI installation. Mark the manifest
as released only at the approved release freeze, record its new hash, refresh
provenance fixtures, and rerun the package checks. Do not alter numerical weights.

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

The October 19 Linux review passed 164 source tests, 164 installed-wheel tests,
and 30-protein feature/readout parity. ChimeraX 1.12 passed 66 focused tests
and seven live GUI checks. macOS and Windows are not qualified by these checks.

Publish and verify the Python backend before submitting the compatible bundle
to the Toolshed. A passing Python wheel does not validate the GUI bridge.
See the bundle repository's `RELEASING.md` for submission and rollback.

## Paper and data follow-up

- [ ] Complete and summarize the scheduled ablations and learning curves.
- [ ] Report current-model held-out and coordinate-sensitivity results with
  their actual evaluation history.
- [ ] Deposit approved reproducibility/data archives and fill access URLs/DOIs.
- [ ] Update manuscript/SI methods, figures, and software/data citations.

Data deposition is pending; inference uses the packaged weights and does not
require the training archive. The verified local archive is described in
[DATA_ACCESS.json](../training/DATA_ACCESS.json).
