# v2 release checklist

This branch is a reviewable preview (`2.0.0a1`), not a published release.
It starts from GitHub main at `add292f` (0.1.4). During review, v1 remains the
default; v2 is explicit with `--model v2`. The final default switch is a deliberate
release decision, not an undocumented change inside the packaging work.

## Included

- Portable inference, twelve selected checkpoints and checksum verification.
- Shared extraction for Fdewet/PC1/PC2/PC3; no research dependencies.
- Histidine assignments, stable residue IDs, additive parts and provenance.
- PDB preparation, structure maps and streaming trajectory preview.
- Chain-local reference frames and disconnected-component handling.
- v1 compatibility, regression tests and wheel data declarations.

## Before a final release

- Review numerical parity against the selected research implementation.
- Run old/new tests on supported Python versions and CI platforms, including macOS.
- Test the installed wheel outside the checkout and inspect its contents.
- Review preparation, alternate locations, insertion codes, explicit histidine
  names, chain termini and mixed-complex behavior.
- Review trajectory atom identity and coordinate precision.
- Decide user-facing handling of buried sites and coordinate sensitivity.
- Complete or explicitly defer assembly MD validation; make no accuracy claim.
- Review evaluation history and publication metrics separately from packaging.
- Publish the curated v2 training recipe, dataset provenance/splits and data
  access instructions; the old training folder is not the v2 reproduction path.
- Confirm force-field/data attribution and distribution permissions.
- Verify ChimeraX integration; the existing bundle is not a v2 validation.
- Decide final version/default behavior, update release notes/citation, and tag.

## Local verification

The full Python 3.12 test suite, including histidine-ID and two-frame trajectory
checks, passes locally. Portable v2 predictions agree with the
research runtime on 100 development structures plus five assembly/perturbation
fixtures, with a maximum absolute difference of 2.87e-6 across all four outputs.
This is an implementation-parity check, not label-based accuracy evaluation.

The wheel and source archive build locally. An installed-wheel test from outside
the checkout loads all four outputs without research imports or PyTorch Geometric.
The verification venv shares the existing CPU scientific dependencies; it is not
evidence of a completely fresh dependency resolution on every supported platform.

## Suggested PR summary

“Add the FastHydroMap v2 preview: improved labels and histidine-aware surface
models, portable CPU inference, shared all-output prediction, and regression
tests. Preserve v1 behavior and document remaining validation work.”

Do not include research caches, trajectories, experimental models, local paths,
or personal credentials. Nothing in this branch publishes to GitHub/PyPI or
changes main automatically.
