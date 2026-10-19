# Changelog

## 2.0.0a1 unreleased

- Use the current bundled model by default (`--model latest`).
- Retain the previous published model as `--model v0`.
- Predict Fdewet and PC1–PC3 with one shared residue representation and a
  three-member ensemble.
- Always export all four maps and official residue embeddings for structures.
  `--quantity` selects individual maps only with the legacy model.
- Add histidine assignments,
  heavy-atom preparation, and input/model provenance.
- Retain familiar residue columns and B-factor maps; default all-target exports
  use quantity-suffixed PDBs. Legacy v0
  remains Fdewet-only by default with unchanged names.
- Reject trajectory preparation before inference or output creation.
- Stream trajectory scores into a compact NPZ with residue IDs and frame/time
  information, plus provenance JSON; no trajectory PDBs. CSV and per-frame
  embeddings are opt-in. Stage outputs to protect existing files on prediction failure.
- Retain original PC color limits without clipping numerical outputs.
- Include both C-terminal oxygens and the appropriate terminal chemistry in
  surface features; refit the ensemble on all 800 original training proteins.
- Reject incompatible preview weights and incomplete terminal chemistry.

The current model has no intrinsic/contextual decomposition; use the legacy
model for `--parts`. See [legacy usage](docs/LEGACY.md) for compatibility details.
The corrected candidate is undergoing package and ChimeraX verification.
Public data deposition, attribution review, and cross-platform release checks
remain pending. Nothing has been published.
