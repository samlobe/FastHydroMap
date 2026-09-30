# Changelog

## 2.0.0a1 — unreleased preview

- Add the selected v2 Fdewet, PC1, PC2 and PC3 ensembles with bundled weights.
- Add explicit HID/HIE/HIP assignments and shared all-output inference.
- Add portable surface/chemical features without PyTorch Geometric or research files.
- Use chain-local geometry references and separate disconnected components.
- Add optional PDB preparation, additive parts, provenance and input checks.
- Add v2 structure/trajectory CLI support under `--model v2`.
- Preserve the published v1 behavior as the preview default.

Known limits: assembly accuracy is not yet MD-validated; hard cavity features
remain sensitive to small coordinate changes, especially at buried sites.
Trajectory preview uses PDB-precision intermediate coordinates. ChimeraX v2
integration and the final publication evaluation remain to be reviewed.
