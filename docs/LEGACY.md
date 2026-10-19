# Reproducing the previous model

`--model latest` is the default. It selects the model bundled with the installed
package, without downloading or checking for updates. Record the package version
and output JSON to reproduce results.

```bash
fasthydromap predict protein.pdb --model v0 -o old_scores
fasthydromap predict protein.pdb --model v0 --quantity pc1 -o old_pc1
fasthydromap predict protein.pdb --model v0 --parts -o old_parts
```

`v0` identifies the model released in package 0.1.4, not a new training run.
Its weights and `FastHydroMap.predictors.fdewet.FdewetPredictor` API remain
available. For the entire old environment, install `fasthydromap==0.1.4` in a
separate environment and pin dependencies. Old releases and tags are preserved.

The latest model has no intrinsic/context decomposition; `--parts` requires
`--model v0` and never silently switches models. Only latest supports
`--quantity all`, explicit histidine states, preparation and embeddings.

Structure CSVs retain `residue` and selected score columns first, followed by
chain-aware `res_uid`, state and caution columns. Single-chain display labels
retain the old numbering convention; multichain labels include the chain.
Read columns by name rather than requiring exactly two columns. Trajectory
columns follow the same labels; JSON records their mapping to `res_uid`.
Latest structure exports always include all four targets, four quantity-suffixed
PDBs, and the official residue embeddings. Latest trajectories instead produce
a numerical NPZ and JSON, no PDBs; `--csv` adds the optional `_total.csv`.
Legacy `v0` still defaults to Fdewet with unchanged output naming and formats.
Use `--model v0 --quantity fdewet` to reproduce the previous single-score workflow.

The unpublished preview spellings `v1` and `v2` remain unadvertised aliases
for `v0` and `latest`. They are not additional models.
