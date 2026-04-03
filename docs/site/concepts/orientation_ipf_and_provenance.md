# Orientation, IPF Diagnostics, and Provenance

## Why Euler Angles Are Exported

Balanced phase counts are necessary but not sufficient. Two balanced datasets can still differ dramatically in orientation coverage.

That is why dataset preparation now exports per-sample Euler angles, together with:

- phase name
- label
- split
- source scan ID
- `.oh5` path
- pixel coordinates
- quality values

These exports make orientation-space inspection part of ordinary dataset validation rather than a post hoc forensic step.

## Euler Convention

The exported Euler convention is recorded explicitly as **Bunge ZXZ**. The raw `.oh5` values are read from:

- `Phi1`
- `Phi`
- `Phi2`

When present, the reader detects whether the source values are in radians or degrees and normalizes the exported values to degrees for downstream CSV/JSON consistency.

## Why IPF Scatter Diagnostics Matter

Inverse pole figure diagnostics let users inspect whether:

- train/val/test splits cover orientation space similarly,
- one phase has strong clustering that may overfit the classifier,
- balancing by count still leaves undesirable orientation bias.

This is especially important in EBSD because strong texture or restricted orientation sampling can make a dataset look larger and more diverse than it really is.

## Confidence and provenance together

The repository intentionally couples confidence-bearing outputs with provenance-bearing metadata. In practice that means:

- split assignments are recorded
- quality gates are recorded
- source scan and source file lineage are recorded
- orientation exports are linked from manifests and HTML summaries

The goal is not just reproducibility in a narrow software sense. It is scientific defensibility: a user must be able to answer **which pixels were used, why they were accepted, what orientation they represented, and which split/model/report they influenced**.

## Relevant Reference

For orientation coloring and IPF mapping principles, see Nolze and Hielscher (2016).
