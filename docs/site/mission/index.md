# Mission and Principles

This section translates the repository mission into explicit documentation-facing guidance for users and contributors.

## Mission

The repository exists to deliver a **scientifically rigorous, modular, and reproducible EBSD analysis framework** for phase discrimination in mixed microstructures where conventional indexing is not reliable enough on its own.

The central operational idea is to preserve two complementary evidence tracks:

1. **NCC-first evidence** based on consistent preprocessing and direct similarity scoring.
2. **ML-first evidence** based on supervised classification from labeled experimental patterns.

This dual-track posture is not redundancy for its own sake. It is how the repository protects against overconfidence:

- NCC remains interpretable and auditable.
- ML can capture discriminative cues that correlation-only logic may miss.
- full provenance on both branches keeps future fusion possible.

## Engineering Principles

::::{div} doc-metric-grid
:::{div} doc-metric
**Correctness first**  
Prefer scientifically reliable decisions over throughput.
:::
:::{div} doc-metric
**Reproducibility first**  
Every runnable workflow should leave machine-readable lineage.
:::
:::{div} doc-metric
**Documentation as product**  
Docs, figures, manifests, and reports are first-class deliverables.
:::
:::{div} doc-metric
**Traceability over convenience**  
Keep reject reasons, split assignments, and evidence artifacts.
:::
::::

## Why The Workflow Is Structured This Way

### Why `.oh5` is treated as source-of-truth

The repository does not invent its own primary EBSD container. It treats `.oh5` as the authoritative experimental source because:

- it preserves the scan grid and associated scalar fields,
- it keeps the analysis grounded in the original acquisition outputs,
- it lets quality filtering, phase assignment, orientation export, and inference remain tied to the same pixel-level provenance.

### Why quality filtering happens early

Low-quality patterns can dominate downstream behavior if they are not excluded early. The dataset-prep workflows therefore evaluate fields such as `CI`, `IQ`, `Fit`, and `Valid` before sample inclusion so the resulting training set and inference diagnostics are scientifically defensible.

### Why balanced datasets matter

When one phase dominates raw accepted counts, a classifier can achieve deceptively strong headline accuracy while still being weak on minority phases. Balancing to the minimum accepted phase count before split assignment is therefore an explicit, auditable policy rather than an implicit convenience.

### Why orientation/IPF diagnostics were added

Balanced counts alone do not guarantee useful orientation coverage. Euler export and IPF diagnostics exist so users can inspect whether train/val/test coverage is acceptably distributed in orientation space rather than clustered around a narrow subset of textures.

## Scientific Scope

The current scope is EBSD-only phase identification, with architecture choices kept compatible with future evidence fusion. For the present stage, the deliverables are:

- stable NCC/Hough evaluation surfaces
- ML dataset preparation, training, benchmarking, and inference surfaces
- cross-linked reports and manifests
- documentation detailed enough to serve as the primary interaction surface for the repo

## Foundational References

- EBSD and indexing reliability context: Nowell and Wright (2005)
- strain/quality and EBSD analysis posture: Wright, Nowell, and Field (2011)
- NCC formulation baseline: Lewis (1995)
- orientation coloring and IPF rationale: Nolze and Hielscher (2016)

See {doc}`../reference/citations` for the repository bibliography.
