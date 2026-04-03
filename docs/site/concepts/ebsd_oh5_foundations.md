# EBSD and `.oh5` Foundations

## Experimental Container Assumptions

The repository treats `.oh5` as the source-of-truth EBSD container. The minimal scan capabilities required by most workflows are:

- `/<scan>/EBSD/Header/nColumns`
- `/<scan>/EBSD/Header/nRows`
- `/<scan>/EBSD/Data/Pattern` or `Patterns`

Optional but highly useful scalar fields include:

- `CI` or `Confidence Index`
- `IQ` or `Image Quality`
- `Fit`
- `Valid`
- `Phi1`, `Phi`, `Phi2` for Euler export when available

## Pattern Layouts

The reader supports both common pattern layouts:

1. flattened stack:
   - `(nRows*nColumns, H, W)`
2. grid-preserving stack:
   - `(nRows, nColumns, H, W)`

The point is not just convenience. Preserving both layouts safely prevents accidental loss of scan geometry while still allowing per-pattern access.

## Scan Geometry and Pixel Identity

Every pixel can be referenced either by:

- Cartesian scan coordinates `(x, y)`, or
- a deterministic flattened index

For a scan width $n_x$, the flattened index is

$$
i = y\,n_x + x
$$

and the inverse mapping is

$$
y = \left\lfloor \frac{i}{n_x} \right\rfloor,\qquad x = i \bmod n_x.
$$

This matters for provenance because CSV labels, manifests, GUI maps, and inference outputs all need a common and reversible pixel identity.

## Why aliasing is necessary

Real `.oh5` files are not always field-name stable. That is why the reader resolves aliases like:

- `CI` vs `Confidence Index`
- `IQ` vs `Image Quality`
- `Pattern` vs `Patterns`

Robust aliasing is a scientific reliability measure, not just a convenience feature. It prevents silently missing key fields due to vendor or export naming variation.

## Read-only policy

The repository reads `.oh5` containers as input truth. Downstream artifacts are emitted as separate manifests, CSV/JSON summaries, split bundles, and HTML reports, preserving a clear distinction between:

- original experimental containers
- derived workflow artifacts

For exact structure notes, see the legacy source file `docs/oh5_structure.md`.
