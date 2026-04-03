# `.oh5` Contracts

This page is the Sphinx-facing summary of the repository's `.oh5` expectations.

## Required scan structure

- scan group containing `EBSD/Header` and `EBSD/Data`
- `nColumns`
- `nRows`
- `Pattern` or `Patterns`

## Optional scalar fields

- `CI` / `Confidence Index`
- `IQ` / `Image Quality`
- `Fit`
- `Valid`
- Euler triplet `Phi1`, `Phi`, `Phi2`

## Why these fields matter

- `Pattern`: core experimental signal
- `CI`, `IQ`, `Fit`, `Valid`: quality acceptance logic
- Euler triplet: orientation provenance and IPF diagnostics

## Exact structure notes

For the lower-level structure reference and aliasing notes, see the legacy source file `docs/oh5_structure.md`.
