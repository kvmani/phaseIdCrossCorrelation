# Documentation Governance

This repository treats documentation as part of the product.

## Canonical rule

The Sphinx site under `docs/site/` is the canonical user-facing documentation surface. Root README files and legacy Markdown docs support it, but they do not replace it.

## Required updates

Any change affecting:

- behavior
- commands
- configs
- data contracts
- artifacts
- GUI layout or meaning
- scientific assumptions

must update the relevant Sphinx pages in the same change.

## Version-controlled figures

New workflow and GUI help should prefer deterministic, editable assets:

- SVG for figures and schematic GUI views
- Mermaid for flow charts

## Why this standard exists

The repository is intended to be used primarily through documentation. That means the documentation must be complete enough to function as an interface layer, not just a narrative supplement.
