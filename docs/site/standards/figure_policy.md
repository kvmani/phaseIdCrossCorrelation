# Figure Policy

The repository prefers **version-controlled, deterministic, and editable** figures.

## Preferred forms

- SVG for architecture diagrams, GUI schematics, and algorithm figures
- Mermaid for workflow flowcharts embedded in docs pages

## Avoid by default

- screenshot-only documentation as the primary figure source
- figures that require manual regeneration in external GUI software without a tracked source

## Why

These docs are intended to evolve with the codebase. SVG and Mermaid keep the diagrams:

- diffable in git
- reviewable in pull requests
- easy to refine when workflows or GUIs change
