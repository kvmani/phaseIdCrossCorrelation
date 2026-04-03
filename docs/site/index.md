# phaseIdCrossCorrelation Documentation

::::{div} doc-hero
This Sphinx site is the canonical documentation surface for the repository. It is intended to be the main entry point for both first-time users and advanced users who need to run, inspect, and extend the EBSD phase-identification workflows.
::::

This repository is organized around two complementary EBSD evidence tracks:

- **Track A: masked normalized cross-correlation (NCC)** against externally simulated or curated references.
- **Track B: supervised ML classification** from experimental Kikuchi patterns extracted from `.oh5` containers.

The documentation is intentionally dual-purpose:

- **workflow help**: exact commands, configuration guidance, report interpretation, and GUI usage
- **scientific foundation**: rationale, notation, algorithm formulations, traceability requirements, and references

:::{figure} figures/dual_track_architecture.svg
:alt: Dual-track architecture schematic
:width: 100%

Dual-track architecture: raw `.oh5` data enters NCC/Hough and ML branches, each preserving evidence and provenance for future fusion.
:::

```{mermaid}
flowchart TD
    A[Raw .oh5 scans] --> B[Quality-aware ingestion]
    B --> C[NCC and Hough evaluation]
    B --> D[ML dataset preparation]
    D --> E[Training and benchmark suite]
    E --> F[Inference CLI and GUI]
    C --> G[Evidence reports]
    F --> G
    G --> H[Scientific decision support]
```

## Start Here

```{toctree}
:maxdepth: 2
:caption: Documentation Home

getting_started/index
mission/index
concepts/index
workflows/index
guis/index
data_contracts/index
reports/index
reference/index
standards/index
```

## Quick Navigation

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} First Successful Run
:link: getting_started/index.html
Install the docs toolchain, build the site, and run your first debug workflow with exact commands.
:::

:::{grid-item-card} ML Dataset Preparation
:link: workflows/dataset_preparation.html
Understand balanced dataset preparation, Euler/IPF diagnostics, split policies, and manifest outputs.
:::

:::{grid-item-card} Benchmark Suite and Full Cycle
:link: workflows/benchmark_and_full_cycle.html
See how base-train configs, suite configs, and full-cycle orchestration fit together.
:::

:::{grid-item-card} Inference and GUIs
:link: guis/inference_gui.html
Learn single-image prediction, full-scan `.oh5` mapping mode, explorer GUI, and diagnostic gallery usage.
:::
::::

## What This Site Covers

- mission goals, scientific scope, and engineering principles
- EBSD `.oh5` semantics and data contracts
- mathematically important formulations for NCC, split logic, and orientation/IPF diagnostics
- exact commands for scripts under `scripts/`
- GUI help with schematic SVG layouts
- report interpretation for manifests, summaries, suite HTML, and inference outputs
- standards that future tasks must follow when touching code or behavior

## Review and Build Commands

Build the HTML site:

```powershell
python .\scripts\build_docs.py --clean
```

Build and open it:

```powershell
python .\scripts\build_docs.py --clean --open
```

The built local entry point is:

- `docs/_build/html/index.html`
