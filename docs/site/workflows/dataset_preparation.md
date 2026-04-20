# Dataset Preparation

Dataset preparation is the hinge between raw `.oh5` scans and every ML workflow downstream.

```{mermaid}
flowchart LR
    A[Configured .oh5 sources] --> B[Field discovery and label resolution]
    B --> C[Quality gating]
    C --> D[Accepted qualified records]
    D --> E[Optional phase balancing]
    E --> F[Deterministic split assignment]
    F --> G[NPZ splits and records.csv]
    D --> H[Euler export CSV/JSON]
    F --> I[Selected orientation CSV/JSON]
    I --> J[IPF plots by split and phase]
    G --> K[manifest.json and summary.html]
```

:::{figure} ../figures/dataset_lineage.svg
:alt: Dataset lineage diagram
:width: 100%

Dataset lineage from source `.oh5` scans to balanced splits, orientation exports, IPF diagnostics, and manifest-linked reports.
:::

## Main inputs

Dataset preparation supports:

1. `oh5_csv_labels`
2. `single_phase_scan_map`
3. the concise v3 `listOfFiles` schema

The most relevant production templates currently are:

- `configs/ml/dataset_prepare.data_march2026.balanced.debug.yml`
- `configs/ml/dataset_prepare.data_march2026.balanced.yml`
- `configs/ml/dataset_prepare.april2026_cu_ni_balanced.yml`
- `configs/ml/dataset_prepare.april2026_cu_ni_balanced.smoke.yml`

## Main commands

### Debug balanced inspection run

```powershell
python .\scripts\run_ml_dataset_prepare.py --config .\configs\ml\dataset_prepare.data_march2026.balanced.debug.yml --debug
```

This uses exact per-phase split caps for rapid inspection and quick IPF review.

### Production balanced run

```powershell
python .\scripts\run_ml_dataset_prepare.py --config .\configs\ml\dataset_prepare.data_march2026.balanced.yml --debug
```

This uses the full qualified balanced pool with the configured `80/10/10` split policy.

### April 2026 Cu/Ni balanced run

```powershell
python .\scripts\run_ml_dataset_prepare.py --config .\configs\ml\dataset_prepare.april2026_cu_ni_balanced.yml --debug
```

This configuration:

- includes `Cu-1..15` except held-out `Cu-6.oh5`
- includes `Ni-1..10` except held-out `Ni-6.oh5`
- applies quality filtering first
- then equalizes Cu and Ni by qualified pattern count before split assignment

### April 2026 Cu/Ni smoke run

```powershell
python .\scripts\run_ml_dataset_prepare.py --config .\configs\ml\dataset_prepare.april2026_cu_ni_balanced.smoke.yml --debug
```

The smoke config uses the same source policy and balancing rule but caps the final split sizes per phase for fast benchmark validation.

## What the workflow writes

- `manifest.json`
- `records.csv`
- `splits/train.npz`
- `splits/val.npz`
- `splits/test.npz`
- `events.jsonl`
- `summary.html`
- `orientation_exports/qualified_orientations.csv`
- `orientation_exports/qualified_orientations.json`
- `orientation_exports/selected_orientations.csv`
- `orientation_exports/selected_orientations.json`
- `orientation_exports/ipf_index.json`
- `orientation_exports/ipf/...png`

## What to inspect first

1. `summary.html`
2. `manifest.json`
3. `orientation_exports/ipf_index.json`
4. the per-phase IPF PNGs

These outputs tell you:

- how many raw pixels were considered,
- how many passed quality filters,
- how many survived balancing,
- how the split counts landed per phase,
- whether orientation coverage looks acceptable.

## Why the workflow is designed this way

### Quality before balancing

Balancing before quality filtering would artificially equalize counts using patterns that should never be trusted downstream.

### Balancing before split assignment

Balancing first ensures train/val/test all sample from the same post-balance phase pool rather than letting majority-phase structure leak into some splits more strongly than others.

### Orientation export at both qualified and selected stages

The repository exports:

- **qualified** orientations: post-filter, pre-balance
- **selected** orientations: final dataset actually used downstream

This is deliberate. It lets you see whether balancing itself introduced orientation distortions.

## Related source material

Legacy source files:

- `docs/ml_input_data_runbook.md`
- `docs/ml_classifier_workflow.md`
