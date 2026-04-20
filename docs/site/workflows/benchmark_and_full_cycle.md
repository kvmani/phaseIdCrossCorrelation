# Benchmark Suite and Full-Cycle Runs

This page explains how single-run training, benchmark suites, and the full-cycle orchestrator fit together.

:::{figure} ../figures/full_cycle_flow.svg
:alt: Full-cycle flow schematic
:width: 100%

Full-cycle orchestration: dataset prep feeds the suite, the suite produces per-run reports and a suite HTML, and the full-cycle layer links them into one browsable summary.
:::

## Relationship between configs

The ML experiment stack has three layers:

1. **dataset prep config**
2. **base train config**
3. **benchmark suite config**

The full-cycle workflow resolves these into a deterministic run where:

- dataset prep runs first,
- the generated dataset manifest is injected into the resolved base train config,
- the suite runs from that resolved base,
- optional presentation output is generated afterward.

## Train one model

```powershell
python .\scripts\run_ml_train_classifier.py --config .\configs\ml\train.data_march2026.balanced.debug.base.yml --debug
```

Use this when you want to inspect one model family in isolation before running the whole suite.

## Run a debug benchmark suite

```powershell
python .\scripts\run_ml_benchmark_suite.py --config .\configs\ml\benchmark_suite.data_march2026.balanced.debug.yml --debug
```

Use this when you want:

- quick end-to-end validation,
- report-link sanity checks,
- a first read of model ordering on a small balanced sample.

## Run the production benchmark suite

```powershell
python .\scripts\run_ml_benchmark_suite.py --config .\configs\ml\benchmark_suite.data_march2026.balanced.yml --debug
```

April 2026 Cu/Ni-only balanced production suite:

```powershell
python .\scripts\run_ml_benchmark_suite.py --config .\configs\ml\benchmark_suite.april2026_cu_ni_balanced.yml --debug
```

## Run the debug full cycle

```powershell
python .\scripts\run_ml_full_cycle.py --config .\configs\ml\full_cycle.data_march2026.balanced.debug.yml --debug
```

## Run the production full cycle

```powershell
python .\scripts\run_ml_full_cycle.py --config .\configs\ml\full_cycle.data_march2026.balanced.yml --debug
```

April 2026 Cu/Ni-only balanced production full cycle:

```powershell
python .\scripts\run_ml_full_cycle.py --config .\configs\ml\full_cycle.april2026_cu_ni_balanced.yml --debug
```

The April 2026 production full-cycle YAML exposes `benchmark_batch_size` as the single batch-size control for the entire benchmark suite.

April 2026 Cu/Ni smoke full-cycle runs for batch-size validation:

```powershell
python .\scripts\run_ml_full_cycle.py --config .\configs\ml\full_cycle.april2026_cu_ni_balanced.smoke.batch64.yml --debug
python .\scripts\run_ml_full_cycle.py --config .\configs\ml\full_cycle.april2026_cu_ni_balanced.smoke.batch32.yml --debug
```

## Outputs to read

### Benchmark suite

- `suite_summary.json`
- `suite_summary.md`
- `suite_report.html`
- one subfolder per experiment with `report.json`, manifests, checkpoints, and history

### Full cycle

- `full_cycle_summary.json`
- `full_cycle_summary.html`
- resolved dataset/suite/train configs
- optional PPTX

## Why full-cycle exists

The full-cycle runner exists to reduce user-side coordination errors. It ensures that:

- the suite is always tied to the exact dataset prep output it used,
- resolved configs are preserved,
- the high-level HTML page points to the dataset and suite drill-down surfaces,
- the experiment remains reviewable after the terminal session is gone.

## Source docs

Legacy source files:

- `docs/ml_training_inference_workflow.md`
- `configs/ml/README.md`
