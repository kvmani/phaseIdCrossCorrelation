# G0 Data Intake Validation

This document explains how to run the G0 gate validator on a student data packet.

## Purpose

Before any NCC implementation work, verify that incoming data is complete, internally consistent, and ready for pipeline development.

## CLI

Run:

```bash
python3 scripts/run_g0_data_intake_validation.py \
  --packet-dir student_data_packet_phaseid \
  --out-dir reports
```

Optional flags:

- `--debug`: verbose logs.
- `--strict`: exits non-zero when gate status is not `GO`.

## Checks Performed

1. JSON syntax and required keys for all four packet JSON files.
2. File existence for all referenced experimental/simulated images and `.oh5` scan files.
3. Allowed phase label enforcement:
   - `fe_bcc`
   - `fe3o4_magnetite`
   - `feo_wustite`
4. Simulated-candidate structure checks:
   - exactly 3 candidates per record,
   - one candidate per assumed phase,
   - fallback rules for failed indexing.
5. `.oh5` triad consistency checks:
   - shared grid/pattern shape across three phase-isolated scan files,
   - declared grid info matches observed scan info when files are present.

Notes:

- Manual check points can be fewer than 10 in early-stage development.
- Image paths can use any supported format as long as files exist and are readable by the downstream loader.

## Outputs

- Markdown report: `reports/data_intake_validation.md`
- Machine-readable manifest: `reports/data_intake_manifest.json`

## Gate Semantics

- `GO`: no validation errors.
- `HOLD`: one or more validation errors; resolve and re-run.

Warnings do not block by themselves, but should be addressed where possible.
