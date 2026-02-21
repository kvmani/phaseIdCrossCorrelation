# Curated NCC Critical Analysis (Current Dataset)

## Scope
- Packet: `data/test/student_data_packet_phaseid`
- Results: `reports/curated_ncc`
- Cases evaluated: 3 curated experimental patterns (1 per phase)

## Key Outcome
- Top-1 accuracy: 33.33% (1/3)
- All predictions collapsed to `fe_bcc`.
- Both misclassified cases selected a candidate with `indexing_status=failed` and `is_fallback_orientation=true`.

## Why NCC is Failing Here
1. **Fallback winners are dominating decisions**
   - `r002` and `r003` are both won by failed/fallback candidates.
   - Current decision rule uses pure max-NCC and does not penalize failed indexing status.

2. **Low phase separability in candidate simulations**
   - Sim-vs-sim NCC is high across phases (roughly ~0.45 to ~0.71).
   - Magnetite vs wustite simulated candidates are especially close (~0.67 to ~0.71), so NCC ranking margin is weak/fragile.

3. **One error is genuinely ambiguous under current preprocessing**
   - `r002` margin is only ~0.0387 (top and second are nearly tied).
   - This indicates insufficient discriminative signal after current mask + min-max normalization.

4. **Current preprocessing is minimal for EBSD band discrimination**
   - Pipeline currently uses only circular mask + min-max normalization.
   - No background flattening, band-pass filtering, or gradient-domain scoring yet.

5. **Dataset is too small to set robust confidence rules**
   - Only 1 curated orientation per phase is currently present.
   - Not enough coverage to tune thresholds and stress-test ranking behavior.

## High-Impact Next Focus
1. Add decision policy guardrails:
   - Penalize or reject `indexing_status=failed` candidates unless margin over best `ok` candidate exceeds a strict threshold.
   - Add "uncertain" output when margin is below threshold.

2. Improve preprocessing for structural band signal:
   - Add optional background subtraction/high-pass path before NCC.
   - Compare raw-intensity NCC vs gradient-NCC and report both.

3. Add geometric tolerance:
   - Evaluate local shift search (small translation window) and take best NCC.

4. Expand curated set:
   - At least 2 orientations per phase (minimum planned set), ideally more diverse cases.

5. Keep diagnostics-first reporting:
   - Continue per-case artifact with candidate images, status, NCC table, margins, and winner rationale.

## Primary Artifact for Inspection
- `reports/curated_ncc/inspection_report.html`
