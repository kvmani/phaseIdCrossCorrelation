# G0 Data Intake Validation Report

- Packet directory: `data/test/student_data_packet_phaseid2`
- Timestamp (UTC): `2026-02-26T17:52:50+00:00`
- Gate status: **HOLD**

## Summary

- Errors: **3**
- Warnings: **6**
- Total findings: **9**

## Checked Files

- `experimental`: `01_experimental_patterns_template.json`
- `simulated`: `02_simulated_patterns_template.json`
- `scan`: `03_scan_files_template.json`
- `processing`: `04_processing_template.json`

## Findings

| # | Severity | Code | Location | Message |
|---|---|---|---|---|
| 1 | warning | placeholder_text | `experimental_json.records[0].notes` | Placeholder text still present; replace with real notes. |
| 2 | warning | placeholder_text | `experimental_json.records[1].notes` | Placeholder text still present; replace with real notes. |
| 3 | warning | placeholder_text | `experimental_json.records[2].notes` | Placeholder text still present; replace with real notes. |
| 4 | warning | placeholder_text | `simulated_json.records[0].notes` | Placeholder text still present; replace with real notes. |
| 5 | warning | placeholder_text | `simulated_json.records[1].notes` | Placeholder text still present; replace with real notes. |
| 6 | warning | placeholder_text | `simulated_json.records[2].notes` | Placeholder text still present; replace with real notes. |
| 7 | error | oh5_read_error | `scan_json.scan_records[0].scan_files.assume_fe3o4_magnetite` | Failed reading assume_fe3o4_magnetite: "Unable to synchronously open object (object 'Pattern' doesn't exist)" |
| 8 | error | oh5_read_error | `scan_json.scan_records[0].scan_files.assume_feo_wustite` | Failed reading assume_feo_wustite: "Unable to synchronously open object (object 'Pattern' doesn't exist)" |
| 9 | error | oh5_read_error | `scan_json.scan_records[0].scan_files.assume_fe_bcc` | Failed reading assume_fe_bcc: "Unable to synchronously open object (object 'Pattern' doesn't exist)" |

## Gate Decision Guidance

Hold progression. Resolve all `error` findings, then re-run G0 validation.
