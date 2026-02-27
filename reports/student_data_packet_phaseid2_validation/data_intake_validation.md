# G0 Data Intake Validation Report

- Packet directory: `data/test/student_data_packet_phaseid2`
- Timestamp (UTC): `2026-02-26T17:36:53+00:00`
- Gate status: **HOLD**

## Summary

- Errors: **18**
- Warnings: **6**
- Total findings: **24**

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
| 4 | error | missing_file | `simulated_json.records[0].experimental_image` | Referenced file does not exist: experimental_patterns/fe_bcc_Ori_1.png |
| 5 | error | cross_file_mismatch | `simulated_json.records[0]` | experimental_image does not match image_file for same record_id. |
| 6 | error | missing_file | `simulated_json.records[0].simulated_candidates[0].simulated_image` | Referenced file does not exist: simulated_patterns/assume_fe_bcc/fe_bcc_Ori_1.png |
| 7 | error | missing_file | `simulated_json.records[0].simulated_candidates[1].simulated_image` | Referenced file does not exist: simulated_patterns/assume_fe3o4_magnetite/fe_bcc_Ori_1.png |
| 8 | error | missing_file | `simulated_json.records[0].simulated_candidates[2].simulated_image` | Referenced file does not exist: simulated_patterns/assume_feo_wustite/fe_bcc_Ori_1.png |
| 9 | warning | placeholder_text | `simulated_json.records[0].notes` | Placeholder text still present; replace with real notes. |
| 10 | error | missing_file | `simulated_json.records[1].experimental_image` | Referenced file does not exist: experimental_patterns/fe3o4_magnetite_Ori_1.png |
| 11 | error | cross_file_mismatch | `simulated_json.records[1]` | experimental_image does not match image_file for same record_id. |
| 12 | error | missing_file | `simulated_json.records[1].simulated_candidates[0].simulated_image` | Referenced file does not exist: simulated_patterns/assume_fe_bcc/fe3o4_magnetite_Ori_1.png |
| 13 | error | missing_file | `simulated_json.records[1].simulated_candidates[1].simulated_image` | Referenced file does not exist: simulated_patterns/assume_fe3o4_magnetite/fe3o4_magnetite_Ori_1.png |
| 14 | error | missing_file | `simulated_json.records[1].simulated_candidates[2].simulated_image` | Referenced file does not exist: simulated_patterns/assume_feo_wustite/fe3o4_magnetite_Ori_1.png |
| 15 | warning | placeholder_text | `simulated_json.records[1].notes` | Placeholder text still present; replace with real notes. |
| 16 | error | missing_file | `simulated_json.records[2].experimental_image` | Referenced file does not exist: experimental_patterns/feo_wustite_Ori_1.png |
| 17 | error | cross_file_mismatch | `simulated_json.records[2]` | experimental_image does not match image_file for same record_id. |
| 18 | error | missing_file | `simulated_json.records[2].simulated_candidates[0].simulated_image` | Referenced file does not exist: simulated_patterns/assume_fe_bcc/feo_wustite_Ori_1.png |
| 19 | error | missing_file | `simulated_json.records[2].simulated_candidates[1].simulated_image` | Referenced file does not exist: simulated_patterns/assume_fe3o4_magnetite/feo_wustite_Ori_1.png |
| 20 | error | missing_file | `simulated_json.records[2].simulated_candidates[2].simulated_image` | Referenced file does not exist: simulated_patterns/assume_feo_wustite/feo_wustite_Ori_1.png |
| 21 | warning | placeholder_text | `simulated_json.records[2].notes` | Placeholder text still present; replace with real notes. |
| 22 | error | oh5_read_error | `scan_json.scan_records[0].scan_files.assume_fe3o4_magnetite` | Failed reading assume_fe3o4_magnetite: "Unable to synchronously open object (object 'Pattern' doesn't exist)" |
| 23 | error | oh5_read_error | `scan_json.scan_records[0].scan_files.assume_fe_bcc` | Failed reading assume_fe_bcc: "Unable to synchronously open object (object 'Pattern' doesn't exist)" |
| 24 | error | oh5_read_error | `scan_json.scan_records[0].scan_files.assume_feo_wustite` | Failed reading assume_feo_wustite: "Unable to synchronously open object (object 'Pattern' doesn't exist)" |

## Gate Decision Guidance

Hold progression. Resolve all `error` findings, then re-run G0 validation.
