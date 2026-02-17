# G0 Data Intake Validation Report

- Packet directory: `student_data_packet_phaseid`
- Timestamp (UTC): `2026-02-17T17:03:40+00:00`
- Gate status: **HOLD**

## Summary

- Errors: **15**
- Warnings: **6**
- Total findings: **21**

## Checked Files

- `experimental`: `01_experimental_patterns_template.json`
- `simulated`: `02_simulated_patterns_template.json`
- `scan`: `03_scan_files_template.json`
- `processing`: `04_processing_template.json`

## Findings

| # | Severity | Code | Location | Message |
|---|---|---|---|---|
| 1 | error | missing_file | `experimental_json.records[0].image_file` | Referenced file does not exist: experimental_patterns/fe_bcc_Ori_1.png |
| 2 | warning | placeholder_text | `experimental_json.records[0].notes` | Placeholder text still present; replace with real notes. |
| 3 | error | missing_file | `experimental_json.records[1].image_file` | Referenced file does not exist: experimental_patterns/fe3o4_magnetite_Ori_1.png |
| 4 | warning | placeholder_text | `experimental_json.records[1].notes` | Placeholder text still present; replace with real notes. |
| 5 | error | missing_file | `experimental_json.records[2].image_file` | Referenced file does not exist: experimental_patterns/feo_wustite_Ori_1.png |
| 6 | warning | placeholder_text | `experimental_json.records[2].notes` | Placeholder text still present; replace with real notes. |
| 7 | error | missing_file | `simulated_json.records[0].experimental_image` | Referenced file does not exist: experimental_patterns/fe_bcc_Ori_1.png |
| 8 | error | missing_file | `simulated_json.records[0].simulated_candidates[0].simulated_image` | Referenced file does not exist: simulated_patterns/assume_fe_bcc/fe_bcc_Ori_1.png |
| 9 | error | missing_file | `simulated_json.records[0].simulated_candidates[1].simulated_image` | Referenced file does not exist: simulated_patterns/assume_fe3o4_magnetite/fe_bcc_Ori_1.png |
| 10 | error | missing_file | `simulated_json.records[0].simulated_candidates[2].simulated_image` | Referenced file does not exist: simulated_patterns/assume_feo_wustite/fe_bcc_Ori_1.png |
| 11 | warning | placeholder_text | `simulated_json.records[0].notes` | Placeholder text still present; replace with real notes. |
| 12 | error | missing_file | `simulated_json.records[1].experimental_image` | Referenced file does not exist: experimental_patterns/feo_wustite_Ori_1.png |
| 13 | error | missing_file | `simulated_json.records[1].simulated_candidates[0].simulated_image` | Referenced file does not exist: simulated_patterns/assume_fe_bcc/feo_wustite_Ori_1.png |
| 14 | error | missing_file | `simulated_json.records[1].simulated_candidates[1].simulated_image` | Referenced file does not exist: simulated_patterns/assume_fe3o4_magnetite/feo_wustite_Ori_1.png |
| 15 | error | missing_file | `simulated_json.records[1].simulated_candidates[2].simulated_image` | Referenced file does not exist: simulated_patterns/assume_feo_wustite/feo_wustite_Ori_1.png |
| 16 | warning | placeholder_text | `simulated_json.records[1].notes` | Placeholder text still present; replace with real notes. |
| 17 | error | missing_file | `scan_json.scan_records[0].scan_files.assume_fe_bcc` | Referenced file does not exist: scan_files/scan_s001__assume_fe_bcc.oh5 |
| 18 | error | missing_file | `scan_json.scan_records[0].scan_files.assume_fe3o4_magnetite` | Referenced file does not exist: scan_files/scan_s001__assume_fe3o4_magnetite.oh5 |
| 19 | error | missing_file | `scan_json.scan_records[0].scan_files.assume_feo_wustite` | Referenced file does not exist: scan_files/scan_s001__assume_feo_wustite.oh5 |
| 20 | warning | few_manual_points | `scan_json.scan_records[0].manual_check_points` | manual_check_points has 3 points; recommend at least 10 for baseline validation. |
| 21 | error | missing_simulated_record | `simulated_json.records` | Experimental record_id 'r002' missing in simulated records. |

## Gate Decision Guidance

Hold progression. Resolve all `error` findings, then re-run G0 validation.
