# Read Me First: EBSD Data Pack (Student Version)

Thank you for helping prepare data for our EBSD phase-identification study.

## What this data is for

We are testing whether cross-correlation can correctly identify which phase (Fe BCC, magnetite, or wustite) best matches each experimental Kikuchi pattern.

## What you need to provide

1. Experimental patterns (manually trusted phase labels).
2. Simulated patterns (3 per experimental pattern, one from each assumed phase solution).
3. Three `.oh5` scan files for the same scan (each indexed assuming only one phase).
4. Filled JSON forms in this packet.

## Naming to use

Use this file naming style for experimental patterns:

- `{phase}_Ori_{id}.{ext}`
- Examples: `fe_bcc_Ori_1.png`, `fe3o4_magnetite_Ori_2.tif`, `feo_wustite_Ori_1.bmp`

For simulated patterns, use the same base filename and place into the correct folder:

- `simulated_patterns/assume_fe_bcc/`
- `simulated_patterns/assume_fe3o4_magnetite/`
- `simulated_patterns/assume_feo_wustite/`

Example:

- Experimental: `experimental_patterns/fe3o4_magnetite_Ori_1.png`
- Simulated variants:
  - `simulated_patterns/assume_fe_bcc/fe3o4_magnetite_Ori_1.png`
  - `simulated_patterns/assume_fe3o4_magnetite/fe3o4_magnetite_Ori_1.png`
  - `simulated_patterns/assume_feo_wustite/fe3o4_magnetite_Ori_1.png`

## Fill these 4 JSON files

- `01_experimental_patterns_template.json`
- `02_simulated_patterns_template.json`
- `03_scan_files_template.json`
- `04_processing_template.json`

Please replace example values with real values.

## Important notes

- Angles are in degrees.
- Use phase names exactly as:
  - `fe_bcc`
  - `fe3o4_magnetite`
  - `feo_wustite`
- If indexing failed for a simulated pattern, keep angle as `0,0,0` only if you mark:
  - `indexing_status = "failed"`
  - `is_fallback_orientation = true`

## How to return

1. Keep this folder structure unchanged.
2. Add your data files and fill all JSON files.
3. Zip the whole folder.
4. Send the zip back.
