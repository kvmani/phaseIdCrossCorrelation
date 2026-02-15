# data/test

This directory stores small deterministic fixtures for development and debug workflows.

Canonical dataset structure, naming, and manifest requirements are defined in:

- `docs/test_data_setup_plan.md`

Do not add ad-hoc files directly without updating manifests.

Manifest templates are pre-created under:

- `data/test/manifests/curated_cases.csv`
- `data/test/manifests/curated_pairs.csv`
- `data/test/manifests/scan_bundle_manifest.csv`
- `data/test/manifests/preprocessing_policy.yaml`

Preferred student-facing templates are JSON examples:

- `data/test/manifests/curated_cases.example.json`
- `data/test/manifests/curated_candidates.example.json`
- `data/test/manifests/scan_bundle.example.json`
- `data/test/manifests/preprocessing_policy.example.json`
