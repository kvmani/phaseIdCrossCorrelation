# Source Layout

Core package code lives under `src/phase_id_xcorr/`.

- `intake`: incoming packet validation
- `preprocessing`: image loading, masking, normalization
- `similarity`: NCC scoring primitives
- `features`: KikuchiPy Hough feature extraction
- `evaluation`: curated NCC and Hough comparison workflows
- `reporting`: run manifest helpers
- `ml`: `.oh5` ingestion, labels, quality gating, splits, dataset prep, training, suite reporting, and GUI exploration/diagnosis

See `docs/architecture.md` for the higher-level workflow map.
