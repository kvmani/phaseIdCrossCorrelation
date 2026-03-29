# ML Configs

YAML configs for the ML classifier branch.

Files:

- `dataset_prepare.default.yml`: build combined train/val/test dataset from `.oh5` + CSV label pairs (`input_mode: oh5_csv_labels`).
- `dataset_prepare.single_phase_scan_map.debug.yml`: build combined train/val/test dataset from `.oh5` files where each file maps to one phase (`input_mode: single_phase_scan_map`).
- `dataset_prepare.v3_al_ni_cu.example.yml`: concise v3 schema using `data_source_folder` + `listOfFiles`, expression quality filters, unified preprocessing, and capped leakage-safe splits.
- `train.convnextv2_nano.pretrained.debug.yml`: debug training run with ConvNeXtV2 pretrained init.
- `train.simple_cnn.debug.yml`: minimal fast smoke training run.
- `benchmark_suite.debug.yml`: multi-model debug benchmark suite.
- `benchmark_suite.classiication_training_data.smoke.yml`: smoke benchmark suite on local `Al/Ni/Cu` cropped `.oh5` data.
- `benchmark_suite.ni_cu_al.production.yml`: production benchmark suite template for the Windows `Ni-Cu-Al` scan folder.
- `full_cycle.debug.yml`: one-go orchestration config (dataset prep -> suite -> HTML + PPTX).
- `full_cycle.ni_cu_al.production.yml`: production full-cycle template using the Windows `Ni-Cu-Al` scan folder.
- `dataset_prepare.classiication_training_data.filtered.yml`: local `Al/Ni/Cu` cropped `.oh5` prep with `CI > 0.4 && Fit < 1.5`.
- `dataset_prepare.ni_cu_al.production.yml`: production dataset-prep template pointing at `F:\PhaseID_Training_Data\Ni-Cu-Al_Scans`.
- `dataset_prepare.data_march2026.balanced.yml`: March 2026 `Al/Cu/Ni-2_1.oh5` dataset prep with `CI > 0.5 && Fit < 1.0`, `0.8/0.1/0.1` split, and phase balancing down to the smallest accepted phase count.
- `phase_explorer.ni_cu_al.production.yml`: production explorer GUI config for `Al-1.oh5`, `Ni.oh`, and `Cu-1.oh5`.
- `train.classiication_training_data.smoke.yml`: smoke single-run training config on the filtered local dataset.
- `train.ni_cu_al.production.base.yml`: production baseline train config for the filtered Windows dataset.
- `train.data_march2026.balanced.base.yml`: balanced March 2026 training base config using the balanced dataset manifest.
- `benchmark_suite.data_march2026.balanced.yml`: benchmark suite config over the balanced March 2026 dataset.
- `full_cycle.data_march2026.balanced.yml`: one-go full-cycle config for the balanced March 2026 dataset.
- `oh5_sample_inference.ni_different_condition.example.yml`: example unseen-scan CNN inference config for filtered random sampling from new `.oh5` files under a Windows folder.
- `oh5_sample_inference.data_march2026.example.yml`: explicit three-scan inference config for `F:/PhaseID_Training_Data/Data_March2026/Data_1.oh5` to `Data_3.oh5`, sampling 5 filtered patterns per scan.
- `diagnostic_gallery.example.yml`: cross-condition gallery template with reference/unknown source groups, quality gating, prediction filters, and contact-sheet export.

Run commands:

```bash
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.default.yml --debug
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.single_phase_scan_map.debug.yml --debug
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.v3_al_ni_cu.example.yml --debug
python scripts/run_ml_dataset_prepare.py --config configs/ml/dataset_prepare.data_march2026.balanced.yml --debug
python scripts/run_ml_train_classifier.py --config configs/ml/train.convnextv2_nano.pretrained.debug.yml --debug
python scripts/run_ml_train_classifier.py --config configs/ml/train.data_march2026.balanced.base.yml --debug
python scripts/run_ml_benchmark_suite.py --config configs/ml/benchmark_suite.debug.yml --debug
python scripts/run_ml_benchmark_suite.py --config configs/ml/benchmark_suite.classiication_training_data.smoke.yml --debug
python scripts/run_ml_benchmark_suite.py --config configs/ml/benchmark_suite.ni_cu_al.production.yml --debug
python scripts/run_ml_benchmark_suite.py --config configs/ml/benchmark_suite.data_march2026.balanced.yml --debug
python scripts/run_ml_phase_explorer.py --config configs/ml/dataset_prepare.v3_al_ni_cu.example.yml --debug
python scripts/run_ml_phase_explorer.py --config configs/ml/phase_explorer.ni_cu_al.production.yml --debug
python scripts/run_ml_oh5_sample_inference.py --config configs/ml/oh5_sample_inference.ni_different_condition.example.yml --debug
python scripts/run_ml_oh5_sample_inference.py --config configs/ml/oh5_sample_inference.data_march2026.example.yml --debug
python scripts/run_ml_diagnostic_gallery.py --config configs/ml/diagnostic_gallery.example.yml --debug
python scripts/run_ml_full_cycle.py --config configs/ml/full_cycle.debug.yml --debug
python scripts/run_ml_full_cycle.py --config configs/ml/full_cycle.ni_cu_al.production.yml --debug
python scripts/run_ml_full_cycle.py --config configs/ml/full_cycle.data_march2026.balanced.yml --debug
```

Each workflow writes both `manifest.json` and `events.jsonl` under its output directory. The sampled `.oh5` inference workflow now also writes `sample_predictions.json` alongside the existing CSV summaries. Dataset HTML summaries also include phase-wise split composition, CI/Fit/IQ statistics, and modal intensity values. Benchmark HTML summaries include best-model selection, confusion matrices, per-class metrics, and links to per-run artifacts.

For the phase explorer, publication-export styling is set under `explorer.export`. Supported fields now include `dpi`, `figure_size_inches`, `font_family`, `font_size`, `title_font_size`, `label_font_size`, `tick_label_size`, `x_tick_label_size`, `y_tick_label_size`, `tick_width`, `tick_length`, `minor_tick_width`, `minor_tick_length`, `tick_direction`, `x_tick_rotation`, `y_tick_rotation`, `spine_line_width`, `grid_line_width`, `grid_alpha`, `title_pad`, `label_pad`, `figure_facecolor`, `axes_facecolor`, `show_minor_ticks`, and `savefig_pad_inches`. Plot-specific limits and labeling are controlled under `explorer.intensity_plot` and `explorer.attribute_plot`, including `x_min`, `x_max`, `y_min`, `y_max`, `field_ranges`, `field_y_ranges`, `title(_template)`, `x_label(_template)`, `y_label(_template)`, `color`, `edge_color`, and `bar_line_width`.
