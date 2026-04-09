"""ML dataset preparation, training, inference, and diagnostic workflows."""

from .dataset_builder import PrepareDatasetResult, prepare_ml_dataset
from .diagnostic_gallery import (
    DiagnosticGallerySession,
    add_manual_record,
    build_diagnostic_gallery_session,
    build_diagnostic_gallery_session_from_config,
    export_diagnostic_gallery_artifacts,
)
from .full_cycle import FullCycleResult, run_full_cycle
from .oh5_inference import FullScanSuiteInferenceResult, Oh5InferenceResult, run_oh5_sample_inference, run_suite_full_scan_inference
from .phase_explorer import ExplorerDataset, load_explorer_dataset
from .suite import SuiteResult, run_benchmark_suite
from .training import TrainResult, train_classifier

__all__ = [
    "PrepareDatasetResult",
    "prepare_ml_dataset",
    "DiagnosticGallerySession",
    "add_manual_record",
    "build_diagnostic_gallery_session",
    "build_diagnostic_gallery_session_from_config",
    "export_diagnostic_gallery_artifacts",
    "FullCycleResult",
    "run_full_cycle",
    "TrainResult",
    "train_classifier",
    "SuiteResult",
    "run_benchmark_suite",
    "ExplorerDataset",
    "load_explorer_dataset",
    "Oh5InferenceResult",
    "run_oh5_sample_inference",
    "FullScanSuiteInferenceResult",
    "run_suite_full_scan_inference",
]
