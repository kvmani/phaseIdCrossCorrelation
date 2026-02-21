from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from PIL import Image

from phase_id_xcorr.evaluation import run_curated_ncc


def _save_gray(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def test_run_curated_ncc_end_to_end(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    (packet / "experimental_patterns").mkdir(parents=True)
    (packet / "simulated_patterns" / "assume_fe_bcc").mkdir(parents=True)
    (packet / "simulated_patterns" / "assume_fe3o4_magnetite").mkdir(parents=True)
    (packet / "simulated_patterns" / "assume_feo_wustite").mkdir(parents=True)

    # Experimental image is closest to fe_bcc simulation
    exp = np.zeros((32, 32), dtype=np.uint8)
    exp[8:24, 10:22] = 220
    sim_good = exp.copy()
    sim_mid = np.roll(exp, shift=2, axis=1)
    sim_bad = 255 - exp

    _save_gray(packet / "experimental_patterns" / "fe_bcc_Ori_1.bmp", exp)
    _save_gray(packet / "simulated_patterns" / "assume_fe_bcc" / "fe_bcc_Ori_1.bmp", sim_good)
    _save_gray(packet / "simulated_patterns" / "assume_fe3o4_magnetite" / "fe_bcc_Ori_1.bmp", sim_mid)
    _save_gray(packet / "simulated_patterns" / "assume_feo_wustite" / "fe_bcc_Ori_1.bmp", sim_bad)

    (packet / "01_experimental_patterns_template.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "r001",
                        "image_file": "experimental_patterns/fe_bcc_Ori_1.bmp",
                        "true_phase": "fe_bcc",
                        "orientation_angles_degrees": {"phi1": 0.0, "PHI": 0.0, "phi2": 0.0},
                        "image_info": {"bit_depth": 8, "height": 32, "width": 32, "format": "bmp"},
                        "label_source": "test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    (packet / "02_simulated_patterns_template.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "r001",
                        "experimental_image": "experimental_patterns/fe_bcc_Ori_1.bmp",
                        "true_phase": "fe_bcc",
                        "simulated_candidates": [
                            {
                                "assumed_phase": "fe_bcc",
                                "simulated_image": "simulated_patterns/assume_fe_bcc/fe_bcc_Ori_1.bmp",
                                "candidate_angles_degrees": {"phi1": 1.0, "PHI": 2.0, "phi2": 3.0},
                                "indexing_status": "ok",
                                "is_fallback_orientation": False,
                            },
                            {
                                "assumed_phase": "fe3o4_magnetite",
                                "simulated_image": "simulated_patterns/assume_fe3o4_magnetite/fe_bcc_Ori_1.bmp",
                                "candidate_angles_degrees": {"phi1": 1.0, "PHI": 2.0, "phi2": 3.0},
                                "indexing_status": "ok",
                                "is_fallback_orientation": False,
                            },
                            {
                                "assumed_phase": "feo_wustite",
                                "simulated_image": "simulated_patterns/assume_feo_wustite/fe_bcc_Ori_1.bmp",
                                "candidate_angles_degrees": {"phi1": 1.0, "PHI": 2.0, "phi2": 3.0},
                                "indexing_status": "ok",
                                "is_fallback_orientation": False,
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    (packet / "04_processing_template.json").write_text(
        json.dumps(
            {
                "settings": {
                    "normalization_method": "minmax_inside_mask",
                }
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    result = run_curated_ncc(
        packet_dir=packet,
        out_dir=out_dir,
        repo_root=tmp_path,
        debug=True,
    )

    summary = result["summary"]
    assert summary["cases_total"] == 1
    assert summary["cases_correct"] == 1
    assert summary["top1_accuracy"] == 1.0

    assert (out_dir / "scores.csv").exists()
    assert (out_dir / "decisions.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "error_cases.md").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "cases" / "r001_panel.png").exists()
