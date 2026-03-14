from __future__ import annotations

from phase_id_xcorr.ml.quality import evaluate_quality, quality_policy_from_config


def test_quality_expression_with_aliases() -> None:
    policy = quality_policy_from_config(
        {
            "expression": "CI > 0.5 && Fit < 0.5",
        }
    )
    ok = evaluate_quality(
        {"confidence_index": 0.8, "image_quality": 12.0, "fit": 0.3, "valid": True},
        policy,
    )
    bad = evaluate_quality(
        {"confidence_index": 0.2, "image_quality": 12.0, "fit": 0.3, "valid": True},
        policy,
    )
    assert ok.accept is True
    assert bad.accept is False
    assert "quality_expression_false" in bad.reasons
