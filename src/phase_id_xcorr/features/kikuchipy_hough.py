"""KikuchiPy-backed Hough (Radon) feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import kikuchipy as kp
import numpy as np
from orix.crystal_map import PhaseList


@dataclass(slots=True)
class HoughTransformConfig:
    """Configuration for KikuchiPy/PyEBSDIndex Hough transform extraction."""

    n_theta: int = 180
    n_rho: int = 90
    n_bands: int = 9
    sample_tilt_deg: float = 70.0
    detector_tilt_deg: float = 0.0
    pc: tuple[float, float, float] = (0.5, 0.5, 0.5)
    use_convolved_map: bool = True
    # Used only to build an indexer plan; no phase decision logic uses this stub phase.
    plan_phase_name: str = "bcc_hough_plan"
    plan_phase_space_group: int = 229


@dataclass(slots=True)
class HoughTransformResult:
    """Continuous Hough map and metadata."""

    hough_map: np.ndarray
    radon_raw_map: np.ndarray
    radon_conv_map: np.ndarray
    pad_rho: int
    pad_theta: int
    n_rho: int
    n_theta: int
    source_shape: tuple[int, int]
    use_convolved_map: bool
    raw_min: float
    raw_max: float
    conv_min: float
    conv_max: float


def _normalize_minmax(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    lo = float(np.min(arr)) if arr.size else 0.0
    hi = float(np.max(arr)) if arr.size else 0.0
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def binarize_hough_map(hough_map: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize normalized Hough map using absolute threshold in [0, 1]."""

    t = float(threshold)
    if t < 0.0 or t > 1.0:
        raise ValueError(f"Binary threshold must be in [0, 1], got {threshold}")
    out = np.zeros_like(hough_map, dtype=np.float32)
    out[np.asarray(hough_map) >= t] = 1.0
    return out


class KikuchiPyHoughExtractor:
    """Extract Hough maps via KikuchiPy's PyEBSDIndex-backed transform path."""

    def __init__(
        self,
        image_shape: tuple[int, int],
        config: HoughTransformConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or HoughTransformConfig()
        self.log = logger or logging.getLogger(__name__)
        self.image_shape = tuple(int(v) for v in image_shape)
        if len(self.image_shape) != 2:
            raise ValueError(f"image_shape must be 2D (rows, cols), got {image_shape}")

        phase_list = PhaseList(
            names=[self.config.plan_phase_name],
            space_groups=[int(self.config.plan_phase_space_group)],
        )
        det = kp.detectors.EBSDDetector(
            shape=self.image_shape,
            pc=self.config.pc,
            sample_tilt=float(self.config.sample_tilt_deg),
            tilt=float(self.config.detector_tilt_deg),
        )
        indexer = det.get_indexer(
            phase_list,
            nTheta=int(self.config.n_theta),
            nRho=int(self.config.n_rho),
            nBands=int(self.config.n_bands),
        )
        self.band_detect = indexer.bandDetectPlan
        self.pad_rho = int(self.band_detect.padding[0])
        self.pad_theta = int(self.band_detect.padding[1])

        self.log.debug(
            (
                "Initialized KikuchiPyHoughExtractor | shape=%s nTheta=%d nRho=%d nBands=%d "
                "padding=(%d,%d) use_convolved=%s"
            ),
            self.image_shape,
            int(self.band_detect.nTheta),
            int(self.band_detect.nRho),
            int(self.band_detect.nBands),
            self.pad_rho,
            self.pad_theta,
            bool(self.config.use_convolved_map),
        )

    def _crop_padding(self, array2d: np.ndarray) -> np.ndarray:
        arr = np.asarray(array2d, dtype=np.float32)
        r0 = self.pad_rho
        t0 = self.pad_theta
        if r0 <= 0 and t0 <= 0:
            return arr
        r1 = arr.shape[0] - r0
        t1 = arr.shape[1] - t0
        if r1 <= r0 or t1 <= t0:
            return arr
        return arr[r0:r1, t0:t1]

    def transform(self, image_float01: np.ndarray) -> HoughTransformResult:
        """Compute normalized Hough map from a 2D float image in [0, 1]."""

        if image_float01.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {image_float01.shape}")
        if tuple(int(v) for v in image_float01.shape) != self.image_shape:
            raise ValueError(
                f"Image shape {image_float01.shape} does not match extractor shape {self.image_shape}"
            )

        pat = np.ascontiguousarray(image_float01[None, :, :].astype(np.float32, copy=False))

        radon = self.band_detect.radonPlan.radon_faster(
            pat,
            padding=self.band_detect.padding,
            fixArtifacts=False,
            background=self.band_detect.backgroundsub,
        )
        radon_raw = np.asarray(radon[:, :, 0], dtype=np.float32)
        radon_conv, _ = self.band_detect.rdn_conv(radon.copy())
        radon_conv = np.asarray(radon_conv[:, :, 0], dtype=np.float32)

        radon_raw_crop = self._crop_padding(radon_raw)
        radon_conv_crop = self._crop_padding(radon_conv)

        raw_norm = _normalize_minmax(radon_raw_crop)
        conv_norm = _normalize_minmax(radon_conv_crop)
        chosen = conv_norm if self.config.use_convolved_map else raw_norm

        return HoughTransformResult(
            hough_map=chosen,
            radon_raw_map=raw_norm,
            radon_conv_map=conv_norm,
            pad_rho=self.pad_rho,
            pad_theta=self.pad_theta,
            n_rho=int(self.band_detect.nRho),
            n_theta=int(self.band_detect.nTheta),
            source_shape=self.image_shape,
            use_convolved_map=bool(self.config.use_convolved_map),
            raw_min=float(np.min(radon_raw_crop)) if radon_raw_crop.size else 0.0,
            raw_max=float(np.max(radon_raw_crop)) if radon_raw_crop.size else 0.0,
            conv_min=float(np.min(radon_conv_crop)) if radon_conv_crop.size else 0.0,
            conv_max=float(np.max(radon_conv_crop)) if radon_conv_crop.size else 0.0,
        )

