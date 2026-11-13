"""Input contract and geometry-safe helpers for the MONAI BraTS bundle."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from nibabel.orientations import aff2axcodes


CHANNEL_ORDER = ("t1c", "t1", "t2", "flair")
MODEL_NAME = "MONAI brats_mri_segmentation"
MODEL_VERSION = "0.5.4"
MODEL_LICENSE = "Apache-2.0"
MODEL_WEIGHT_SHA256 = "860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae"
EXPECTED_SPACING_MM = np.array([1.0, 1.0, 1.0], dtype=np.float64)
AFFINE_TOLERANCE_MM = 1e-3
SPACING_TOLERANCE_MM = 0.02
# BraTS 2018/2021 use label 4 for enhancing tumor; BraTS 2023 relabeled it 3.
# BraTS 2024 post-treatment GLI uses 1=NETC, 2=SNFH, 3=ET, 4=resection cavity.
SUPPORTED_LABELS = (0, 1, 2, 3, 4)


class InputProtocolError(ValueError):
    """A safe, user-facing input validation failure."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class AlignedStudy:
    image: np.ndarray  # [channel, i, j, k], float32
    affine: np.ndarray
    spacing_mm: np.ndarray
    shape: tuple[int, int, int]


def voxel_spacing(affine: np.ndarray) -> np.ndarray:
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise InputProtocolError("INVALID_GEOMETRY")
    spacing = np.linalg.norm(matrix[:3, :3], axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise InputProtocolError("INVALID_GEOMETRY")
    return spacing


def validate_study(volumes: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> AlignedStudy:
    """Validate bundle-preprocessed NIfTI volumes without silently registering them."""
    if set(volumes) != set(CHANNEL_ORDER):
        raise InputProtocolError("MISSING_SEQUENCE")

    arrays: list[np.ndarray] = []
    reference_shape: tuple[int, int, int] | None = None
    reference_affine: np.ndarray | None = None
    reference_spacing: np.ndarray | None = None
    for channel in CHANNEL_ORDER:
        data, affine = volumes[channel]
        array = np.asarray(data)
        if array.ndim != 3 or any(dimension < 8 for dimension in array.shape):
            raise InputProtocolError("INVALID_DIMENSIONS")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise InputProtocolError("INVALID_PIXELS")
        matrix = np.asarray(affine, dtype=np.float64)
        spacing = voxel_spacing(matrix)
        if not np.allclose(spacing, EXPECTED_SPACING_MM, atol=SPACING_TOLERANCE_MM, rtol=0):
            raise InputProtocolError("UNSUPPORTED_SPACING")
        if aff2axcodes(matrix) != ("R", "A", "S"):
            raise InputProtocolError("UNSUPPORTED_ORIENTATION")
        if reference_shape is None:
            reference_shape, reference_affine, reference_spacing = array.shape, matrix, spacing
        elif array.shape != reference_shape:
            raise InputProtocolError("MISMATCHED_GEOMETRY")
        elif not np.allclose(matrix, reference_affine, atol=AFFINE_TOLERANCE_MM, rtol=0):
            raise InputProtocolError("MISMATCHED_GEOMETRY")
        elif not np.allclose(spacing, reference_spacing, atol=SPACING_TOLERANCE_MM, rtol=0):
            raise InputProtocolError("MISMATCHED_GEOMETRY")
        arrays.append(array.astype(np.float32, copy=False))

    assert reference_shape is not None and reference_affine is not None and reference_spacing is not None
    image = np.stack(arrays, axis=0)
    if not np.any(image):
        raise InputProtocolError("EMPTY_IMAGE")
    return AlignedStudy(image, reference_affine, reference_spacing, reference_shape)


def region_masks_to_labels(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Map [TC, WT, ET] logits/probabilities to the bundle's 0/1/2/4 label map."""
    values = np.asarray(probabilities)
    if values.ndim < 4 or values.shape[0] != 3 or not np.isfinite(values).all():
        raise InputProtocolError("INVALID_PREDICTION")
    tc, wt, et = values > threshold
    labels = np.zeros(values.shape[1:], dtype=np.uint8)
    labels[wt] = 2
    labels[tc] = 1
    labels[et] = 4
    return labels


def segmentation_volumes_mm3(labels: np.ndarray, spacing_mm: np.ndarray) -> dict[str, float]:
    """Calculate nested BraTS region volumes using physical voxel spacing."""
    mask = np.asarray(labels)
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    if mask.ndim != 3 or spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise InputProtocolError("INVALID_GEOMETRY")
    voxel_volume = float(np.prod(spacing))
    whole = np.isin(mask, (1, 2, 4))
    core = np.isin(mask, (1, 4))
    enhancing = mask == 4
    return {
        "whole_tumor_mm3": float(np.count_nonzero(whole) * voxel_volume),
        "tumor_core_mm3": float(np.count_nonzero(core) * voxel_volume),
        "enhancing_tumor_mm3": float(np.count_nonzero(enhancing) * voxel_volume),
    }


def segmentation_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    spacing_mm: np.ndarray,
    reference_convention: str = "brats_2018_2021",
) -> dict[str, dict[str, float | None]]:
    """Compute WT/TC/ET metrics using an explicitly selected reference label convention."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    pred = np.asarray(prediction)
    truth = np.asarray(reference)
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    label_regions = {
        "brats_2018_2021": ((1, 2, 4), (1, 4), (4,), (0, 1, 2, 4)),
        "brats_2023": ((1, 2, 3), (1, 3), (3,), (0, 1, 2, 3)),
        "brats_2024_gli_post_treatment": ((1, 2, 3), (1, 3), (3,), (0, 1, 2, 3, 4)),
        "msd_task01": ((1, 2, 3), (2, 3), (3,), (0, 1, 2, 3)),
    }
    if reference_convention not in label_regions:
        raise InputProtocolError("INVALID_REFERENCE")
    if (
        pred.ndim != 3
        or truth.shape != pred.shape
        or spacing.shape != (3,)
        or not np.isfinite(spacing).all()
        or np.any(spacing <= 0)
        or not np.isin(pred, (0, 1, 2, 4)).all()
        or not np.isin(truth, label_regions[reference_convention][3]).all()
    ):
        raise InputProtocolError("INVALID_REFERENCE")
    ref_wt, ref_tc, ref_et, _ = label_regions[reference_convention]
    regions = {
        "whole_tumor": (np.isin(pred, (1, 2, 4)), np.isin(truth, ref_wt)),
        "tumor_core": (np.isin(pred, (1, 4)), np.isin(truth, ref_tc)),
        "enhancing_tumor": (pred == 4, np.isin(truth, ref_et)),
    }
    result = {}
    for name, (a, b) in regions.items():
        intersection = int(np.count_nonzero(a & b))
        count_a, count_b = int(np.count_nonzero(a)), int(np.count_nonzero(b))
        denominator = count_a + count_b
        if denominator == 0:
            dice, iou, hd95 = 1.0, 1.0, 0.0
        else:
            dice = 2.0 * intersection / denominator
            union = int(np.count_nonzero(a | b))
            iou = intersection / union if union else 1.0
            if count_a == 0 or count_b == 0:
                hd95 = None
            else:
                surface_a = a ^ binary_erosion(a)
                surface_b = b ^ binary_erosion(b)
                distance_to_b = distance_transform_edt(~surface_b, sampling=spacing)
                distance_to_a = distance_transform_edt(~surface_a, sampling=spacing)
                distances = np.concatenate((distance_to_b[surface_a], distance_to_a[surface_b]))
                hd95 = float(np.percentile(distances, 95)) if distances.size else 0.0
        result[name] = {"dice": float(dice), "iou": float(iou), "hausdorff95_mm": hd95}
    return result
