"""MONAI implementation for the pinned BraTS SegResNet research bundle."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from .protocol import (
    MODEL_LICENSE,
    MODEL_NAME,
    MODEL_VERSION,
    MODEL_WEIGHT_SHA256,
    AlignedStudy,
    InputProtocolError,
    region_masks_to_labels,
    segmentation_volumes_mm3,
)


class ModelUnavailable(RuntimeError):
    pass


class BrainTumorSegmenter:
    def __init__(self, weights_path: Path, *, device: str | None = None):
        self.weights_path = Path(weights_path).expanduser()
        self.device_name = device
        self.network = None
        self.inferer = None
        self.normalizer = None
        self._torch = None

    def load(self) -> None:
        if not self.weights_path.is_file():
            raise ModelUnavailable("MODEL_WEIGHTS_MISSING")
        digest = hashlib.sha256()
        try:
            with self.weights_path.open("rb") as weights:
                for block in iter(lambda: weights.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise ModelUnavailable("MODEL_WEIGHTS_UNAVAILABLE") from exc
        if digest.hexdigest() != MODEL_WEIGHT_SHA256:
            raise ModelUnavailable("MODEL_WEIGHTS_CHECKSUM_MISMATCH")

        try:
            import torch
            from monai.inferers import SlidingWindowInferer
            from monai.networks.nets import SegResNet
            from monai.transforms import NormalizeIntensity

            if self.device_name is None:
                if torch.backends.mps.is_available():
                    self.device_name = "mps"
                elif torch.cuda.is_available():
                    self.device_name = "cuda:0"
                else:
                    self.device_name = "cpu"
            self._torch = torch
            self.network = SegResNet(
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                init_filters=16,
                in_channels=4,
                out_channels=3,
                dropout_prob=0.2,
            ).to(self.device_name)
            checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.network.load_state_dict(state, strict=True)
            self.network.eval()
            self.inferer = SlidingWindowInferer(roi_size=(240, 240, 160), sw_batch_size=1, overlap=0.5)
            self.normalizer = NormalizeIntensity(nonzero=True, channel_wise=True)
        except Exception as exc:
            # Never expose the raw exception: torch errors can include local paths.
            raise ModelUnavailable("MODEL_INITIALIZATION_FAILED") from exc

    def predict(self, study: AlignedStudy) -> tuple[np.ndarray, float]:
        if self.network is None or self._torch is None:
            self.load()
        assert self._torch is not None and self.normalizer is not None
        torch = self._torch
        try:
            normalized = self.normalizer(study.image.copy())
            image = torch.as_tensor(np.asarray(normalized), dtype=torch.float32, device=self.device_name).unsqueeze(0)
            started = time.perf_counter()
            with torch.inference_mode():
                prediction = self.inferer(image, self.network)
                probabilities = torch.sigmoid(prediction)[0].float().cpu().numpy()
            elapsed = (time.perf_counter() - started) * 1000.0
            labels = region_masks_to_labels(probabilities)
            if labels.shape != study.shape:
                raise InputProtocolError("MAPPING_FAILED")
            return labels, elapsed
        except InputProtocolError:
            raise
        except Exception as exc:
            raise ModelUnavailable("INFERENCE_FAILED") from exc

    @property
    def provenance(self) -> dict[str, str]:
        return {
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "license": MODEL_LICENSE,
            "device": self.device_name or "not initialized",
        }


def summarize_prediction(study: AlignedStudy, labels: np.ndarray) -> dict:
    return {
        "volumes": segmentation_volumes_mm3(labels, study.spacing_mm),
        "affine": study.affine.tolist(),
        "shape": list(study.shape),
        "spacingMm": study.spacing_mm.tolist(),
        "regions": {
            "wholeTumor": int(np.isin(labels, (1, 2, 4)).sum()),
            "tumorCore": int(np.isin(labels, (1, 4)).sum()),
            "enhancingTumor": int((labels == 4).sum()),
        },
    }
