"""Local, reproducible evaluation and experiment utilities.

Manifests explicitly name each input volume. The loader never discovers files by
scanning directories and output records use ordinal aliases instead of source IDs.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .inference import BrainTumorSegmenter, ModelUnavailable, summarize_prediction
from .protocol import (
    CHANNEL_ORDER,
    MODEL_NAME,
    MODEL_VERSION,
    MODEL_WEIGHT_SHA256,
    AlignedStudy,
    InputProtocolError,
    segmentation_metrics,
    validate_study,
)

MSD_IMAGE_CHANNEL_ORDER = ("flair", "t1", "t1c", "t2")
MANIFEST_SCHEMA = "medvision-evaluation-v1"


class ResearchDataError(ValueError):
    """Safe, path-free research data validation error."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class EvaluationCase:
    study: AlignedStudy
    reference: np.ndarray
    convention: str
    dataset: dict[str, Any]


def safe_provenance_text(value: Any, fallback: str = "unspecified", *, url: bool = False) -> str:
    """Keep output provenance useful without serializing paths/control text."""
    if not isinstance(value, str):
        return fallback
    text = value.strip()
    if not text or len(text) > 256 or any(ord(char) < 32 for char in text) or "\\" in text:
        return fallback
    if url:
        return text if text.startswith("https://") and "/Users/" not in text and "/private/" not in text else fallback
    if text.startswith(("/", "~")) or ":\\" in text:
        return fallback
    return text


def _load_volume(path_value: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        image = nib.load(str(Path(path_value).expanduser()))
        return np.asarray(image.dataobj), np.asarray(image.affine, dtype=np.float64)
    except Exception as exc:
        raise ResearchDataError("UNREADABLE_VOLUME") from exc


def _canonical(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        image = nib.as_closest_canonical(nib.Nifti1Image(data, affine), enforce_diag=True)
        return np.asarray(image.dataobj), np.asarray(image.affine, dtype=np.float64)
    except Exception as exc:
        raise ResearchDataError("INVALID_GEOMETRY") from exc


def load_evaluation_case(entry: dict[str, Any], dataset: dict[str, Any]) -> EvaluationCase:
    """Load either four named volumes or one MSD 4D volume and its reference."""
    convention = str(entry.get("referenceConvention", dataset.get("referenceConvention", "")))
    reference_value = entry.get("reference")
    if not isinstance(reference_value, str) or not reference_value:
        raise ResearchDataError("MISSING_REFERENCE")
    if isinstance(entry.get("modalities"), dict):
        sources = entry["modalities"]
        if set(sources) != set(CHANNEL_ORDER) or not all(isinstance(value, str) and value for value in sources.values()):
            raise ResearchDataError("INVALID_MODALITY_MAP")
        volumes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for role in CHANNEL_ORDER:
            data, affine = _load_volume(sources[role])
            data, affine = _canonical(data, affine)
            volumes[role] = (data, affine)
        try:
            study = validate_study(volumes)
        except InputProtocolError as exc:
            raise ResearchDataError(exc.code) from exc
    elif isinstance(entry.get("image"), str) and entry.get("imageChannelOrder") == list(MSD_IMAGE_CHANNEL_ORDER):
        data, affine = _load_volume(entry["image"])
        if data.ndim != 4 or data.shape[-1] != 4:
            raise ResearchDataError("INVALID_MSD_CHANNELS")
        data, affine = _canonical(data, affine)
        try:
            study = validate_study({
                role: (data[..., MSD_IMAGE_CHANNEL_ORDER.index(role)], affine)
                for role in CHANNEL_ORDER
            })
        except InputProtocolError as exc:
            raise ResearchDataError(exc.code) from exc
    else:
        raise ResearchDataError("INVALID_MODALITY_MAP")

    reference, reference_affine = _load_volume(reference_value)
    reference, reference_affine = _canonical(reference, reference_affine)
    if reference.ndim != 3:
        raise ResearchDataError("INVALID_REFERENCE_DIMENSIONS")
    if reference.shape != study.shape or not np.allclose(reference_affine, study.affine, atol=1e-3, rtol=0):
        raise ResearchDataError("REFERENCE_GEOMETRY_MISMATCH")
    if convention not in {"brats_2018_2021", "brats_2023", "brats_2024_gli_post_treatment", "msd_task01"}:
        raise ResearchDataError("UNSUPPORTED_LABEL_CONVENTION")
    if not np.issubdtype(reference.dtype, np.number) or not np.isfinite(reference).all():
        raise ResearchDataError("INVALID_REFERENCE_LABELS")
    # Validate labels before an expensive model call, without rewriting the reference.
    try:
        segmentation_metrics(np.zeros(study.shape, dtype=np.uint8), reference, study.spacing_mm, convention)
    except InputProtocolError as exc:
        raise ResearchDataError("INVALID_REFERENCE_LABELS") from exc
    return EvaluationCase(study, reference, convention, dataset)


def choose_device(torch: Any, requested: str = "auto") -> str:
    """Resolve an explicit backend request without pretending availability is compatibility."""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ModelUnavailable("DEVICE_UNAVAILABLE_MPS")
        return "mps"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ModelUnavailable("DEVICE_UNAVAILABLE_CUDA")
        return "cuda:0"
    raise ModelUnavailable("DEVICE_UNSUPPORTED")


class ResilientSegmenter:
    """Keeps the pinned segmenter intact and falls back to verified CPU weights."""

    def __init__(self, weights: Path, device: str = "auto", *, fallback_to_cpu: bool = True):
        import torch

        self.torch = torch
        self.weights = Path(weights)
        self.device = choose_device(torch, device)
        self.fallback_to_cpu = fallback_to_cpu
        self.segmenter = BrainTumorSegmenter(self.weights, device=self.device)
        self.load_ms = 0.0
        self.fallback_used = False
        self.peak_device_memory_bytes: int | None = None

    def load(self) -> None:
        started = time.perf_counter()
        try:
            self.segmenter.load()
        except ModelUnavailable:
            if self.device == "cpu" or not self.fallback_to_cpu:
                raise
            self.segmenter = BrainTumorSegmenter(self.weights, device="cpu")
            self.segmenter.load()
            self.device = "cpu"
            self.fallback_used = True
        self.load_ms = (time.perf_counter() - started) * 1000

    @property
    def provenance(self) -> dict[str, str]:
        return {**self.segmenter.provenance, "checkpointSha256": MODEL_WEIGHT_SHA256, "checkpointRole": "pinned-baseline"}

    def predict(self, study: AlignedStudy) -> tuple[np.ndarray, float]:
        if self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats()
        try:
            result = self.segmenter.predict(study)
            self.peak_device_memory_bytes = int(self.torch.cuda.max_memory_allocated()) if self.device.startswith("cuda") else None
            return result
        except ModelUnavailable:
            if self.device == "cpu" or not self.fallback_to_cpu:
                raise
            cpu = BrainTumorSegmenter(self.weights, device="cpu")
            started = time.perf_counter()
            cpu.load()
            load_ms = (time.perf_counter() - started) * 1000
            labels, inference_ms = cpu.predict(study)
            self.segmenter = cpu
            self.device = "cpu"
            self.load_ms += load_ms
            self.fallback_used = True
            self.peak_device_memory_bytes = None
            return labels, inference_ms


class ResearchCheckpointSegmenter:
    """Load an experiment checkpoint only after matching its local metadata hash."""

    def __init__(self, weights_path: Path, metadata_path: Path, device: str = "auto", *, fallback_to_cpu: bool = True):
        import torch

        self.torch = torch
        self.weights_path = Path(weights_path)
        self.metadata_path = Path(metadata_path)
        self.device_name = choose_device(torch, device)
        self.fallback_to_cpu = fallback_to_cpu
        self.network = None
        self.inferer = None
        self.normalizer = None
        self.load_ms = 0.0
        self.fallback_used = False
        self.peak_device_memory_bytes = None
        self._metadata: dict[str, Any] | None = None

    def load(self) -> None:
        started = time.perf_counter()
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if metadata.get("schema") != "medvision-training-experiment-v1":
                raise ResearchDataError("EXPERIMENT_METADATA_INVALID")
            model_info = metadata.get("model", {})
            if model_info.get("baseCheckpointSha256") != MODEL_WEIGHT_SHA256:
                raise ResearchDataError("EXPERIMENT_BASE_CHECKPOINT_MISMATCH")
            if model_info.get("bestCheckpointSha256") != checkpoint_sha256(self.weights_path):
                raise ResearchDataError("EXPERIMENT_CHECKPOINT_HASH_MISMATCH")
            from monai.inferers import SlidingWindowInferer
            from monai.networks.nets import SegResNet
            from monai.transforms import NormalizeIntensity

            self.network = SegResNet(
                blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), init_filters=16,
                in_channels=4, out_channels=3, dropout_prob=0.2,
            ).to(self.device_name)
            checkpoint = self.torch.load(self.weights_path, map_location="cpu", weights_only=True)
            state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.network.load_state_dict(state, strict=True)
            self.network.eval()
            self.inferer = SlidingWindowInferer(roi_size=(240, 240, 160), sw_batch_size=1, overlap=0.5)
            self.normalizer = NormalizeIntensity(nonzero=True, channel_wise=True)
            self._metadata = metadata
            self.load_ms = (time.perf_counter() - started) * 1000
        except ResearchDataError:
            raise
        except Exception as exc:
            if self.device_name != "cpu" and self.fallback_to_cpu:
                self.device_name = "cpu"
                self.fallback_used = True
                self.load()
                return
            raise ModelUnavailable("EXPERIMENT_MODEL_INITIALIZATION_FAILED") from exc

    @property
    def provenance(self) -> dict[str, str]:
        return {
            "name": MODEL_NAME,
            "version": f"{MODEL_VERSION}-finetuned",
            "license": "Apache-2.0 base bundle; experiment checkpoint provenance in metadata",
            "device": self.device_name,
            "checkpointSha256": self._metadata.get("model", {}).get("bestCheckpointSha256", "unavailable") if self._metadata else "not initialized",
            "checkpointRole": "fine-tuned-experiment",
            "experimentId": self._metadata.get("experimentId", "unavailable") if self._metadata else "not initialized",
        }

    def predict(self, study: AlignedStudy) -> tuple[np.ndarray, float]:
        if self.network is None:
            self.load()
        assert self.normalizer is not None and self.inferer is not None
        started = time.perf_counter()
        try:
            if self.device_name.startswith("cuda"):
                self.torch.cuda.reset_peak_memory_stats()
            normalized = self.normalizer(study.image.copy())
            tensor = self.torch.as_tensor(np.asarray(normalized), dtype=self.torch.float32, device=self.device_name).unsqueeze(0)
            with self.torch.inference_mode():
                probabilities = self.torch.sigmoid(self.inferer(tensor, self.network))[0].float().cpu().numpy()
            self.peak_device_memory_bytes = int(self.torch.cuda.max_memory_allocated()) if self.device_name.startswith("cuda") else None
            return region_masks_to_labels(probabilities), (time.perf_counter() - started) * 1000
        except Exception as exc:
            if self.device_name != "cpu" and self.fallback_to_cpu:
                cpu = ResearchCheckpointSegmenter(self.weights_path, self.metadata_path, device="cpu", fallback_to_cpu=False)
                cpu.load()
                result = cpu.predict(study)
                self.network, self.inferer, self.normalizer = cpu.network, cpu.inferer, cpu.normalizer
                self.device_name = "cpu"
                self._metadata = cpu._metadata
                self.peak_device_memory_bytes = None
                self.load_ms += cpu.load_ms
                self.fallback_used = True
                return result
            raise ModelUnavailable("EXPERIMENT_INFERENCE_FAILED") from exc


def _aggregate(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, dict[str, float | int | None]]:
    result = {}
    for field in fields:
        values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float)) and math.isfinite(float(row[field]))]
        result[field] = {
            "n": len(values),
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "stddev": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return result


def aggregate_evaluations(cases: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [case for case in cases if case.get("status") == "success"]
    failed = [case for case in cases if case.get("status") == "failed"]
    pending = [case for case in cases if case.get("status") == "pending"]
    metrics = {}
    for region in ("whole_tumor", "tumor_core", "enhancing_tumor"):
        for metric in ("dice", "iou"):
            key = f"{region}.{metric}"
            metrics[key] = _aggregate(
                [{key: case.get("metrics", {}).get(region, {}).get(metric)} for case in successful], [key]
            )[key]
    metrics.update(_aggregate(successful, [
        "inferenceMs", "predictedWholeCm3", "predictedCoreCm3", "predictedEnhancingCm3",
        "referenceWholeCm3", "referenceCoreCm3", "referenceEnhancingCm3",
    ]))
    return {"caseCount": len(cases), "successfulCaseCount": len(successful), "failedCaseCount": len(failed), "pendingCaseCount": len(pending), "metrics": metrics}


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".medvision-eval-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


@contextmanager
def _evaluation_lock(path: Path):
    """Use an OS lock so interrupted runs release ownership automatically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResearchDataError("EVALUATION_ALREADY_RUNNING") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "startedAtUtc": datetime.now(timezone.utc).isoformat()}) + "\n")
        handle.flush()
        yield
    finally:
        handle.close()


def _write_evaluation(output_path: Path, document: dict[str, Any]) -> None:
    _atomic_json(output_path, document)
    _write_csv(output_path.with_suffix(".csv"), document["cases"])


def evaluate_manifest(
    manifest_path: Path,
    output_path: Path,
    segmenter: Any,
    *,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate explicit cases, atomically checkpointing each result for safe resume."""
    try:
        manifest_path = Path(manifest_path).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ResearchDataError("INVALID_MANIFEST") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ResearchDataError("INVALID_MANIFEST")
    dataset = manifest.get("dataset")
    entries = manifest.get("cases")
    if not isinstance(dataset, dict) or not isinstance(entries, list) or not entries:
        raise ResearchDataError("INVALID_MANIFEST")
    if len(entries) > 10000:
        raise ResearchDataError("CASE_LIMIT_EXCEEDED")

    output_path = Path(output_path).expanduser().resolve()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    weight_value = getattr(segmenter, "weights", getattr(segmenter, "weights_path", None))
    checkpoint_path = Path(weight_value).expanduser().resolve() if weight_value is not None else None
    checkpoint_hash = checkpoint_sha256(checkpoint_path) if checkpoint_path and checkpoint_path.is_file() else None
    identity = {"manifestSha256": manifest_hash, "checkpointSha256": checkpoint_hash}

    with _evaluation_lock(output_path):
        existing = None
        if output_path.is_file():
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ResearchDataError("EXISTING_RESULTS_INVALID") from exc
            old_identity = existing.get("execution", {})
            if existing.get("schema") != "medvision-evaluation-results-v1" or any(old_identity.get(key) != value for key, value in identity.items()):
                raise ResearchDataError("RESULTS_RESUME_MISMATCH")

        old_cases = existing.get("cases", []) if existing else []
        results = []
        for index in range(1, len(entries) + 1):
            alias = f"case_{index:06d}"
            old = old_cases[index - 1] if index <= len(old_cases) else None
            if isinstance(old, dict) and old.get("case") == alias and old.get("status") == "success":
                results.append(old)
            else:
                results.append(old if isinstance(old, dict) and old.get("case") == alias else {"case": alias, "status": "pending"})

        execution = {
            **identity,
            "manifestPath": str(manifest_path),
            "checkpointPath": str(checkpoint_path) if checkpoint_path else None,
            "outputDirectory": str(output_path.parent),
            "command": command or [],
            "pid": os.getpid(),
            "startedAtUtc": existing.get("execution", {}).get("startedAtUtc") if existing else datetime.now(timezone.utc).isoformat(),
            "lastStartedAtUtc": datetime.now(timezone.utc).isoformat(),
            "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        provenance = {
        "dataset": {
            "name": safe_provenance_text(dataset.get("name")),
            "version": safe_provenance_text(dataset.get("version")),
            "license": safe_provenance_text(dataset.get("license")),
            "source": safe_provenance_text(dataset.get("source"), url=True),
            "independenceAssessment": safe_provenance_text(dataset.get("independenceAssessment"), "not established"),
            "referenceConvention": safe_provenance_text(dataset.get("referenceConvention"), "per case"),
        },
        "model": {
            "name": segmenter.provenance.get("name", MODEL_NAME),
            "version": segmenter.provenance.get("version", MODEL_VERSION),
            "license": segmenter.provenance.get("license", "unknown"),
            "checkpointSha256": segmenter.provenance.get("checkpointSha256", MODEL_WEIGHT_SHA256),
            "checkpointRole": segmenter.provenance.get("checkpointRole", "pinned-baseline"),
            "experimentId": segmenter.provenance.get("experimentId"),
        },
        "preprocessing": {"channelOrder": list(CHANNEL_ORDER), "orientation": "RAS", "spacingMm": [1, 1, 1], "normalization": "nonzero, per-channel MONAI NormalizeIntensity"},
        "runtime": {
            "device": segmenter.provenance.get("device", "unknown"),
            "modelLoadMs": round(float(segmenter.load_ms), 3),
            "cpuFallbackUsed": bool(getattr(segmenter, "fallback_used", False)),
            "processPeakRssBytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)),
            "peakDeviceMemoryMeasured": bool(str(segmenter.provenance.get("device", "")).startswith("cuda")),
        },
        }
        document = {
        "schema": "medvision-evaluation-results-v1",
        "createdAtUtc": existing.get("createdAtUtc") if existing else datetime.now(timezone.utc).isoformat(),
        "researchUseOnly": True,
        "datasetProvenance": provenance,
        "summary": aggregate_evaluations(results),
        "cases": results,
        "execution": execution,
        }
        _write_evaluation(output_path, document)

        model_loaded = False
        model_load_error: str | None = None
        for index, entry in enumerate(entries, start=1):
            if results[index - 1].get("status") == "success":
                continue
            alias = f"case_{index:06d}"
            results[index - 1] = {"case": alias, "status": "pending"}
            document["summary"] = aggregate_evaluations(results)
            execution["updatedAtUtc"] = datetime.now(timezone.utc).isoformat()
            _write_evaluation(output_path, document)
            started = time.perf_counter()
            if not model_loaded and model_load_error is None:
                try:
                    segmenter.load()
                    model_loaded = True
                except ModelUnavailable as exc:
                    model_load_error = str(exc)
                except Exception:
                    model_load_error = "MODEL_LOAD_FAILED"
            if model_load_error:
                results[index - 1] = {"case": alias, "status": "failed", "errorCode": model_load_error}
            else:
                try:
                    if not isinstance(entry, dict):
                        raise ResearchDataError("INVALID_CASE_ENTRY")
                    case = load_evaluation_case(entry, dataset)
                    prediction, elapsed_ms = segmenter.predict(case.study)
                    metrics = segmentation_metrics(prediction, case.reference, case.study.spacing_mm, case.convention)
                    pred_volumes = summarize_prediction(case.study, prediction)["volumes"]
                    reference_mm3 = _reference_volumes(case.reference, case.study.spacing_mm, case.convention)
                    results[index - 1] = {
                        "case": alias,
                        "status": "success",
                        "referenceConvention": case.convention,
                        "metrics": metrics,
                        "inferenceMs": round(float(elapsed_ms), 3),
                        "predictedWholeCm3": pred_volumes["whole_tumor_mm3"] / 1000,
                        "predictedCoreCm3": pred_volumes["tumor_core_mm3"] / 1000,
                        "predictedEnhancingCm3": pred_volumes["enhancing_tumor_mm3"] / 1000,
                        "referenceWholeCm3": reference_mm3["whole_tumor_mm3"] / 1000,
                        "referenceCoreCm3": reference_mm3["tumor_core_mm3"] / 1000,
                        "referenceEnhancingCm3": reference_mm3["enhancing_tumor_mm3"] / 1000,
                        "device": segmenter.provenance.get("device", "unknown"),
                        "peakDeviceMemoryBytes": getattr(segmenter, "peak_device_memory_bytes", None),
                        "caseProcessingMs": round((time.perf_counter() - started) * 1000, 3),
                    }
                except (ResearchDataError, InputProtocolError) as exc:
                    code = exc.code if hasattr(exc, "code") else "INVALID_INPUT"
                    results[index - 1] = {"case": alias, "status": "failed", "errorCode": str(code)}
                except ModelUnavailable as exc:
                    results[index - 1] = {"case": alias, "status": "failed", "errorCode": str(exc)}
                except Exception:
                    # Raw exceptions can include local paths or identifiers.
                    results[index - 1] = {"case": alias, "status": "failed", "errorCode": "CASE_PROCESSING_FAILED"}
            document["summary"] = aggregate_evaluations(results)
            document["datasetProvenance"]["runtime"].update({
                "device": segmenter.provenance.get("device", "unknown"),
                "modelLoadMs": round(float(segmenter.load_ms), 3),
                "cpuFallbackUsed": bool(getattr(segmenter, "fallback_used", False)),
            })
            execution["updatedAtUtc"] = datetime.now(timezone.utc).isoformat()
            execution["completedCaseCount"] = sum(row.get("status") in {"success", "failed"} for row in results)
            execution["failedCaseCount"] = sum(row.get("status") == "failed" for row in results)
            _write_evaluation(output_path, document)

        execution["status"] = "completed"
        execution["updatedAtUtc"] = datetime.now(timezone.utc).isoformat()
        execution["completedCaseCount"] = len(entries)
        execution["failedCaseCount"] = sum(row.get("status") == "failed" for row in results)
        document["summary"] = aggregate_evaluations(results)
        _write_evaluation(output_path, document)
        return document


def _reference_volumes(reference: np.ndarray, spacing: np.ndarray, convention: str) -> dict[str, float]:
    regions = {
        "brats_2018_2021": ((1, 2, 4), (1, 4), (4,)),
        "brats_2023": ((1, 2, 3), (1, 3), (3,)),
        "brats_2024_gli_post_treatment": ((1, 2, 3), (1, 3), (3,)),
        "msd_task01": ((1, 2, 3), (2, 3), (3,)),
    }
    if convention not in regions:
        raise ResearchDataError("UNSUPPORTED_LABEL_CONVENTION")
    voxel = float(np.prod(spacing))
    whole, core, enhancing = regions[convention]
    return {
        "whole_tumor_mm3": float(np.isin(reference, whole).sum() * voxel),
        "tumor_core_mm3": float(np.isin(reference, core).sum() * voxel),
        "enhancing_tumor_mm3": float(np.isin(reference, enhancing).sum() * voxel),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["case", "status", "errorCode", "referenceConvention", "inferenceMs", "device", "predictedWholeCm3", "predictedCoreCm3", "predictedEnhancingCm3", "referenceWholeCm3", "referenceCoreCm3", "referenceEnhancingCm3"]
    for region in ("whole_tumor", "tumor_core", "enhancing_tumor"):
        fields.extend((f"{region}_dice", f"{region}_iou", f"{region}_hausdorff95_mm"))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".medvision-eval-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                flat = {key: row.get(key) for key in fields}
                for region, metrics in row.get("metrics", {}).items():
                    for metric, value in metrics.items():
                        flat[f"{region}_{metric}"] = value
                writer.writerow(flat)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def paired_comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Pair by ordinal anonymized case aliases and report experimental deltas."""
    def cohort(document: dict[str, Any]) -> tuple[Any, ...] | None:
        provenance = document.get("datasetProvenance")
        dataset = provenance.get("dataset") if isinstance(provenance, dict) else None
        if not isinstance(dataset, dict):
            return None
        return tuple(dataset.get(key) for key in ("name", "version", "license", "source", "referenceConvention"))

    baseline_cohort, candidate_cohort = cohort(baseline), cohort(candidate)
    if baseline_cohort is not None and candidate_cohort is not None and baseline_cohort != candidate_cohort:
        raise ResearchDataError("COMPARISON_DATASET_MISMATCH")
    for document in (baseline, candidate):
        for row in document.get("cases", []):
            alias = row.get("case") if isinstance(row, dict) else None
            if not isinstance(alias, str) or len(alias) != 11 or not alias.startswith("case_") or not alias[5:].isdigit():
                raise ResearchDataError("INVALID_RESULT_CASE_ALIAS")
    left_all = {row["case"]: row for row in baseline.get("cases", [])}
    right_all = {row["case"]: row for row in candidate.get("cases", [])}
    paired = []
    failure_cases = []
    for alias in sorted(left_all.keys() | right_all.keys()):
        left_row, right_row = left_all.get(alias), right_all.get(alias)
        if left_row is None or right_row is None or left_row.get("status") != "success" or right_row.get("status") != "success":
            failure_cases.append({
                "case": alias,
                "baselineStatus": left_row.get("status", "missing") if left_row else "missing",
                "candidateStatus": right_row.get("status", "missing") if right_row else "missing",
                "baselineErrorCode": left_row.get("errorCode") if left_row else None,
                "candidateErrorCode": right_row.get("errorCode") if right_row else None,
            })
            continue
        row: dict[str, Any] = {"case": alias}
        for region in ("whole_tumor", "tumor_core", "enhancing_tumor"):
            for metric in ("dice", "iou"):
                key = f"{region}.{metric}"
                row[key] = float(right_row["metrics"][region][metric]) - float(left_row["metrics"][region][metric])
        row["inferenceMs"] = float(right_row["inferenceMs"]) - float(left_row["inferenceMs"])
        left_memory, right_memory = left_row.get("peakDeviceMemoryBytes"), right_row.get("peakDeviceMemoryBytes")
        row["peakDeviceMemoryBytes"] = (int(right_memory) - int(left_memory)) if isinstance(left_memory, (int, float)) and isinstance(right_memory, (int, float)) else None
        paired.append(row)
    metric_differences = {}
    for region in ("whole_tumor", "tumor_core", "enhancing_tumor"):
        for metric in ("dice", "iou"):
            key = f"{region}.{metric}"
            values = [row[key] for row in paired]
            metric_differences[key] = statistics.fmean(values) if values else None
    memory_differences = [row["peakDeviceMemoryBytes"] for row in paired if row["peakDeviceMemoryBytes"] is not None]
    baseline_provenance = baseline.get("datasetProvenance")
    candidate_provenance = candidate.get("datasetProvenance")
    baseline_runtime = baseline_provenance.get("runtime", {}) if isinstance(baseline_provenance, dict) else {}
    candidate_runtime = candidate_provenance.get("runtime", {}) if isinstance(candidate_provenance, dict) else {}
    return {
        "pairedCaseCount": len(paired),
        "failedOrUnpairedCases": failure_cases,
        "pairingNote": "Pairs use ordinal aliases and are valid only when both results use the same manifest in identical order.",
        "cases": paired,
        "meanPairedDifferences": {
            **metric_differences,
            "inferenceMs": statistics.fmean([row["inferenceMs"] for row in paired]) if paired else None,
            "peakDeviceMemoryBytes": statistics.fmean(memory_differences) if memory_differences else None,
        },
        "baselineProcessPeakRssBytes": baseline_runtime.get("processPeakRssBytes") if isinstance(baseline_runtime, dict) else None,
        "candidateProcessPeakRssBytes": candidate_runtime.get("processPeakRssBytes") if isinstance(candidate_runtime, dict) else None,
    }


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
