"""Loopback-only API. Requests are processed locally and files are never retained."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Annotated

import nibabel as nib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict
from nibabel.orientations import io_orientation, ornt_transform, apply_orientation
from starlette.concurrency import run_in_threadpool

from .inference import BrainTumorSegmenter, ModelUnavailable, summarize_prediction
from .protocol import (
    CHANNEL_ORDER,
    MODEL_NAME,
    MODEL_VERSION,
    InputProtocolError,
    segmentation_metrics,
    validate_study,
    voxel_spacing,
)
from .rag import RAGService, new_context_id

MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
PUBLIC_SAMPLE_DIR = Path(os.environ.get("MEDVISION_PUBLIC_SAMPLE_DIR", "/private/tmp/medvision-brats-case"))
PUBLIC_SAMPLE_RESULT = Path(os.environ.get("MEDVISION_PUBLIC_SAMPLE_RESULT", "/private/tmp/medvision-real-inference-result.json"))
PUBLIC_SAMPLE_MASK = Path(os.environ.get("MEDVISION_PUBLIC_SAMPLE_MASK", "/private/tmp/medvision-real-prediction.nii.gz"))
DEFAULT_WEIGHTS = Path.home() / ".cache" / "medvision" / "models" / "brats_mri_segmentation-v0.5.4-model.pt"
SAFE_MESSAGES = {
    "MISSING_SEQUENCE": "Select one volume for each required MRI sequence: T1c, T1, T2, and FLAIR.",
    "INVALID_DIMENSIONS": "Each input must be a 3D MRI volume with a supported spatial shape.",
    "INVALID_PIXELS": "An input contains invalid or non-finite image values.",
    "INVALID_GEOMETRY": "A volume has invalid spatial geometry.",
    "UNSUPPORTED_SPACING": "This model requires BraTS-preprocessed 1 mm isotropic MRI volumes.",
    "UNSUPPORTED_ORIENTATION": "This model requires BraTS-preprocessed RAS orientation.",
    "MISMATCHED_GEOMETRY": "The four volumes are not co-registered on the same voxel grid. No prediction was produced.",
    "EMPTY_IMAGE": "The selected volumes contain no nonzero image data.",
    "INVALID_PREDICTION": "The model returned an invalid segmentation result.",
    "MAPPING_FAILED": "The predicted mask could not be mapped to the source voxel grid.",
    "INVALID_REFERENCE": "The reference mask is incompatible with the prediction geometry.",
}


def _load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    return np.asarray(image.dataobj, dtype=np.float32), np.asarray(image.affine, dtype=np.float64)


def _to_ras(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the bundle's axis reorientation without interpolation or resampling."""
    image = nib.Nifti1Image(data, affine)
    canonical = nib.as_closest_canonical(image, enforce_diag=True)
    return np.asarray(canonical.dataobj, dtype=np.float32), np.asarray(canonical.affine, dtype=np.float64)


def _restore_orientation(data: np.ndarray, source_affine: np.ndarray, target_affine: np.ndarray) -> np.ndarray:
    """Return a RAS prediction to the first input's voxel indexing and orientation."""
    transform = ornt_transform(io_orientation(source_affine), io_orientation(target_affine))
    return np.asarray(apply_orientation(data, transform), dtype=np.uint8)


def _mask_digest(mask: np.ndarray, affine: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(mask, dtype=np.uint8).tobytes(order="C"))
    digest.update(np.asarray(affine, dtype=np.float64).tobytes(order="C"))
    digest.update(json.dumps(list(mask.shape)).encode())
    return digest.hexdigest()


class RAGQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    studyContextId: str | None = None


def create_app(segmenter: BrainTumorSegmenter | None = None) -> FastAPI:
    app = FastAPI(title="MedVision local research service", docs_url=None, redoc_url=None)
    app.state.segmenter = segmenter or BrainTumorSegmenter(Path(os.environ.get("MEDVISION_MODEL_PATH", DEFAULT_WEIGHTS)))
    app.state.rag = RAGService()
    app.state.study_contexts = {}

    @app.middleware("http")
    async def reject_remote_browser_origins(request: Request, call_next):
        # Loopback binding prevents remote clients from reaching this API. This
        # Origin check blocks cross-site form posts from sending scans to localhost.
        origin = request.headers.get("origin")
        if origin:
            from urllib.parse import urlparse

            parsed = urlparse(origin)
            if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "[::1]"}:
                return JSONResponse({"detail": "Only requests from a local MedVision session are accepted."}, status_code=403)
        return await call_next(request)

    @app.get("/api/health")
    async def health():
        model = app.state.segmenter
        ready = model.network is not None
        return {
            "ready": ready,
            "model": MODEL_NAME,
            "version": MODEL_VERSION,
            "localOnly": app.state.rag.provider_name == "local-extractive",
        }

    @app.get("/api/rag/status")
    async def rag_status():
        return {
            "provider": app.state.rag.provider_name,
            "remoteStudyContextEnabled": app.state.rag.remote_study_context_enabled,
        }

    if os.environ.get("MEDVISION_ENABLE_PUBLIC_BRATS_SAMPLE") == "1":
        @app.get("/api/dev/public-brats/flair")
        async def public_sample_flair():
            path = PUBLIC_SAMPLE_DIR / "flair.nii.gz"
            if not path.is_file():
                raise HTTPException(status_code=503, detail="The fixed public development sample is not staged on this computer.")
            return FileResponse(path, media_type="application/gzip", filename="flair.nii.gz", headers={"Cache-Control": "no-store"})

        @app.get("/api/dev/public-brats/result")
        async def public_sample_result():
            if not PUBLIC_SAMPLE_RESULT.is_file() or not PUBLIC_SAMPLE_MASK.is_file():
                raise HTTPException(status_code=503, detail="A real local inference result is not available for the public development sample.")
            try:
                stored = json.loads(PUBLIC_SAMPLE_RESULT.read_text())
                mask_image = nib.load(str(PUBLIC_SAMPLE_MASK))
                source_image = nib.load(str(PUBLIC_SAMPLE_DIR / "flair.nii.gz"))
                if mask_image.shape != source_image.shape or not np.allclose(
                    mask_image.affine, source_image.affine, atol=1e-3, rtol=0
                ):
                    raise ValueError("The fixed sample prediction does not match its MRI grid.")
                raw_volumes = stored["volumesCm3"]
                result = {
                    "model": stored["model"],
                    "elapsedMs": float(stored["inferenceMs"]),
                    "volumes": {
                        "whole_tumor_mm3": float(raw_volumes["whole_tumor_mm3"]) * 1000,
                        "tumor_core_mm3": float(raw_volumes["tumor_core_mm3"]) * 1000,
                        "enhancing_tumor_mm3": float(raw_volumes["enhancing_tumor_mm3"]) * 1000,
                    },
                    "limitations": stored["limitations"],
                    "metrics": stored.get("metrics"),
                    "referenceConvention": stored.get("referenceConvention"),
                    "shape": list(source_image.shape),
                    "affine": np.asarray(source_image.affine, dtype=np.float64).tolist(),
                    "spacingMm": voxel_spacing(source_image.affine).tolist(),
                    "maskEncoding": "gzip-nifti-uint8-base64",
                    "mask": base64.b64encode(PUBLIC_SAMPLE_MASK.read_bytes()).decode("ascii"),
                }
                context = {
                    "modelName": stored["model"]["name"], "modelVersion": stored["model"]["version"],
                    "modalities": ["T1c", "T1", "T2", "FLAIR"],
                    "predictedVolumesCm3": {key: float(value) for key, value in raw_volumes.items()},
                    "inferenceMs": float(stored["inferenceMs"]),
                    "groundTruthMetrics": stored.get("metrics"),
                    "referenceConvention": stored.get("referenceConvention"),
                    "_predictionDigest": _mask_digest(np.asarray(mask_image.dataobj), source_image.affine),
                }
                result["studyContextId"] = new_context_id(app.state.study_contexts, context)
            except Exception as exc:
                raise HTTPException(status_code=503, detail="The public development result could not be read safely.") from exc
            return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @app.post("/api/segment")
    async def segment(
        t1c: Annotated[UploadFile, File()],
        t1: Annotated[UploadFile, File()],
        t2: Annotated[UploadFile, File()],
        flair: Annotated[UploadFile, File()],
    ):
        uploads = {"t1c": t1c, "t1": t1, "t2": t2, "flair": flair}
        sizes: dict[str, bytes] = {}
        total = 0
        try:
            for role in CHANNEL_ORDER:
                payload = await uploads[role].read(MAX_FILE_BYTES + 1)
                total += len(payload)
                if len(payload) > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                    raise HTTPException(status_code=413, detail="The selected research study exceeds the local processing size limit.")
                sizes[role] = payload
            with tempfile.TemporaryDirectory(prefix="medvision-ai-") as temp_dir:
                root = Path(temp_dir)
                volumes = {}
                source_affine = None
                source_shape = None
                for role, payload in sizes.items():
                    suffix = ".nii.gz" if payload[:2] == b"\x1f\x8b" else ".nii"
                    path = root / f"{role}{suffix}"
                    path.write_bytes(payload)
                    try:
                        data, affine = _load_nifti(path)
                        if role == CHANNEL_ORDER[0]:
                            source_affine, source_shape = affine.copy(), data.shape
                        volumes[role] = _to_ras(data, affine)
                    except Exception as exc:
                        raise HTTPException(status_code=400, detail="An input is not a readable NIfTI volume.") from exc
                study = validate_study(volumes)
                segmenter = app.state.segmenter
                labels, elapsed_ms = await run_in_threadpool(segmenter.predict, study)
                if source_affine is None or source_shape is None:
                    raise HTTPException(status_code=422, detail="The selected study does not contain a usable source geometry.")
                labels = _restore_orientation(labels, study.affine, source_affine)
                if labels.shape != source_shape:
                    raise HTTPException(status_code=422, detail="Prediction geometry could not be restored to the source MRI.")
                output_study = study.__class__(
                    study.image,
                    source_affine,
                    np.linalg.norm(source_affine[:3, :3], axis=0),
                    tuple(source_shape),
                )
                summary = summarize_prediction(output_study, labels)
                summary.update({
                    "model": segmenter.provenance,
                    "elapsedMs": round(elapsed_ms, 1),
                    "limitations": [
                        "Research and educational use only; not for diagnosis or treatment decisions.",
                        "Validated only for co-registered, skull-stripped BraTS-like T1c, T1, T2, and FLAIR volumes at 1 mm isotropic resolution.",
                        "Output is a model prediction, not a medically confirmed finding.",
                    ],
                    "maskEncoding": "gzip-nifti-uint8-base64",
                    "mask": base64.b64encode(gzip.compress(nib.Nifti1Image(labels, source_affine).to_bytes(), compresslevel=6)).decode("ascii"),
                })
                volumes_cm3 = {key.removesuffix("_mm3"): value / 1000.0 for key, value in summary["volumes"].items()}
                context = {
                    "modelName": segmenter.provenance["name"], "modelVersion": segmenter.provenance["version"],
                    "modalities": ["T1c", "T1", "T2", "FLAIR"],
                    "predictedVolumesCm3": volumes_cm3, "inferenceMs": float(elapsed_ms),
                    "groundTruthMetrics": None, "referenceConvention": None,
                    "_predictionDigest": _mask_digest(labels, source_affine),
                }
                summary["studyContextId"] = new_context_id(app.state.study_contexts, context)
                return JSONResponse(summary)
        except InputProtocolError as exc:
            raise HTTPException(status_code=422, detail=SAFE_MESSAGES.get(exc.code, "The selected study does not meet the model input protocol.")) from exc
        except ModelUnavailable as exc:
            detail = {
                "MODEL_WEIGHTS_MISSING": "The locally installed pretrained model weights are missing. Install the official checksum-verified MONAI bundle first.",
                "MODEL_WEIGHTS_UNAVAILABLE": "The local pretrained model weights could not be read.",
                "MODEL_WEIGHTS_CHECKSUM_MISMATCH": "The local pretrained model weights failed integrity verification.",
                "MODEL_INITIALIZATION_FAILED": "The pretrained model could not be initialized with the installed local runtime.",
                "INFERENCE_FAILED": "Local model inference failed. Check available memory and supported hardware.",
            }.get(str(exc), "Local model inference is unavailable.")
            raise HTTPException(status_code=503, detail=detail) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Local segmentation could not be completed safely.") from exc
        finally:
            for upload in uploads.values():
                await upload.close()

    @app.post("/api/evaluate")
    async def evaluate(
        prediction: Annotated[UploadFile, File()],
        reference: Annotated[UploadFile, File()],
        reference_convention: Annotated[str, Form()],
        study_context_id: Annotated[str | None, Form()] = None,
    ):
        try:
            pred_bytes = await prediction.read(MAX_FILE_BYTES + 1)
            ref_bytes = await reference.read(MAX_FILE_BYTES + 1)
            if len(pred_bytes) > MAX_FILE_BYTES or len(ref_bytes) > MAX_FILE_BYTES:
                raise HTTPException(status_code=413, detail="The selected masks exceed the local evaluation size limit.")
            with tempfile.TemporaryDirectory(prefix="medvision-eval-") as temp_dir:
                pred_path, ref_path = Path(temp_dir) / "prediction.nii.gz", Path(temp_dir) / "reference.nii.gz"
                pred_path.write_bytes(pred_bytes)
                ref_path.write_bytes(ref_bytes)
                try:
                    pred_img, ref_img = nib.load(str(pred_path)), nib.load(str(ref_path))
                    pred = np.asarray(pred_img.dataobj)
                    ref = np.asarray(ref_img.dataobj)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail="Both evaluation inputs must be readable NIfTI masks.") from exc
                if pred.shape != ref.shape or not np.allclose(pred_img.affine, ref_img.affine, atol=1e-3, rtol=0):
                    raise HTTPException(status_code=422, detail="Prediction and reference masks do not share the same spatial grid.")
                context = app.state.study_contexts.get(study_context_id) if study_context_id else None
                if study_context_id and (context is None or context["_predictionDigest"] != _mask_digest(pred, pred_img.affine)):
                    raise HTTPException(status_code=409, detail="Evaluation does not match the active model prediction context.")
                metrics = await run_in_threadpool(
                    segmentation_metrics,
                    pred,
                    ref,
                    np.linalg.norm(pred_img.affine[:3, :3], axis=0),
                    reference_convention,
                )
                if context is not None:
                    context["groundTruthMetrics"] = metrics
                    context["referenceConvention"] = reference_convention
                return {
                    "metrics": metrics,
                    "referenceIsUserSupplied": True,
                    "referenceConvention": reference_convention,
                }
        except InputProtocolError as exc:
            raise HTTPException(status_code=422, detail=SAFE_MESSAGES.get(exc.code, "The masks are incompatible.")) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Local evaluation could not be completed safely.") from exc
        finally:
            await prediction.close()
            await reference.close()

    @app.post("/api/rag/query")
    async def rag_query(payload: RAGQuestion):
        context = None
        if payload.studyContextId:
            stored = app.state.study_contexts.get(payload.studyContextId)
            if stored is None:
                raise HTTPException(status_code=404, detail="The active study context is unavailable. Run local segmentation again.")
            context = {key: value for key, value in stored.items() if not key.startswith("_")}
        try:
            return app.state.rag.answer(payload.question, context)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="The local research assistant could not complete this question safely.") from exc

    return app


app = create_app()
