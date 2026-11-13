"""One-step, research-only MOTUM fine-tuning adapter for the MONAI SegResNet."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import resource
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import MODEL_WEIGHT_SHA256
from .research import ResearchDataError


MODALITY_SUFFIXES = {
    "t1c": "_t1ce.nii.gz",
    "t1": "_t1.nii.gz",
    "t2": "_t2.nii.gz",
    "flair": "_flair.nii.gz",
}
MASK_SUFFIXES = {
    "nce_flair": "flair_seg_label1.nii.gz",
    "ce_t1c": "t1ce_seg_label2.nii.gz",
}
# The request's 19/4/5 counts total 28, while the verified cohort has 29 cases.
# Keep validation and untouched evaluation at the requested sizes; assign the
# additional subject to training so that no acquired glioma case is discarded.
SPLIT_COUNTS = {"train": 20, "validation": 4, "evaluation": 5}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _acquisition_cases(dataset_root: Path) -> list[dict[str, Any]]:
    acquisition_path = dataset_root / "provenance" / "selective-acquisition.json"
    try:
        acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
        records = acquisition["files"]
    except Exception as exc:
        raise ResearchDataError("MOTUM_ACQUISITION_MANIFEST_INVALID") from exc

    grouped: dict[str, dict[str, str]] = {}
    for record in records:
        subject = record.get("subjectId")
        relative_path = record.get("relativePath")
        archive_path = record.get("archivePath")
        if not isinstance(subject, str) or not isinstance(relative_path, str) or not isinstance(archive_path, str):
            raise ResearchDataError("MOTUM_FILE_RECORD_INVALID")
        basename = Path(archive_path).name
        role = next((name for name, suffix in MODALITY_SUFFIXES.items() if basename.endswith(suffix)), None)
        role = role or next((name for name, suffix in MASK_SUFFIXES.items() if basename == suffix), None)
        if role is None or role in grouped.setdefault(subject, {}):
            raise ResearchDataError("MOTUM_FILE_SET_INVALID")
        resolved = (dataset_root / relative_path).resolve()
        subjects_root = (dataset_root / "subjects").resolve()
        if subjects_root not in resolved.parents or not resolved.is_file():
            raise ResearchDataError("MOTUM_FILE_MISSING")
        grouped[subject][role] = str(resolved)

    required = set(MODALITY_SUFFIXES) | set(MASK_SUFFIXES)
    if len(grouped) != 29 or any(set(files) != required for files in grouped.values()):
        raise ResearchDataError("MOTUM_COHORT_INCOMPLETE")
    return [
        {"patientId": subject, **files}
        for subject, files in sorted(grouped.items())
    ]


def _split_cases(cases: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    if len(cases) != sum(SPLIT_COUNTS.values()):
        raise ResearchDataError("MOTUM_SPLIT_COUNT_MISMATCH")
    order = list(range(len(cases)))
    random.Random(seed).shuffle(order)
    train_end = SPLIT_COUNTS["train"]
    validation_end = train_end + SPLIT_COUNTS["validation"]
    partitions = {
        "train": [cases[index] for index in order[:train_end]],
        "validation": [cases[index] for index in order[train_end:validation_end]],
        "evaluation": [cases[index] for index in order[validation_end:]],
    }
    groups = [{case["patientId"] for case in items} for items in partitions.values()]
    if any(groups[left] & groups[right] for left in range(3) for right in range(left + 1, 3)):
        raise ResearchDataError("MOTUM_PATIENT_SPLIT_LEAKAGE")
    return partitions


def _preprocess_case(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Resample to canonical RAS 1 mm; images linear, independent masks nearest-neighbor."""
    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to, resample_to_output

        source_images = {
            role: nib.as_closest_canonical(nib.load(case[role]))
            for role in MODALITY_SUFFIXES
        }
        source_masks = {
            role: nib.as_closest_canonical(nib.load(case[role]))
            for role in MASK_SUFFIXES
        }
        source_reference = source_images["t1c"]
        for volume in (*source_images.values(), *source_masks.values()):
            if volume.shape != source_reference.shape or not np.allclose(volume.affine, source_reference.affine, atol=1e-3, rtol=0):
                raise ResearchDataError("MOTUM_SOURCE_GEOMETRY_MISMATCH")
        reference = resample_to_output(source_images["t1c"], voxel_sizes=(1.0, 1.0, 1.0), order=1)
        target = (reference.shape[:3], reference.affine)
        images = []
        for role in ("t1c", "t1", "t2", "flair"):
            aligned = resample_from_to(source_images[role], target, order=1, mode="constant", cval=0)
            if aligned.shape != reference.shape or not np.allclose(aligned.affine, reference.affine, atol=1e-5, rtol=0):
                raise ResearchDataError("MOTUM_RESAMPLED_GEOMETRY_MISMATCH")
            values = np.asarray(aligned.dataobj, dtype=np.float32)
            if values.ndim != 3 or not np.isfinite(values).all():
                raise ResearchDataError("MOTUM_IMAGE_INVALID")
            images.append(values)
        targets = []
        observed_codes: dict[str, list[int]] = {}
        for role in ("nce_flair", "ce_t1c"):
            aligned = resample_from_to(source_masks[role], target, order=0, mode="constant", cval=0)
            if aligned.shape != reference.shape or not np.allclose(aligned.affine, reference.affine, atol=1e-5, rtol=0):
                raise ResearchDataError("MOTUM_RESAMPLED_GEOMETRY_MISMATCH")
            values = np.asarray(aligned.dataobj)
            if values.ndim != 3 or not np.isfinite(values).all():
                raise ResearchDataError("MOTUM_MASK_INVALID")
            foreground, codes = _binary_foreground(values)
            observed_codes[role] = codes
            # These are separate binary region masks. Foreground is any nonzero code,
            # including the one file whose foreground is encoded as 1 instead of 2.
            targets.append(foreground)

        image = np.stack(images, axis=0)
        masks = np.stack(targets, axis=0).astype(np.float32, copy=False)
        if image.shape[1:] != masks.shape[1:] or any(a.shape != reference.shape for a in images):
            raise ResearchDataError("MOTUM_RESAMPLED_GEOMETRY_MISMATCH")
        if not np.allclose(nib.affines.voxel_sizes(reference.affine), (1, 1, 1), atol=1e-5, rtol=0):
            raise ResearchDataError("MOTUM_RESAMPLED_SPACING_INVALID")

        from monai.transforms import NormalizeIntensity

        image = np.asarray(NormalizeIntensity(nonzero=True, channel_wise=True)(image), dtype=np.float32)
        return image, masks, {
            "shape": list(reference.shape),
            "spacingMm": [float(value) for value in nib.affines.voxel_sizes(reference.affine)],
            "orientation": list(nib.aff2axcodes(reference.affine)),
            "maskForegroundCodes": observed_codes,
        }
    except ResearchDataError:
        raise
    except Exception as exc:
        raise ResearchDataError("MOTUM_PREPROCESSING_FAILED") from exc


def _foreground_patch(image: np.ndarray, targets: np.ndarray, patch_size: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    spatial = image.shape[1:]
    if any(size % 16 or size < 16 for size in patch_size) or any(dim < size for dim, size in zip(spatial, patch_size)):
        raise ResearchDataError("MOTUM_PATCH_SIZE_INVALID")
    occupied = np.argwhere(targets.any(axis=0))
    center = occupied[len(occupied) // 2] if len(occupied) else np.asarray(spatial) // 2
    starts = [max(0, min(int(point) - size // 2, dim - size)) for point, size, dim in zip(center, patch_size, spatial)]
    slices = tuple(slice(start, start + size) for start, size in zip(starts, patch_size))
    return np.ascontiguousarray(image[(slice(None), *slices)]), np.ascontiguousarray(targets[(slice(None), *slices)])


def _binary_foreground(values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    values = np.asarray(values)
    if values.ndim != 3 or not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ResearchDataError("MOTUM_MASK_INVALID")
    codes = np.unique(values)
    if not np.allclose(codes, np.round(codes), atol=0, rtol=0):
        raise ResearchDataError("MOTUM_MASK_NONINTEGER_LABEL")
    return values != 0, [int(code) for code in codes]


def _motum_loss(logits: Any, targets: Any) -> Any:
    import torch

    if logits.shape != targets.shape or logits.ndim != 5 or logits.shape[1] != 2:
        raise ResearchDataError("MOTUM_MODEL_TARGET_SHAPE_MISMATCH")
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(2, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    dice_loss = 1.0 - ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()
    return 0.5 * bce + 0.5 * dice_loss


def _balanced_motum_loss(logits: Any, targets: Any) -> Any:
    """Average positive and negative BCE per region so background cannot dominate."""
    import torch

    if logits.shape != targets.shape or logits.ndim != 5 or logits.shape[1] != 2:
        raise ResearchDataError("MOTUM_MODEL_TARGET_SHAPE_MISMATCH")
    channel_losses = []
    for channel in range(2):
        prediction = logits[:, channel].reshape(-1)
        truth = targets[:, channel].reshape(-1) >= 0.5
        if not truth.any() or truth.all():
            raise ResearchDataError("MOTUM_BALANCED_PATCH_REQUIRES_BOTH_CLASSES")
        positive_loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction[truth], torch.ones_like(prediction[truth]))
        negative_loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction[~truth], torch.zeros_like(prediction[~truth]))
        channel_losses.append(0.5 * (positive_loss + negative_loss))
    bce = torch.stack(channel_losses).mean()
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(2, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    dice_loss = 1.0 - ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()
    return 0.5 * bce + 0.5 * dice_loss


def _initialize_motum_head(network: Any, foreground_prevalence: list[float]) -> list[float]:
    """Zero new output kernels and set biases to training-patch log-odds priors."""
    import torch

    if len(foreground_prevalence) != 2 or any(not 0 < value < 1 for value in foreground_prevalence):
        raise ResearchDataError("MOTUM_FOREGROUND_PRIOR_INVALID")
    head = network.conv_final[2].conv
    with torch.no_grad():
        head.weight.zero_()
        probabilities = torch.as_tensor(foreground_prevalence, dtype=head.bias.dtype, device=head.bias.device)
        head.bias.copy_(torch.logit(probabilities))
    return [float(value) for value in head.bias.detach().cpu()]


def _probability_diagnostics(logits: Any, targets: Any) -> dict[str, Any]:
    probabilities = logits.sigmoid()
    result = {"dice": _motum_dice(logits, targets), "channels": {}}
    for channel, name in ((0, "nceFlair"), (1, "ceT1c")):
        truth = targets[:, channel] >= 0.5
        values = probabilities[:, channel]
        result["channels"][name] = {
            "foregroundMeanProbability": float(values[truth].mean().detach().cpu()),
            "backgroundMeanProbability": float(values[~truth].mean().detach().cpu()),
            "foregroundPredictedPositiveFractions": {str(t): float((values[truth] >= t).float().mean().detach().cpu()) for t in (0.3, 0.5, 0.7, 0.9)},
            "backgroundPredictedPositiveFractions": {str(t): float((values[~truth] >= t).float().mean().detach().cpu()) for t in (0.3, 0.5, 0.7, 0.9)},
        }
    return result


def _motum_dice(logits: Any, targets: Any) -> dict[str, float]:
    import torch

    predictions = torch.sigmoid(logits) >= 0.5
    truth = targets >= 0.5
    dimensions = tuple(range(2, predictions.ndim))
    intersection = (predictions & truth).sum(dim=dimensions).float()
    denominator = predictions.sum(dim=dimensions).float() + truth.sum(dim=dimensions).float()
    scores = (2.0 * intersection + 1e-8) / (denominator + 1e-8)
    mean_scores = scores.mean(dim=0).detach().cpu().tolist()
    return {"nceFlair": float(mean_scores[0]), "ceT1c": float(mean_scores[1])}


def _network_from_pretrained(torch: Any, source_state: dict[str, Any], device: str) -> tuple[Any, int]:
    from monai.networks.nets import SegResNet

    network = SegResNet(
        blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), init_filters=16,
        in_channels=4, out_channels=2, dropout_prob=0.2,
    )
    target_state = network.state_dict()
    compatible = {key: value for key, value in source_state.items() if key in target_state and value.shape == target_state[key].shape}
    missing, unexpected = network.load_state_dict(compatible, strict=False)
    if unexpected or set(missing) != {"conv_final.2.conv.weight", "conv_final.2.conv.bias"}:
        raise ResearchDataError("MOTUM_PRETRAINED_WEIGHT_MAPPING_INVALID")
    for parameter in network.parameters():
        parameter.requires_grad_(True)
    return network.to(device), len(compatible)


def _load_step_one_network(torch: Any, checkpoint_path: Path, device: str) -> tuple[Any, dict[str, Any]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema") != "medvision-motum-experimental-checkpoint-v1":
            raise ResearchDataError("MOTUM_STEP_ONE_CHECKPOINT_INVALID")
        if checkpoint.get("baseCheckpointSha256") != MODEL_WEIGHT_SHA256:
            raise ResearchDataError("MOTUM_STEP_ONE_BASE_CHECKPOINT_MISMATCH")
        from monai.networks.nets import SegResNet

        network = SegResNet(
            blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), init_filters=16,
            in_channels=4, out_channels=2, dropout_prob=0.2,
        )
        network.load_state_dict(checkpoint["model"], strict=True)
        for parameter in network.parameters():
            parameter.requires_grad_(True)
        return network.to(device), checkpoint
    except ResearchDataError:
        raise
    except Exception as exc:
        raise ResearchDataError("MOTUM_STEP_ONE_CHECKPOINT_INVALID") from exc


def _atomic_torch_save(torch: Any, value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_motum_one_step(
    dataset_root: Path,
    weights_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    patch_size: tuple[int, int, int] = (64, 64, 32),
    device: str = "auto",
) -> dict[str, Any]:
    """Run one real optimization step, persist the split and an experimental checkpoint."""
    import torch
    from .inference import BrainTumorSegmenter

    dataset_root = Path(dataset_root).expanduser().resolve()
    weights_path = Path(weights_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if weights_path == output_dir or output_dir.exists() and any(output_dir.iterdir()):
        raise ResearchDataError("MOTUM_OUTPUT_DIRECTORY_NOT_EMPTY")
    if shutil.disk_usage(output_dir.parent).free < 5 * 1024**3 + 128 * 1024**2:
        raise ResearchDataError("MOTUM_MINIMUM_FREE_SPACE_NOT_AVAILABLE")
    if _sha256(weights_path) != MODEL_WEIGHT_SHA256:
        raise ResearchDataError("BASE_CHECKPOINT_HASH_MISMATCH")
    cases = _acquisition_cases(dataset_root)
    partitions = _split_cases(cases, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = {}
    for split, split_cases in partitions.items():
        path = output_dir / f"{split}-manifest.json"
        _safe_atomic_json({
            "schema": "medvision-motum-split-v1",
            "dataset": "MOTUM",
            "doi": "10.12751/g-node.tvzqc5",
            "license": "CC BY 4.0",
            "seed": seed,
            "split": split,
            "cases": split_cases,
        }, path)
        manifest_paths[split] = str(path)

    # Use the existing loader to verify the official checkpoint and obtain its exact
    # SegResNet definition, normalization transform, and architecture configuration.
    baseline = BrainTumorSegmenter(weights_path, device="cpu")
    baseline.load()
    del baseline
    import gc

    gc.collect()
    selected_device = device
    if selected_device == "auto":
        selected_device = "mps" if torch.backends.mps.is_available() else "cpu"
    if selected_device == "mps" and not torch.backends.mps.is_available():
        raise ResearchDataError("MOTUM_MPS_UNAVAILABLE")
    if selected_device not in {"cpu", "mps"}:
        raise ResearchDataError("MOTUM_DEVICE_UNSUPPORTED")

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    source_state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    network, _ = _network_from_pretrained(torch, source_state, selected_device)
    network.train()
    # Every shape-compatible parameter is initialized from the original checkpoint;
    # all are trainable for this one-step adapter sanity run. Only the final 1x1
    # two-channel convolution is newly initialized. No BraTS target channels are used.
    for parameter in network.parameters():
        parameter.requires_grad_(True)

    selected_case = partitions["train"][0]
    image, targets, geometry = _preprocess_case(selected_case)
    image, targets = _foreground_patch(image, targets, patch_size)
    input_tensor = torch.as_tensor(image[None], dtype=torch.float32, device=selected_device)
    target_tensor = torch.as_tensor(targets[None], dtype=torch.float32, device=selected_device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-5, weight_decay=1e-5)
    before = {key: value.detach().clone() for key, value in network.state_dict().items()}
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = network(input_tensor)
    if logits.shape != target_tensor.shape:
        raise ResearchDataError("MOTUM_MODEL_TARGET_SHAPE_MISMATCH")
    loss = _motum_loss(logits, target_tensor)
    if not torch.isfinite(loss):
        raise ResearchDataError("MOTUM_NONFINITE_LOSS")
    loss.backward()
    optimizer.step()
    step_seconds = time.perf_counter() - started
    changed = [key for key, value in network.state_dict().items() if not torch.equal(value.detach().cpu(), before[key].cpu())]
    if not changed:
        raise ResearchDataError("MOTUM_WEIGHTS_NOT_UPDATED")

    base_hash = _sha256(weights_path)
    config = {
        "seed": seed, "patchSizeVoxels": list(patch_size), "learningRate": 1e-5,
        "weightDecay": 1e-5, "optimizer": "Adam", "objective": "mean(BCEWithLogits + soft Dice)/2",
        "device": selected_device, "imageChannelOrder": ["t1c", "t1", "t2", "flair"],
        "targetChannelOrder": ["nce_flair", "ce_t1c"], "targetSemantics": "independent binary masks; nonzero values become foreground",
        "resampling": {"images": "linear", "masks": "nearest-neighbor", "orientation": "RAS", "spacingMm": [1, 1, 1]},
    }
    torch_checkpoint = {
        "schema": "medvision-motum-experimental-checkpoint-v1",
        "model": {key: value.detach().cpu() for key, value in network.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "baseCheckpointSha256": base_hash,
        "pretrainedLayerPolicy": {
            "pretrainedAndTrainable": "All shape-compatible SegResNet parameters, including encoder, decoder, and conv_final.0 feature projection",
            "newlyInitializedAndTrainable": "conv_final.2.conv 1x1 output convolution with two channels",
            "frozen": "None",
            "excluded": "Original three-channel output convolution; BraTS TC/WT/ET targets and losses",
        },
        "training": {**config, "loss": float(loss.detach().cpu()), "optimizerSteps": 1, "stepDurationSeconds": round(step_seconds, 3), "changedTensorCount": len(changed)},
    }
    checkpoint_path = output_dir / "motum-experimental-step1.pt"
    _atomic_torch_save(torch, torch_checkpoint, checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reload_network = SegResNet(
        blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), init_filters=16,
        in_channels=4, out_channels=2, dropout_prob=0.2,
    )
    reload_network.load_state_dict(saved["model"], strict=True)
    reload_ok = all(torch.equal(reload_network.state_dict()[key], saved["model"][key]) for key in saved["model"])
    if not reload_ok:
        raise ResearchDataError("MOTUM_CHECKPOINT_RELOAD_MISMATCH")

    metadata = {
        "schema": "medvision-motum-sanity-v1", "researchUseOnly": True,
        "dataset": {"name": "MOTUM", "doi": "10.12751/g-node.tvzqc5", "license": "CC BY 4.0", "caseCounts": SPLIT_COUNTS},
        "splits": manifest_paths,
        "model": {"baseCheckpointSha256": base_hash, "experimentalCheckpointSha256": _sha256(checkpoint_path), "experimentalCheckpointPath": str(checkpoint_path), "reloadVerified": reload_ok, **torch_checkpoint["pretrainedLayerPolicy"]},
        "sanityStep": {"loss": float(loss.detach().cpu()), "optimizerSteps": 1, "changedTensorCount": len(changed), "stepDurationSeconds": round(step_seconds, 3), "device": selected_device, "patchSizeVoxels": list(patch_size), "preprocessedGeometry": geometry, "targetOrder": config["targetChannelOrder"], "trainingSubjectCount": len(partitions["train"])},
        "software": {"python": platform.python_version(), "torch": torch.__version__, "monai": __import__("monai").__version__},
        "evaluationStatus": "untouched; no metrics computed or used for training/model selection",
        "limitations": ["One optimizer step is a pipeline sanity check, not evidence of segmentation improvement.", "MOTUM masks represent NCE and CE regions, not BraTS WT/TC/ET."],
    }
    _safe_atomic_json(metadata, output_dir / "metadata.json")
    return metadata


def _read_split(path: Path, expected_split: str, dataset_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        cases = manifest["cases"]
    except Exception as exc:
        raise ResearchDataError("MOTUM_SPLIT_MANIFEST_INVALID") from exc
    if manifest.get("schema") != "medvision-motum-split-v1" or manifest.get("split") != expected_split:
        raise ResearchDataError("MOTUM_SPLIT_MANIFEST_INVALID")
    required = set(MODALITY_SUFFIXES) | set(MASK_SUFFIXES) | {"patientId"}
    for case in cases:
        if set(case) != required:
            raise ResearchDataError("MOTUM_SPLIT_MANIFEST_INVALID")
        for key in required - {"patientId"}:
            resolved = Path(case[key]).resolve()
            if dataset_root not in resolved.parents or not resolved.is_file():
                raise ResearchDataError("MOTUM_FILE_MISSING")
    return manifest, cases


def _process_peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _mps_allocated_bytes(torch: Any) -> int | None:
    if not torch.backends.mps.is_available():
        return None
    try:
        return int(torch.mps.driver_allocated_memory())
    except Exception:
        return None


def _write_epoch_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    import csv

    path = output_dir / "metrics.csv"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fields = ["epoch", "trainingLoss", "validationLoss", "validationDiceNceFlair", "validationDiceCeT1c", "epochSeconds", "processPeakRssBytes", "mpsDriverAllocatedBytes", "freeDiskBytes"]
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_motum_three_epochs(
    dataset_root: Path,
    step_one_checkpoint: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    patch_size: tuple[int, int, int] = (64, 64, 32),
    device: str = "auto",
    maximum_runtime_seconds: int = 1200,
) -> dict[str, Any]:
    """Run a bounded 3-epoch MOTUM fine-tune using training and validation only."""
    import torch

    dataset_root = Path(dataset_root).expanduser().resolve()
    step_one_checkpoint = Path(step_one_checkpoint).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ResearchDataError("MOTUM_OUTPUT_DIRECTORY_NOT_EMPTY")
    if shutil.disk_usage(output_dir.parent).free < 5 * 1024**3 + 128 * 1024**2:
        raise ResearchDataError("MOTUM_MINIMUM_FREE_SPACE_NOT_AVAILABLE")
    step_one_hash = _sha256(step_one_checkpoint)
    step_one_meta_path = step_one_checkpoint.parent / "metadata.json"
    try:
        step_one_meta = json.loads(step_one_meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ResearchDataError("MOTUM_STEP_ONE_CHECKPOINT_INVALID") from exc
    if step_one_meta.get("model", {}).get("experimentalCheckpointSha256") != step_one_hash:
        raise ResearchDataError("MOTUM_STEP_ONE_CHECKPOINT_HASH_MISMATCH")

    train_path = step_one_checkpoint.parent / "train-manifest.json"
    validation_path = step_one_checkpoint.parent / "validation-manifest.json"
    train_manifest, train_cases = _read_split(train_path, "train", dataset_root)
    validation_manifest, validation_cases = _read_split(validation_path, "validation", dataset_root)
    if len(train_cases) != 20 or len(validation_cases) != 4:
        raise ResearchDataError("MOTUM_TRAIN_VALIDATION_COUNT_MISMATCH")
    if {case["patientId"] for case in train_cases} & {case["patientId"] for case in validation_cases}:
        raise ResearchDataError("MOTUM_PATIENT_SPLIT_LEAKAGE")

    selected_device = device
    if selected_device == "auto":
        selected_device = "mps" if torch.backends.mps.is_available() else "cpu"
    if selected_device not in {"cpu", "mps"} or selected_device == "mps" and not torch.backends.mps.is_available():
        raise ResearchDataError("MOTUM_DEVICE_UNSUPPORTED")
    torch.manual_seed(seed)
    rng = random.Random(seed)

    # Materialize only one small patch per case. The untouched evaluation manifest
    # and its image files are never opened by this path.
    preprocessed: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"train": [], "validation": []}
    geometry_summary: list[dict[str, Any]] = []
    overall_started = time.perf_counter()
    preprocessing_started = overall_started
    for split, cases in (("train", train_cases), ("validation", validation_cases)):
        for case in cases:
            image, targets, geometry = _preprocess_case(case)
            image_patch, target_patch = _foreground_patch(image, targets, patch_size)
            preprocessed[split].append((image_patch, target_patch))
            if len(geometry_summary) < 1:
                geometry_summary.append(geometry)
            del image, targets
    preprocessing_seconds = time.perf_counter() - preprocessing_started

    # Estimate from the measured step-one update and actual preprocessing. Refuse
    # to launch if the estimate plus a 25% allowance exceeds the user's 20-minute cap.
    measured_update_seconds = float(step_one_meta["sanityStep"]["stepDurationSeconds"])
    validation_forward_seconds = measured_update_seconds * 0.35
    estimated_total = preprocessing_seconds + 3 * (20 * measured_update_seconds + 4 * validation_forward_seconds)
    conservative_estimate = estimated_total * 1.25
    if conservative_estimate > maximum_runtime_seconds:
        raise ResearchDataError("MOTUM_ESTIMATED_RUNTIME_EXCEEDS_LIMIT")

    output_dir.mkdir(parents=True, exist_ok=True)
    network, step_one_state = _load_step_one_network(torch, step_one_checkpoint, selected_device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-5, weight_decay=1e-5)
    try:
        optimizer.load_state_dict(step_one_state["optimizer"])
    except Exception as exc:
        raise ResearchDataError("MOTUM_STEP_ONE_OPTIMIZER_INVALID") from exc

    training_manifest_hash = _sha256(train_path)
    validation_manifest_hash = _sha256(validation_path)
    history: list[dict[str, Any]] = []
    started = overall_started
    started_at = time.time()
    log_path = output_dir / "training.log"
    log_path.write_text(json.dumps({
        "event": "start", "startedAtUnix": started_at, "pid": os.getpid(),
        "device": selected_device, "epochs": 3, "patchSizeVoxels": list(patch_size),
        "trainCount": 20, "validationCount": 4, "evaluationCasesRead": 0,
        "estimatedRuntimeSeconds": round(conservative_estimate, 2),
        "sourceStepOneCheckpointSha256": step_one_hash,
        "trainingManifestSha256": training_manifest_hash,
        "validationManifestSha256": validation_manifest_hash,
    }, separators=(",", ":")) + "\n", encoding="utf-8")

    metadata: dict[str, Any] = {
        "schema": "medvision-motum-training-v1", "researchUseOnly": True,
        "status": "running", "dataset": {"name": "MOTUM", "doi": "10.12751/g-node.tvzqc5", "license": "CC BY 4.0"},
        "splitCounts": {"train": 20, "validation": 4, "untouchedEvaluation": 5},
        "evaluationSubjectsAccessed": 0,
        "manifestPaths": {"training": str(train_path), "validation": str(validation_path)},
        "manifestSha256": {"training": training_manifest_hash, "validation": validation_manifest_hash},
        "sourceStepOneCheckpoint": {"path": str(step_one_checkpoint), "sha256": step_one_hash},
        "configuration": {
            "epochs": 3, "seed": seed, "patchSizeVoxels": list(patch_size),
            "learningRate": optimizer.param_groups[0]["lr"], "weightDecay": 1e-5,
            "optimizer": "Adam", "loss": "0.5 * BCEWithLogits + 0.5 * soft Dice loss",
            "targetOrder": ["nce_flair", "ce_t1c"], "channelOrder": ["t1c", "t1", "t2", "flair"],
            "imageResampling": "linear", "maskResampling": "nearest-neighbor",
            "orientation": "RAS", "spacingMm": [1, 1, 1], "validationUsesTrainingPreprocessing": True,
        },
        "runtimeEstimateSeconds": round(conservative_estimate, 2),
        "preprocessingSeconds": round(preprocessing_seconds, 2),
        "preprocessedGeometrySample": geometry_summary[0] if geometry_summary else None,
        "hardware": {"device": selected_device, "python": platform.python_version(), "torch": torch.__version__, "monai": __import__("monai").__version__},
        "pretrainedLayerPolicy": step_one_state["pretrainedLayerPolicy"],
        "history": history,
    }
    _safe_atomic_json(metadata, output_dir / "metadata.json")

    for epoch in range(1, 4):
        epoch_started = time.perf_counter()
        network.train()
        order = list(range(len(train_cases)))
        rng.shuffle(order)
        train_losses: list[float] = []
        for index in order:
            image_patch, target_patch = preprocessed["train"][index]
            image_tensor = torch.as_tensor(image_patch[None], dtype=torch.float32, device=selected_device)
            target_tensor = torch.as_tensor(target_patch[None], dtype=torch.float32, device=selected_device)
            optimizer.zero_grad(set_to_none=True)
            logits = network(image_tensor)
            loss = _motum_loss(logits, target_tensor)
            if not torch.isfinite(loss):
                raise ResearchDataError("MOTUM_NONFINITE_TRAINING_LOSS")
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            del image_tensor, target_tensor, logits, loss

        network.eval()
        validation_losses: list[float] = []
        dice_by_case: list[dict[str, float]] = []
        with torch.inference_mode():
            for image_patch, target_patch in preprocessed["validation"]:
                image_tensor = torch.as_tensor(image_patch[None], dtype=torch.float32, device=selected_device)
                target_tensor = torch.as_tensor(target_patch[None], dtype=torch.float32, device=selected_device)
                logits = network(image_tensor)
                validation_loss = _motum_loss(logits, target_tensor)
                validation_losses.append(float(validation_loss.detach().cpu()))
                dice_by_case.append(_motum_dice(logits, target_tensor))
                del image_tensor, target_tensor, logits, validation_loss

        free_disk = shutil.disk_usage(output_dir.parent).free
        if free_disk < 5 * 1024**3:
            raise ResearchDataError("MOTUM_MINIMUM_FREE_SPACE_REACHED")
        row = {
            "epoch": epoch,
            "trainingLoss": float(np.mean(train_losses)),
            "validationLoss": float(np.mean(validation_losses)),
            "validationDiceNceFlair": float(np.mean([item["nceFlair"] for item in dice_by_case])),
            "validationDiceCeT1c": float(np.mean([item["ceT1c"] for item in dice_by_case])),
            "epochSeconds": round(time.perf_counter() - epoch_started, 2),
            "processPeakRssBytes": _process_peak_rss_bytes(),
            "mpsDriverAllocatedBytes": _mps_allocated_bytes(torch),
            "freeDiskBytes": free_disk,
        }
        history.append(row)
        _write_epoch_history(history, output_dir)
        metadata["history"] = history
        metadata["status"] = "running" if epoch < 3 else "completed"
        metadata["elapsedSeconds"] = round(time.perf_counter() - started, 2)
        _safe_atomic_json(metadata, output_dir / "metadata.json")
        checkpoint_payload = {
            "schema": "medvision-motum-three-epoch-checkpoint-v1",
            "model": {key: value.detach().cpu() for key, value in network.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "epoch": epoch, "history": history,
            "baseCheckpointSha256": MODEL_WEIGHT_SHA256,
            "sourceStepOneCheckpointSha256": step_one_hash,
            "trainingManifestSha256": training_manifest_hash,
            "validationManifestSha256": validation_manifest_hash,
            "configuration": metadata["configuration"],
        }
        _atomic_torch_save(torch, checkpoint_payload, output_dir / "motum-three-epoch.pt")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "epoch_complete", **row}, separators=(",", ":")) + "\n")
        elapsed = time.perf_counter() - started
        if elapsed >= maximum_runtime_seconds and epoch < 3:
            metadata["status"] = "stopped_runtime_limit"
            _safe_atomic_json(metadata, output_dir / "metadata.json")
            break

    final_path = output_dir / "motum-three-epoch.pt"
    saved = torch.load(final_path, map_location="cpu", weights_only=True)
    reload_network, _ = _load_step_one_network(torch, step_one_checkpoint, "cpu")
    if saved.get("schema") != "medvision-motum-three-epoch-checkpoint-v1":
        raise ResearchDataError("MOTUM_FINAL_CHECKPOINT_INVALID")
    reload_network.load_state_dict(saved["model"], strict=True)
    if any(not torch.equal(reload_network.state_dict()[key].cpu(), saved["model"][key]) for key in saved["model"]):
        raise ResearchDataError("MOTUM_FINAL_CHECKPOINT_RELOAD_MISMATCH")
    metadata["finalCheckpointSha256"] = _sha256(final_path)
    metadata["checkpointReloadVerified"] = True
    metadata["elapsedSeconds"] = round(time.perf_counter() - started, 2)
    metadata["status"] = "completed" if len(history) == 3 else "stopped_runtime_limit"
    _safe_atomic_json(metadata, output_dir / "metadata.json")
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "complete", "status": metadata["status"], "elapsedSeconds": metadata["elapsedSeconds"], "checkpointSha256": metadata["finalCheckpointSha256"]}, separators=(",", ":")) + "\n")
    return metadata


def run_motum_single_patch_overfit(
    dataset_root: Path,
    pretrained_checkpoint: Path,
    output_dir: Path,
    *,
    seed: int = 20261114,
    patch_size: tuple[int, int, int] = (64, 64, 32),
    device: str = "auto",
    steps: int = 60,
) -> dict[str, Any]:
    """Test whether a calibrated two-channel head can separate one real training patch."""
    import torch

    dataset_root = Path(dataset_root).expanduser().resolve()
    pretrained_checkpoint = Path(pretrained_checkpoint).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if steps < 1 or output_dir.exists() and any(output_dir.iterdir()):
        raise ResearchDataError("MOTUM_OVERFIT_CONFIGURATION_INVALID")
    if shutil.disk_usage(output_dir.parent).free < 5 * 1024**3 + 128 * 1024**2:
        raise ResearchDataError("MOTUM_MINIMUM_FREE_SPACE_NOT_AVAILABLE")
    base_hash = _sha256(pretrained_checkpoint)
    if base_hash != MODEL_WEIGHT_SHA256:
        raise ResearchDataError("BASE_CHECKPOINT_HASH_MISMATCH")
    train_path = Path.home() / "MedVision-Research" / "experiments" / "MOTUM-step1" / "train-manifest.json"
    train_manifest, train_cases = _read_split(train_path, "train", dataset_root)
    if len(train_cases) != 20:
        raise ResearchDataError("MOTUM_TRAINING_COUNT_MISMATCH")
    case = train_cases[0]
    image, masks, geometry = _preprocess_case(case)
    image, masks = _foreground_patch(image, masks, patch_size)
    prevalence = [float(masks[channel].mean()) for channel in range(2)]
    if any(value <= 0 or value >= 1 for value in prevalence):
        raise ResearchDataError("MOTUM_OVERFIT_PATCH_REQUIRES_BOTH_CLASSES")

    selected_device = device
    if selected_device == "auto":
        selected_device = "mps" if torch.backends.mps.is_available() else "cpu"
    if selected_device not in {"cpu", "mps"} or selected_device == "mps" and not torch.backends.mps.is_available():
        raise ResearchDataError("MOTUM_DEVICE_UNSUPPORTED")
    torch.manual_seed(seed)
    source_state = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=True)
    network, compatible_count = _network_from_pretrained(torch, source_state, selected_device)
    initial_bias = _initialize_motum_head(network, prevalence)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4, weight_decay=1e-5)
    image_tensor = torch.as_tensor(image[None], dtype=torch.float32, device=selected_device)
    target_tensor = torch.as_tensor(masks[None], dtype=torch.float32, device=selected_device)

    def assess() -> dict[str, Any]:
        network.eval()
        with torch.inference_mode():
            output = network(image_tensor)
            score = _probability_diagnostics(output, target_tensor)
            score["loss"] = float(_balanced_motum_loss(output, target_tensor).detach().cpu())
        network.train()
        return score

    initial = assess()
    history = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        network.train()
        optimizer.zero_grad(set_to_none=True)
        logits = network(image_tensor)
        loss = _balanced_motum_loss(logits, target_tensor)
        if not torch.isfinite(loss):
            raise ResearchDataError("MOTUM_OVERFIT_NONFINITE_LOSS")
        loss.backward()
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == steps:
            history.append({"step": step, "loss": float(loss.detach().cpu()), "elapsedSeconds": round(time.perf_counter() - started, 2)})
    final = assess()
    if shutil.disk_usage(output_dir.parent).free < 5 * 1024**3:
        raise ResearchDataError("MOTUM_MINIMUM_FREE_SPACE_REACHED")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "motum-single-patch-overfit.pt"
    checkpoint_payload = {
        "schema": "medvision-motum-single-patch-overfit-v1",
        "model": {key: value.detach().cpu() for key, value in network.state_dict().items()},
        "optimizer": optimizer.state_dict(), "epoch": 0, "optimizerSteps": steps,
        "baseCheckpointSha256": base_hash,
        "trainingManifestSha256": _sha256(train_path),
        "configuration": {"seed": seed, "patchSizeVoxels": list(patch_size), "learningRate": 1e-4, "weightDecay": 1e-5, "objective": "equal foreground/background BCE per channel plus soft Dice", "targetOrder": ["nce_flair", "ce_t1c"], "outputHeadInitialization": "zero kernel with single-training-patch foreground log-odds biases"},
    }
    _atomic_torch_save(torch, checkpoint_payload, checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    from monai.networks.nets import SegResNet

    restored = SegResNet(blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), init_filters=16, in_channels=4, out_channels=2, dropout_prob=0.2)
    restored.load_state_dict(saved["model"], strict=True)
    reload_ok = all(torch.equal(restored.state_dict()[key], saved["model"][key]) for key in saved["model"])
    if not reload_ok:
        raise ResearchDataError("MOTUM_OVERFIT_CHECKPOINT_RELOAD_FAILED")

    result = {
        "schema": "medvision-motum-single-patch-overfit-result-v1", "researchUseOnly": True,
        "status": "completed", "dataset": "MOTUM", "subjectsUsed": 1,
        "trainingManifestSha256": _sha256(train_path), "evaluationAndValidationSubjectsAccessed": 0,
        "device": selected_device, "preprocessedGeometry": geometry,
        "targetForegroundPrevalence": {"nceFlair": prevalence[0], "ceT1c": prevalence[1]},
        "compatiblePretrainedTensorCount": compatible_count,
        "newOutputHead": {"initialWeight": "zero", "initialBias": initial_bias, "trainable": True},
        "steps": steps, "initial": initial, "final": final, "history": history,
        "durationSeconds": round(time.perf_counter() - started, 2),
        "peakProcessRssBytes": _process_peak_rss_bytes(), "mpsDriverAllocatedBytes": _mps_allocated_bytes(torch),
        "checkpointPath": str(checkpoint_path), "checkpointSha256": _sha256(checkpoint_path), "checkpointReloadVerified": reload_ok,
        "limitations": ["Single-patch overfitting is a pipeline diagnostic only; it is not evidence of generalization or improved accuracy."],
    }
    _safe_atomic_json(result, output_dir / "result.json")
    _safe_atomic_json({"initial": initial, "final": final, "history": history}, output_dir / "metrics.json")
    return result
