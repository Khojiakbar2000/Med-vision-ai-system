"""Opt-in, local fine-tuning for the pinned MONAI SegResNet architecture."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import resource
import shlex
import time
import uuid
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .inference import BrainTumorSegmenter
from .protocol import (
    MODEL_NAME,
    MODEL_VERSION,
    MODEL_WEIGHT_SHA256,
    InputProtocolError,
    region_masks_to_labels,
    segmentation_metrics,
)
from .research import ResearchDataError, checkpoint_sha256, load_evaluation_case, safe_provenance_text


@dataclass(frozen=True)
class TrainingConfig:
    """Defaults mirror the pinned official MONAI bundle train.json."""

    epochs: int = 300
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    seed: int = 42
    early_stopping_patience: int = 30
    crop_size: tuple[int, int, int] = (224, 224, 144)
    batch_size: int = 1
    mixed_precision: bool = True
    source_config: str = "MONAI brats_mri_segmentation train.json (SegResNet/DiceLoss/Adam/CosineAnnealingLR)"

    def validate(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("INVALID_TRAINING_CONFIG")
        if not 0 < self.validation_fraction < 1 or self.early_stopping_patience < 1:
            raise ValueError("INVALID_TRAINING_CONFIG")
        if self.batch_size < 1 or len(self.crop_size) != 3 or any(size < 16 or size % 16 for size in self.crop_size):
            raise ValueError("INVALID_TRAINING_CONFIG")


def patient_level_split(entries: list[dict[str, Any]], validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Return disjoint entry indices grouped by patient, without persisting IDs."""
    if len(entries) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("INSUFFICIENT_PATIENTS")
    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        patient_id = entry.get("patientId")
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError("MISSING_PATIENT_GROUP")
        groups.setdefault(patient_id, []).append(index)
    if len(groups) < 2:
        raise ValueError("INSUFFICIENT_PATIENTS")
    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    val_count = max(1, min(len(group_ids) - 1, round(len(group_ids) * validation_fraction)))
    validation_groups = set(group_ids[:val_count])
    train_indices = [index for patient, indices in groups.items() if patient not in validation_groups for index in indices]
    val_indices = [index for patient, indices in groups.items() if patient in validation_groups for index in indices]
    if set(train_indices) & set(val_indices):
        raise ValueError("PATIENT_SPLIT_LEAKAGE")
    return train_indices, val_indices


def training_validation_split(entries: list[dict[str, Any]], validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Honor declared train/validation partitions; reject all declared held-out data."""
    partitions = [entry.get("partition") for entry in entries]
    if not any(partition is not None for partition in partitions):
        return patient_level_split(entries, validation_fraction, seed)
    if any(partition not in {"train", "validation"} for partition in partitions):
        raise ValueError("HELD_OUT_CASE_IN_TRAINING_MANIFEST")
    patient_partitions: dict[str, set[str]] = {}
    for entry, partition in zip(entries, partitions):
        patient = entry.get("patientId")
        if not isinstance(patient, str) or not patient.strip():
            raise ValueError("MISSING_PATIENT_GROUP")
        patient_partitions.setdefault(patient, set()).add(partition)
    if any(len(values) != 1 for values in patient_partitions.values()):
        raise ValueError("PATIENT_SPLIT_LEAKAGE")
    train = [index for index, partition in enumerate(partitions) if partition == "train"]
    validation = [index for index, partition in enumerate(partitions) if partition == "validation"]
    if not train or not validation:
        raise ValueError("INSUFFICIENT_TRAIN_VALIDATION_CASES")
    return train, validation


def assert_patient_separation(training_entries: list[dict[str, Any]], held_out_entries: list[dict[str, Any]]) -> None:
    """Require stable patient group keys and prove training/evaluation disjointness."""
    def groups(entries: list[dict[str, Any]]) -> set[str]:
        values = set()
        for entry in entries:
            patient = entry.get("patientId")
            if not isinstance(patient, str) or not patient.strip():
                raise ValueError("MISSING_PATIENT_GROUP")
            values.add(patient)
        return values

    if not training_entries or not held_out_entries:
        raise ValueError("EMPTY_TRAINING_OR_HELDOUT_COHORT")
    if groups(training_entries) & groups(held_out_entries):
        raise ValueError("PATIENT_SPLIT_LEAKAGE")


def _validate_resume_identity(
    checkpoint: dict[str, Any], *, manifest_sha256: str, held_out_manifest_sha256: str,
    config_sha256: str, train_indices: list[int], validation_indices: list[int],
) -> None:
    if checkpoint.get("baseCheckpointSha256") != MODEL_WEIGHT_SHA256:
        raise ResearchDataError("RESUME_BASE_CHECKPOINT_MISMATCH")
    if checkpoint.get("trainingManifestSha256") != manifest_sha256:
        raise ResearchDataError("RESUME_MANIFEST_MISMATCH")
    if checkpoint.get("heldOutManifestSha256") != held_out_manifest_sha256:
        raise ResearchDataError("RESUME_HELDOUT_MANIFEST_MISMATCH")
    if checkpoint.get("trainingConfigSha256") != config_sha256:
        raise ResearchDataError("RESUME_CONFIG_MISMATCH")
    if checkpoint.get("trainIndices") != train_indices or checkpoint.get("validationIndices") != validation_indices:
        raise ResearchDataError("RESUME_SPLIT_MISMATCH")


def _restore_training_state(torch: Any, checkpoint: dict[str, Any], network: Any, optimizer: Any, scheduler: Any, scaler: Any, rng: np.random.Generator, device: str) -> tuple[int, float, int]:
    """Restore model/optimizer/scheduler and RNG state from an identity-checked checkpoint."""
    network.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if "numpyGeneratorState" in checkpoint:
        rng.bit_generator.state = checkpoint["numpyGeneratorState"]
    if "torchRngState" in checkpoint:
        torch.set_rng_state(checkpoint["torchRngState"])
    if device.startswith("cuda") and checkpoint.get("cudaRngStates"):
        torch.cuda.set_rng_state_all(checkpoint["cudaRngStates"])
    return int(checkpoint["epoch"]), float(checkpoint.get("bestDice", -1.0)), int(checkpoint.get("staleEpochs", 0))


def reference_to_regions(labels: np.ndarray, convention: str) -> np.ndarray:
    """Create bundle output targets [TC, WT, ET] from documented labels."""
    mapping = {
        "brats_2018_2021": ((1, 4), (1, 2, 4), (4,)),
        "brats_2023": ((1, 3), (1, 2, 3), (3,)),
        # Label 4 is resection cavity in this release and excluded from WT/TC/ET targets.
        "brats_2024_gli_post_treatment": ((1, 3), (1, 2, 3), (3,)),
        "msd_task01": ((2, 3), (1, 2, 3), (3,)),
    }
    if convention not in mapping:
        raise InputProtocolError("INVALID_REFERENCE")
    allowed = {
        "brats_2018_2021": (0, 1, 2, 4),
        "brats_2023": (0, 1, 2, 3),
        "brats_2024_gli_post_treatment": (0, 1, 2, 3, 4),
        "msd_task01": (0, 1, 2, 3),
    }[convention]
    values = np.asarray(labels)
    if values.ndim != 3 or not np.isin(values, allowed).all():
        raise InputProtocolError("INVALID_REFERENCE")
    return np.stack([np.isin(values, region) for region in mapping[convention]], axis=0).astype(np.float32)


def _augment(image: np.ndarray, label: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for axis in range(3):
        if rng.random() < 0.5:
            image = np.flip(image, axis=axis + 1)
            label = np.flip(label, axis=axis + 1)
    image = image * float(rng.uniform(0.9, 1.1)) + float(rng.uniform(-0.1, 0.1))
    return np.ascontiguousarray(image), np.ascontiguousarray(label)


def _random_crop(image: np.ndarray, label: np.ndarray, crop_size: tuple[int, int, int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    spatial = image.shape[1:]
    if any(dim < size for dim, size in zip(spatial, crop_size)):
        raise ResearchDataError("TRAINING_VOLUME_SMALLER_THAN_CROP")
    starts = [int(rng.integers(0, dim - size + 1)) for dim, size in zip(spatial, crop_size)]
    slices = tuple(slice(start, start + size) for start, size in zip(starts, crop_size))
    return image[(slice(None), *slices)], label[(slice(None), *slices)]


def _atomic_torch_save(torch: Any, value: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def fine_tune_manifest(
    manifest_path: Path,
    weights_path: Path,
    output_dir: Path,
    config: TrainingConfig = TrainingConfig(),
    *,
    device: str = "auto",
    resume_from: Path | None = None,
    held_out_manifest_path: Path | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Fine-tune into a new experiment directory; original checkpoint is read-only."""
    config.validate()
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ResearchDataError("INVALID_MANIFEST") from exc
    if manifest.get("schema") != "medvision-evaluation-v1" or not isinstance(manifest.get("cases"), list):
        raise ResearchDataError("INVALID_MANIFEST")
    training_dataset = manifest.get("dataset", {})
    if training_dataset.get("useForTraining") is not True or training_dataset.get("heldOutEvaluationExcluded") is not True:
        raise ResearchDataError("TRAINING_PROVENANCE_NOT_CONFIRMED")
    if held_out_manifest_path is None:
        raise ResearchDataError("HELDOUT_MANIFEST_REQUIRED")
    try:
        held_out_manifest = json.loads(Path(held_out_manifest_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ResearchDataError("INVALID_HELDOUT_MANIFEST") from exc
    if held_out_manifest.get("schema") != "medvision-evaluation-v1" or not isinstance(held_out_manifest.get("cases"), list):
        raise ResearchDataError("INVALID_HELDOUT_MANIFEST")
    entries = manifest["cases"]
    try:
        assert_patient_separation(entries, held_out_manifest["cases"])
    except ValueError as exc:
        raise ResearchDataError(exc.args[0]) from None
    train_indices, val_indices = training_validation_split(entries, config.validation_fraction, config.seed)
    manifest_sha256 = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    held_out_manifest_sha256 = hashlib.sha256(Path(held_out_manifest_path).read_bytes()).hexdigest()
    config_payload = asdict(config)
    config_sha256 = hashlib.sha256(json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    import torch
    from monai.losses import DiceLoss

    from .research import choose_device

    selected_device = choose_device(torch, device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if selected_device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed)

    base = BrainTumorSegmenter(Path(weights_path), device=selected_device)
    base.load()
    network = base.network
    if selected_device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    use_amp = bool(config.mixed_precision and selected_device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    start_epoch = 0
    best_dice = -1.0
    stale_epochs = 0
    if resume_from:
        try:
            checkpoint = torch.load(Path(resume_from), map_location="cpu", weights_only=True)
            _validate_resume_identity(
                checkpoint, manifest_sha256=manifest_sha256,
                held_out_manifest_sha256=held_out_manifest_sha256,
                config_sha256=config_sha256, train_indices=train_indices,
                validation_indices=val_indices,
            )
            start_epoch, best_dice, stale_epochs = _restore_training_state(
                torch, checkpoint, network, optimizer, scheduler, scaler, rng, selected_device,
            )
        except ResearchDataError:
            raise
        except Exception as exc:
            raise ResearchDataError("RESUME_CHECKPOINT_INVALID") from exc

    output = Path(output_dir).expanduser()
    if output.resolve() == Path(weights_path).expanduser().resolve():
        raise ResearchDataError("OUTPUT_CANNOT_OVERWRITE_BASE_CHECKPOINT")
    if output.exists() and any(output.iterdir()):
        if resume_from is None or Path(resume_from).expanduser().resolve() != (output / "last.pt").resolve():
            raise ResearchDataError("OUTPUT_DIRECTORY_NOT_EMPTY")
    output.mkdir(parents=True, exist_ok=True)
    started_training = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    argv = list(command or [])
    restart_argv = list(argv)
    if "--resume-from" not in restart_argv:
        restart_argv.extend(["--resume-from", str(output / "last.pt")])
    restart_command = shlex.join(restart_argv) if restart_argv else None
    epochs_completed = start_epoch
    history_path = output / "metrics.csv"
    history_exists = history_path.exists()
    log_path = output / "training.log"
    with log_path.open("a" if resume_from else "x", encoding="utf-8") as log_file:
        log_file.write(json.dumps({"event": "start", "pid": os.getpid(), "startedAt": started_at, "resume": bool(resume_from), "device": selected_device, "manifestSha256": manifest_sha256, "heldOutManifestSha256": held_out_manifest_sha256, "configSha256": config_sha256, "command": argv, "restartCommand": restart_command}, separators=(",", ":")) + "\n")
        log_file.flush()
    with history_path.open("a", encoding="utf-8", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=["epoch", "trainLoss", "validationDice", "learningRate"])
        if not history_exists:
            writer.writeheader()
        for epoch in range(start_epoch, config.epochs):
            network.train()
            train_losses = []
            epoch_order = rng.permutation(train_indices).tolist()
            for offset in range(0, len(epoch_order), config.batch_size):
                image_batch, target_batch = [], []
                for index in epoch_order[offset:offset + config.batch_size]:
                    case = load_evaluation_case(entries[index], manifest.get("dataset", {}))
                    target = reference_to_regions(case.reference, case.convention)
                    normalized = base.normalizer(case.study.image.copy())
                    image, target = _random_crop(np.asarray(normalized, dtype=np.float32), target, config.crop_size, rng)
                    image, target = _augment(image, target, rng)
                    image_batch.append(image)
                    target_batch.append(target)
                input_tensor = torch.as_tensor(np.stack(image_batch), dtype=torch.float32, device=selected_device)
                target_tensor = torch.as_tensor(np.stack(target_batch), dtype=torch.float32, device=selected_device)
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        prediction = network(input_tensor)
                        loss = loss_fn(prediction, target_tensor)
                else:
                    prediction = network(input_tensor)
                    loss = loss_fn(prediction, target_tensor)
                if not torch.isfinite(loss):
                    raise ResearchDataError("NONFINITE_TRAINING_LOSS")
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            network.eval()
            val_scores = []
            with torch.inference_mode():
                for index in val_indices:
                    case = load_evaluation_case(entries[index], manifest.get("dataset", {}))
                    target = reference_to_regions(case.reference, case.convention)
                    normalized = base.normalizer(case.study.image.copy())
                    tensor = torch.as_tensor(np.asarray(normalized, dtype=np.float32)[None], dtype=torch.float32, device=selected_device)
                    probabilities = torch.sigmoid(base.inferer(tensor, network)[0]).float().cpu().numpy()
                    labels = region_masks_to_labels(probabilities)
                    metrics = segmentation_metrics(labels, case.reference, case.study.spacing_mm, case.convention)
                    val_scores.append(float(np.mean([metrics[name]["dice"] for name in metrics])))
            validation_dice = float(np.mean(val_scores)) if val_scores else 0.0
            scheduler.step()
            writer.writerow({"epoch": epoch + 1, "trainLoss": float(np.mean(train_losses)), "validationDice": validation_dice, "learningRate": optimizer.param_groups[0]["lr"]})
            history_file.flush()
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"epoch={epoch + 1} train_loss={float(np.mean(train_losses)):.8f} validation_mean_dice={validation_dice:.8f} learning_rate={optimizer.param_groups[0]['lr']:.8g}\n")
            epochs_completed = epoch + 1
            if validation_dice > best_dice:
                best_dice, stale_epochs = validation_dice, 0
                _atomic_torch_save(torch, {"model": network.state_dict()}, output / "best.pt")
            else:
                stale_epochs += 1
            _atomic_torch_save(torch, {
                "model": network.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch + 1, "bestDice": best_dice, "staleEpochs": stale_epochs,
                "baseCheckpointSha256": MODEL_WEIGHT_SHA256, "trainingConfig": asdict(config),
                "trainingManifestSha256": manifest_sha256, "trainingConfigSha256": config_sha256,
                "heldOutManifestSha256": held_out_manifest_sha256,
                "trainIndices": train_indices, "validationIndices": val_indices,
                "numpyGeneratorState": rng.bit_generator.state, "torchRngState": torch.get_rng_state(),
                "cudaRngStates": torch.cuda.get_rng_state_all() if selected_device.startswith("cuda") else [],
            }, output / "last.pt")
            if stale_epochs >= config.early_stopping_patience:
                break

    metadata = {
        "schema": "medvision-training-experiment-v1",
        "experimentId": f"exp-{uuid.uuid4().hex[:12]}",
        "researchUseOnly": True,
        "status": "completed",
        "dataset": {
            "name": safe_provenance_text(manifest.get("dataset", {}).get("name")),
            "version": safe_provenance_text(manifest.get("dataset", {}).get("version")),
            "license": safe_provenance_text(manifest.get("dataset", {}).get("license")),
            "source": safe_provenance_text(manifest.get("dataset", {}).get("source"), url=True),
            "independenceAssessment": safe_provenance_text(manifest.get("dataset", {}).get("independenceAssessment"), "not established"),
            "trainingManifestPath": str(Path(manifest_path).expanduser().resolve()),
            "heldOutManifestPath": str(Path(held_out_manifest_path).expanduser().resolve()),
        },
        "model": {"name": MODEL_NAME, "version": MODEL_VERSION, "baseCheckpointSha256": MODEL_WEIGHT_SHA256, "bestCheckpointSha256": checkpoint_sha256(output / "best.pt")},
        "training": {**asdict(config), "seed": config.seed, "trainCaseCount": len(train_indices), "validationCaseCount": len(val_indices), "heldOutCaseCount": len(held_out_manifest["cases"]), "trainEntryOrdinals": train_indices, "validationEntryOrdinals": val_indices, "manifestSha256": manifest_sha256, "heldOutManifestSha256": held_out_manifest_sha256, "configSha256": config_sha256, "epochsCompleted": epochs_completed, "bestValidationMeanDice": best_dice, "durationSeconds": round(time.perf_counter() - started_training, 3)},
        "hardware": {
            "device": selected_device,
            "mixedPrecisionEnabled": use_amp,
            "cudaPeakAllocatedBytes": int(torch.cuda.max_memory_allocated()) if selected_device.startswith("cuda") else None,
            "processPeakRssBytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)),
        },
        "execution": {
            "pid": os.getpid(), "startedAt": started_at,
            "trainingManifestPath": str(Path(manifest_path).expanduser().resolve()),
            "heldOutManifestPath": str(Path(held_out_manifest_path).expanduser().resolve()),
            "outputDirectory": str(output.resolve()), "logPath": str(log_path.resolve()),
            "command": argv, "restartCommand": restart_command,
        },
        "software": {"python": platform.python_version(), "pytorch": torch.__version__, "monai": __import__("monai").__version__, "numpy": np.__version__},
        "provenance": {"sourceTrainingConfig": config.source_config, "originalCheckpointUnmodified": True},
    }
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps({"event": "complete", "epochsCompleted": epochs_completed, "durationSeconds": metadata["training"]["durationSeconds"], "bestValidationMeanDice": best_dice}, separators=(",", ":")) + "\n")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata
