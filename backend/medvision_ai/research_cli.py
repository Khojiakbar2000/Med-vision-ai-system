"""Command-line entry points for offline research evaluation and fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .inference import ModelUnavailable
from .research import ResearchCheckpointSegmenter, ResearchDataError, ResilientSegmenter, evaluate_manifest, paired_comparison
from .training import TrainingConfig, fine_tune_manifest
from .motum_training import run_motum_one_step, run_motum_single_patch_overfit, run_motum_three_epochs


DEFAULT_WEIGHTS = Path.home() / ".cache" / "medvision" / "models" / "brats_mri_segmentation-v0.5.4-model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Local MedVision research evaluation tools (not for clinical use).")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate all explicitly listed cases in an offline JSON manifest.")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--weights", type=Path, default=Path(os.environ.get("MEDVISION_MODEL_PATH", DEFAULT_WEIGHTS)))
    evaluate.add_argument("--experiment-metadata", type=Path, help="Required with a fine-tuned experiment checkpoint; the manifest hash is verified.")
    evaluate.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    evaluate.add_argument("--no-cpu-fallback", action="store_true")

    train = commands.add_parser("train", help="Fine-tune the existing architecture into a new local experiment folder.")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--held-out-manifest", type=Path, required=True, help="Patient-level-disjoint final evaluation manifest; never used for training or model selection.")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--weights", type=Path, default=Path(os.environ.get("MEDVISION_MODEL_PATH", DEFAULT_WEIGHTS)))
    train.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    train.add_argument("--resume-from", type=Path)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--batch-size", type=int, default=1, help="Patch batch size; increase only when device memory permits.")
    train.add_argument("--crop-size", type=int, nargs=3, default=(224, 224, 144), metavar=("X", "Y", "Z"))
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--early-stopping-patience", type=int, default=30)
    train.add_argument("--no-mixed-precision", action="store_true", help="Disable CUDA-only automatic mixed precision.")

    motum = commands.add_parser("motum-sanity", help="Run one real-data MOTUM optimization step with two independent mask targets.")
    motum.add_argument("--dataset-root", type=Path, default=Path.home() / "MedVision-Research" / "datasets" / "MOTUM")
    motum.add_argument("--output-dir", type=Path, required=True)
    motum.add_argument("--weights", type=Path, default=Path(os.environ.get("MEDVISION_MODEL_PATH", DEFAULT_WEIGHTS)))
    motum.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    motum.add_argument("--seed", type=int, default=42)
    motum.add_argument("--patch-size", type=int, nargs=3, default=(64, 64, 32), metavar=("X", "Y", "Z"))

    motum_train = commands.add_parser("motum-train-3", help="Run a bounded three-epoch MOTUM fine-tuning experiment.")
    motum_train.add_argument("--dataset-root", type=Path, default=Path.home() / "MedVision-Research" / "datasets" / "MOTUM")
    motum_train.add_argument("--step-one-checkpoint", type=Path, default=Path.home() / "MedVision-Research" / "experiments" / "MOTUM-step1" / "motum-experimental-step1.pt")
    motum_train.add_argument("--output-dir", type=Path, required=True)
    motum_train.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    motum_train.add_argument("--seed", type=int, default=42)
    motum_train.add_argument("--patch-size", type=int, nargs=3, default=(64, 64, 32), metavar=("X", "Y", "Z"))

    motum_overfit = commands.add_parser("motum-overfit", help="Run a bounded single-patch MOTUM overfitting diagnostic.")
    motum_overfit.add_argument("--dataset-root", type=Path, default=Path.home() / "MedVision-Research" / "datasets" / "MOTUM")
    motum_overfit.add_argument("--weights", type=Path, default=Path.home() / "MedVision-Research" / "models" / "MONAI" / "brats_mri_segmentation-v0.5.4-model.pt")
    motum_overfit.add_argument("--output-dir", type=Path, required=True)
    motum_overfit.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    motum_overfit.add_argument("--seed", type=int, default=20261114)
    motum_overfit.add_argument("--steps", type=int, default=60)
    motum_overfit.add_argument("--patch-size", type=int, nargs=3, default=(64, 64, 32), metavar=("X", "Y", "Z"))

    compare = commands.add_parser("compare", help="Create paired per-case metric deltas from two saved evaluation JSON files.")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "evaluate":
            if args.experiment_metadata:
                segmenter = ResearchCheckpointSegmenter(args.weights, args.experiment_metadata, args.device, fallback_to_cpu=not args.no_cpu_fallback)
            else:
                segmenter = ResilientSegmenter(args.weights, args.device, fallback_to_cpu=not args.no_cpu_fallback)
            result = evaluate_manifest(args.manifest, args.output, segmenter, command=[sys.executable, *sys.argv])
            print(json.dumps({"summary": result["summary"], "device": result["datasetProvenance"]["runtime"], "resultsWritten": True}, indent=2))
            return
        if args.command == "train":
            config = TrainingConfig(
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                crop_size=tuple(args.crop_size),
                mixed_precision=not args.no_mixed_precision,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
                early_stopping_patience=args.early_stopping_patience,
            )
            result = fine_tune_manifest(args.manifest, args.weights, args.output_dir, config, device=args.device, resume_from=args.resume_from, held_out_manifest_path=args.held_out_manifest, command=[sys.executable, *sys.argv])
            print(json.dumps({"experimentId": result["experimentId"], "status": result["status"], "epochsCompleted": result["training"]["epochsCompleted"], "bestValidationMeanDice": result["training"]["bestValidationMeanDice"]}, indent=2))
            return
        if args.command == "motum-sanity":
            result = run_motum_one_step(args.dataset_root, args.weights, args.output_dir, seed=args.seed, patch_size=tuple(args.patch_size), device=args.device)
            print(json.dumps({"status": "completed", "loss": result["sanityStep"]["loss"], "optimizerSteps": result["sanityStep"]["optimizerSteps"], "changedTensorCount": result["sanityStep"]["changedTensorCount"], "device": result["sanityStep"]["device"], "checkpointPath": result["model"]["experimentalCheckpointPath"], "reloadVerified": result["model"]["reloadVerified"]}, indent=2))
            return
        if args.command == "motum-train-3":
            result = run_motum_three_epochs(args.dataset_root, args.step_one_checkpoint, args.output_dir, seed=args.seed, patch_size=tuple(args.patch_size), device=args.device)
            print(json.dumps({"status": result["status"], "history": result["history"], "elapsedSeconds": result.get("elapsedSeconds"), "checkpointPath": str(args.output_dir / "motum-three-epoch.pt"), "checkpointReloadVerified": result.get("checkpointReloadVerified", False)}, indent=2))
            return
        if args.command == "motum-overfit":
            result = run_motum_single_patch_overfit(args.dataset_root, args.weights, args.output_dir, seed=args.seed, patch_size=tuple(args.patch_size), device=args.device, steps=args.steps)
            print(json.dumps({"status": result["status"], "steps": result["steps"], "initialLoss": result["initial"]["loss"], "finalLoss": result["final"]["loss"], "initialDice": result["initial"]["dice"], "finalDice": result["final"]["dice"], "durationSeconds": result["durationSeconds"], "checkpointPath": result["checkpointPath"], "checkpointReloadVerified": result["checkpointReloadVerified"]}, indent=2))
            return
        if args.command == "compare":
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
            if baseline.get("schema") != "medvision-evaluation-results-v1" or candidate.get("schema") != baseline.get("schema"):
                raise ResearchDataError("INVALID_COMPARISON_INPUT")
            result = paired_comparison(baseline, candidate)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"schema": "medvision-paired-comparison-v1", "researchUseOnly": True, "baselineModel": baseline.get("datasetProvenance", {}).get("model"), "candidateModel": candidate.get("datasetProvenance", {}).get("model"), **result}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps({"pairedCaseCount": result["pairedCaseCount"], "comparisonWritten": True}, indent=2))
    except ResearchDataError as exc:
        print(json.dumps({"errorCode": exc.code}), file=sys.stderr)
        raise SystemExit(2) from None
    except ModelUnavailable as exc:
        print(json.dumps({"errorCode": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from None
    except Exception:
        # Parser/OS exceptions may contain local scan paths or identifiers.
        print(json.dumps({"errorCode": "RESEARCH_OPERATION_FAILED"}), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
