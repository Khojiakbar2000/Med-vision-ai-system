import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))
from medvision_ai.protocol import CHANNEL_ORDER, AlignedStudy  # noqa: E402
from medvision_ai.research import (  # noqa: E402
    MANIFEST_SCHEMA,
    ResearchDataError,
    ResearchCheckpointSegmenter,
    aggregate_evaluations,
    choose_device,
    evaluate_manifest,
    load_evaluation_case,
    paired_comparison,
)
from medvision_ai.training import TrainingConfig, _augment, _random_crop, _restore_training_state, _validate_resume_identity, assert_patient_separation, fine_tune_manifest, patient_level_split, reference_to_regions, training_validation_split  # noqa: E402


class ResearchPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.affine = np.eye(4)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name, data, affine=None):
        path = self.root / name
        nib.save(nib.Nifti1Image(data, self.affine if affine is None else affine), path)
        return path

    def test_msd_channel_order_is_explicit_and_translated_to_model_roles(self):
        image = np.zeros((16, 16, 16, 4), np.float32)
        for channel, value in enumerate((1, 2, 3, 4)):
            image[..., channel] = value
        image_path = self.save("msd-image.nii.gz", image)
        reference_path = self.save("msd-label.nii.gz", np.zeros((16, 16, 16), np.uint8))
        case = load_evaluation_case(
            {"image": str(image_path), "imageChannelOrder": ["flair", "t1", "t1c", "t2"], "reference": str(reference_path)},
            {"referenceConvention": "msd_task01"},
        )
        self.assertEqual(tuple(CHANNEL_ORDER), ("t1c", "t1", "t2", "flair"))
        self.assertEqual([float(case.study.image[i, 0, 0, 0]) for i in range(4)], [3, 2, 4, 1])

    def test_label_validation_rejects_unsupported_convention_labels_without_remapping(self):
        labels = np.zeros((8, 8, 8), np.uint8)
        labels[1, 1, 1] = 4
        with self.assertRaises(Exception):
            reference_to_regions(labels, "msd_task01")
        np.testing.assert_array_equal(labels[1, 1, 1], 4)

    def test_reference_affine_mismatch_fails_before_inference(self):
        image = self.save("image.nii.gz", np.ones((16, 16, 16, 4), np.float32))
        shifted = self.affine.copy()
        shifted[0, 3] = 2
        reference = self.save("reference.nii.gz", np.zeros((16, 16, 16), np.uint8), shifted)
        with self.assertRaisesRegex(ResearchDataError, "REFERENCE_GEOMETRY_MISMATCH"):
            load_evaluation_case({"image": str(image), "imageChannelOrder": ["flair", "t1", "t1c", "t2"], "reference": str(reference), "referenceConvention": "msd_task01"}, {})

    def test_patient_split_keeps_repeated_patient_cases_together_and_is_deterministic(self):
        entries = [{"patientId": "p1"}, {"patientId": "p1"}, {"patientId": "p2"}, {"patientId": "p3"}, {"patientId": "p4"}]
        train, validation = patient_level_split(entries, 0.25, 7)
        train_patients = {entries[index]["patientId"] for index in train}
        validation_patients = {entries[index]["patientId"] for index in validation}
        self.assertFalse(train_patients & validation_patients)
        self.assertEqual((train, validation), patient_level_split(entries, 0.25, 7))

    def test_patient_split_rejects_missing_patient_keys_and_single_patient(self):
        with self.assertRaisesRegex(ValueError, "MISSING_PATIENT_GROUP"):
            patient_level_split([{"patientId": "p1"}, {}], 0.2, 1)
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_PATIENTS"):
            patient_level_split([{"patientId": "p1"}, {"patientId": "p1"}], 0.2, 1)

    def test_declared_test_split_is_rejected_and_explicit_patient_leakage_fails(self):
        with self.assertRaisesRegex(ValueError, "HELD_OUT_CASE_IN_TRAINING_MANIFEST"):
            training_validation_split([{"patientId": "p1", "partition": "train"}, {"patientId": "p2", "partition": "test"}], .2, 1)
        with self.assertRaisesRegex(ValueError, "PATIENT_SPLIT_LEAKAGE"):
            training_validation_split([{"patientId": "p1", "partition": "train"}, {"patientId": "p1", "partition": "validation"}], .2, 1)
        self.assertEqual(training_validation_split([{"patientId": "p1", "partition": "train"}, {"patientId": "p2", "partition": "validation"}], .2, 1), ([0], [1]))

    def test_final_holdout_is_patient_disjoint_and_requires_patient_group_keys(self):
        assert_patient_separation([{"patientId": "train-a"}], [{"patientId": "eval-b"}])
        with self.assertRaisesRegex(ValueError, "PATIENT_SPLIT_LEAKAGE"):
            assert_patient_separation([{"patientId": "same"}], [{"patientId": "same"}])
        with self.assertRaisesRegex(ValueError, "MISSING_PATIENT_GROUP"):
            assert_patient_separation([{"patientId": "train-a"}], [{"caseAlias": "case-1"}])

    def test_training_resume_requires_identical_data_split_and_configuration(self):
        from medvision_ai.protocol import MODEL_WEIGHT_SHA256

        identity = {
            "baseCheckpointSha256": MODEL_WEIGHT_SHA256,
            "trainingManifestSha256": "training-hash",
            "heldOutManifestSha256": "heldout-hash",
            "trainingConfigSha256": "config-hash",
            "trainIndices": [0, 2],
            "validationIndices": [1],
        }
        kwargs = {
            "manifest_sha256": identity["trainingManifestSha256"],
            "held_out_manifest_sha256": identity["heldOutManifestSha256"],
            "config_sha256": identity["trainingConfigSha256"],
        }
        _validate_resume_identity(identity, **kwargs, train_indices=[0, 2], validation_indices=[1])
        for field, expected in (("trainingManifestSha256", "RESUME_MANIFEST_MISMATCH"), ("heldOutManifestSha256", "RESUME_HELDOUT_MANIFEST_MISMATCH"), ("trainingConfigSha256", "RESUME_CONFIG_MISMATCH")):
            changed = {**identity, field: "changed"}
            with self.assertRaisesRegex(ResearchDataError, expected):
                _validate_resume_identity(changed, **kwargs, train_indices=[0, 2], validation_indices=[1])
        with self.assertRaisesRegex(ResearchDataError, "RESUME_SPLIT_MISMATCH"):
            _validate_resume_identity(identity, **kwargs, train_indices=[0], validation_indices=[1, 2])

    def test_training_config_and_label_convention_target_channels(self):
        TrainingConfig().validate()
        TrainingConfig(batch_size=2).validate()
        with self.assertRaisesRegex(ValueError, "INVALID_TRAINING_CONFIG"):
            TrainingConfig(crop_size=(225, 224, 144)).validate()
        labels = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
        target = reference_to_regions(labels, "msd_task01")
        self.assertEqual(target.shape, (3, 1, 1, 4))
        np.testing.assert_array_equal(target[:, 0, 0], [[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]])

        post_treatment = np.array([[[0, 1, 2, 3, 4]]], dtype=np.uint8)
        targets = reference_to_regions(post_treatment, "brats_2024_gli_post_treatment")
        np.testing.assert_array_equal(targets[:, 0, 0], [[0, 1, 0, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0]])

    def test_training_crop_and_augmentation_are_seeded_and_spatially_paired(self):
        image = np.stack([np.arange(8 ** 3).reshape(8, 8, 8)] * 4).astype(np.float32)
        labels = (image[:1] > 250).astype(np.float32)
        cropped_image, cropped_labels = _random_crop(image, labels, (6, 6, 6), np.random.default_rng(2))
        self.assertEqual(cropped_image.shape, (4, 6, 6, 6))
        self.assertEqual(cropped_labels.shape, (1, 6, 6, 6))
        first = _augment(image.copy(), labels.copy(), np.random.default_rng(8))
        second = _augment(image.copy(), labels.copy(), np.random.default_rng(8))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_fine_tuning_requires_explicit_training_cohort_and_holdout_exclusion(self):
        path = self.root / "training-manifest.json"
        path.write_text(json.dumps({"schema": MANIFEST_SCHEMA, "dataset": {}, "cases": []}))
        with self.assertRaisesRegex(ResearchDataError, "TRAINING_PROVENANCE_NOT_CONFIRMED"):
            fine_tune_manifest(path, self.root / "not-read.pt", self.root / "experiment")
        self.assertFalse((self.root / "experiment").exists())

    def test_multi_case_evaluation_writes_success_and_failure_and_aggregates(self):
        image = self.save("image.nii.gz", np.ones((16, 16, 16, 4), np.float32))
        label = np.zeros((16, 16, 16), np.uint8)
        label[3, 3, 3] = 3
        reference = self.save("reference.nii.gz", label)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "dataset": {"name": "synthetic fixture", "version": "test", "license": "synthetic", "referenceConvention": "msd_task01"},
            "cases": [
                {"image": str(image), "imageChannelOrder": ["flair", "t1", "t1c", "t2"], "reference": str(reference)},
                {"modalities": {}, "reference": str(reference)},
            ],
        }
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        class Segmenter:
            load_ms = 12.0
            fallback_used = False
            provenance = {"device": "cpu"}

            def load(self):
                pass

            def predict(self, study: AlignedStudy):
                prediction = np.zeros(study.shape, np.uint8)
                prediction[3, 3, 3] = 4
                return prediction, 25.0

        result = evaluate_manifest(manifest_path, self.root / "results.json", Segmenter())
        self.assertEqual(result["summary"]["caseCount"], 2)
        self.assertEqual(result["summary"]["successfulCaseCount"], 1)
        self.assertEqual(result["summary"]["failedCaseCount"], 1)
        self.assertEqual(result["cases"][0]["metrics"]["enhancing_tumor"]["dice"], 1.0)
        self.assertEqual(result["cases"][1]["status"], "failed")
        self.assertEqual(result["cases"][1]["errorCode"], "INVALID_MODALITY_MAP")
        self.assertAlmostEqual(result["cases"][0]["predictedEnhancingCm3"], 0.001)
        self.assertAlmostEqual(result["cases"][0]["referenceEnhancingCm3"], 0.001)
        self.assertTrue((self.root / "results.csv").is_file())
        self.assertEqual(json.loads((self.root / "results.json").read_text())["summary"], result["summary"])

    def test_evaluation_persists_each_case_and_resumes_without_repeating_successes(self):
        image = self.save("resume-image.nii.gz", np.ones((16, 16, 16, 4), np.float32))
        label = np.zeros((16, 16, 16), np.uint8)
        label[3, 3, 3] = 3
        reference = self.save("resume-reference.nii.gz", label)
        manifest_path = self.root / "resume-manifest.json"
        manifest_path.write_text(json.dumps({
            "schema": MANIFEST_SCHEMA,
            "dataset": {"name": "synthetic fixture", "version": "test", "license": "synthetic", "referenceConvention": "msd_task01"},
            "cases": [
                {"image": str(image), "imageChannelOrder": ["flair", "t1", "t1c", "t2"], "reference": str(reference)},
                {"image": str(image), "imageChannelOrder": ["flair", "t1", "t1c", "t2"], "reference": str(reference)},
            ],
        }))
        output_path = self.root / "resume-results.json"

        class InterruptAfterFirst:
            load_ms = 1.0
            fallback_used = False
            provenance = {"device": "cpu"}
            calls = 0

            def load(self):
                pass

            def predict(self, study):
                self.calls += 1
                if self.calls == 2:
                    raise KeyboardInterrupt()
                prediction = np.zeros(study.shape, np.uint8)
                prediction[3, 3, 3] = 4
                return prediction, 10.0

        interrupted = InterruptAfterFirst()
        with self.assertRaises(KeyboardInterrupt):
            evaluate_manifest(manifest_path, output_path, interrupted, command=["resume-test"])
        partial = json.loads(output_path.read_text())
        self.assertEqual([case["status"] for case in partial["cases"]], ["success", "pending"])
        self.assertEqual(partial["execution"]["completedCaseCount"], 1)

        class ResumeSegmenter:
            load_ms = 2.0
            fallback_used = False
            provenance = {"device": "cpu"}
            calls = 0

            def load(self):
                pass

            def predict(self, study):
                self.calls += 1
                prediction = np.zeros(study.shape, np.uint8)
                prediction[3, 3, 3] = 4
                return prediction, 12.0

        resumed = ResumeSegmenter()
        final = evaluate_manifest(manifest_path, output_path, resumed, command=["resume-test"])
        self.assertEqual(resumed.calls, 1)
        self.assertEqual(final["summary"]["successfulCaseCount"], 2)
        self.assertEqual(final["summary"]["failedCaseCount"], 0)
        self.assertEqual(final["execution"]["status"], "completed")
        self.assertTrue(final["execution"]["manifestPath"].endswith("resume-manifest.json"))
        self.assertTrue((self.root / "resume-results.csv").is_file())

    def test_metric_aggregation_counts_successes_and_comparison_uses_paired_aliases(self):
        summary = aggregate_evaluations([
            {"status": "success", "metrics": {"whole_tumor": {"dice": .5, "iou": .33}}},
            {"status": "failed"},
            {"status": "success", "metrics": {"whole_tumor": {"dice": .9, "iou": .8}}},
        ])
        self.assertEqual(summary["caseCount"], 3)
        self.assertEqual(summary["failedCaseCount"], 1)
        self.assertAlmostEqual(summary["metrics"]["whole_tumor.dice"]["mean"], .7)
        self.assertAlmostEqual(summary["metrics"]["whole_tumor.dice"]["median"], .7)
        self.assertAlmostEqual(summary["metrics"]["whole_tumor.dice"]["stddev"], 0.28284271247461906)
        one_case = aggregate_evaluations([{"status": "success", "metrics": {"whole_tumor": {"dice": 1, "iou": 1}}}])
        self.assertIsNone(one_case["metrics"]["whole_tumor.dice"]["stddev"])
        base = {"cases": [
            {"case": "case_000001", "status": "success", "inferenceMs": 20, "peakDeviceMemoryBytes": 100, "metrics": {"whole_tumor": {"dice": .7, "iou": .5}, "tumor_core": {"dice": .6, "iou": .4}, "enhancing_tumor": {"dice": .5, "iou": .3}}},
            {"case": "case_000002", "status": "failed", "errorCode": "INVALID_GEOMETRY"},
        ]}
        candidate = {"cases": [
            {"case": "case_000001", "status": "success", "inferenceMs": 15, "peakDeviceMemoryBytes": 130, "metrics": {"whole_tumor": {"dice": .8, "iou": .67}, "tumor_core": {"dice": .7, "iou": .54}, "enhancing_tumor": {"dice": .6, "iou": .43}}},
            {"case": "case_000002", "status": "success", "inferenceMs": 15, "metrics": {"whole_tumor": {"dice": .8, "iou": .67}, "tumor_core": {"dice": .7, "iou": .54}, "enhancing_tumor": {"dice": .6, "iou": .43}}},
        ]}
        compared = paired_comparison(base, candidate)
        self.assertEqual(compared["pairedCaseCount"], 1)
        self.assertAlmostEqual(compared["cases"][0]["whole_tumor.dice"], .1)
        self.assertEqual(compared["cases"][0]["inferenceMs"], -5.0)
        self.assertEqual(compared["cases"][0]["peakDeviceMemoryBytes"], 30)
        self.assertEqual(compared["failedOrUnpairedCases"][0]["case"], "case_000002")

    def test_device_selection_requires_available_accelerator_and_cpu_is_always_selectable(self):
        class Backend:
            def is_available(self):
                return False

        class Torch:
            cuda = type("Cuda", (), {"is_available": staticmethod(lambda: False)})
            backends = type("Backends", (), {"mps": Backend()})

        self.assertEqual(choose_device(Torch, "auto"), "cpu")
        self.assertEqual(choose_device(Torch, "cpu"), "cpu")
        with self.assertRaisesRegex(Exception, "DEVICE_UNAVAILABLE_MPS"):
            choose_device(Torch, "mps")
        with self.assertRaisesRegex(Exception, "DEVICE_UNAVAILABLE_CUDA"):
            choose_device(Torch, "cuda")

    def test_candidate_checkpoint_must_match_recorded_hash_and_pinned_parent(self):
        from medvision_ai.protocol import MODEL_WEIGHT_SHA256

        candidate = self.root / "candidate.pt"
        candidate.write_bytes(b"synthetic checkpoint bytes")
        metadata = self.root / "metadata.json"
        metadata.write_text(json.dumps({
            "schema": "medvision-training-experiment-v1",
            "model": {"baseCheckpointSha256": MODEL_WEIGHT_SHA256, "bestCheckpointSha256": "0" * 64},
        }))
        segmenter = ResearchCheckpointSegmenter(candidate, metadata, device="cpu")
        with self.assertRaisesRegex(ResearchDataError, "EXPERIMENT_CHECKPOINT_HASH_MISMATCH"):
            segmenter.load()

    def test_candidate_model_checkpoint_restores_architecture_and_output_channels(self):
        import hashlib
        import torch
        from monai.networks.nets import SegResNet
        from medvision_ai.protocol import MODEL_WEIGHT_SHA256

        candidate = self.root / "restorable-candidate.pt"
        model = SegResNet(blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), init_filters=16, in_channels=4, out_channels=3, dropout_prob=0.2)
        torch.save({"model": model.state_dict()}, candidate)
        candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
        metadata = self.root / "restorable-metadata.json"
        metadata.write_text(json.dumps({
            "schema": "medvision-training-experiment-v1",
            "experimentId": "synthetic-fixture",
            "model": {"baseCheckpointSha256": MODEL_WEIGHT_SHA256, "bestCheckpointSha256": candidate_hash},
        }))
        restored = ResearchCheckpointSegmenter(candidate, metadata, device="cpu")
        restored.load()
        with torch.inference_mode():
            output = restored.network(torch.zeros((1, 4, 32, 32, 32)))
        self.assertEqual(tuple(output.shape), (1, 3, 32, 32, 32))

    def test_resume_rng_state_is_weights_only_serializable(self):
        import torch

        generator = np.random.default_rng(42)
        generator.random(3)
        checkpoint_path = self.root / "rng-state.pt"
        torch.save({"numpyGeneratorState": generator.bit_generator.state, "torchRngState": torch.get_rng_state()}, checkpoint_path)
        restored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        next_value = np.random.default_rng().bit_generator
        next_value.state = restored["numpyGeneratorState"]
        self.assertEqual(int(next_value.random_raw()), int(generator.bit_generator.random_raw()))

    def test_training_resume_restores_model_optimizer_scheduler_and_random_state(self):
        import torch

        torch.manual_seed(3)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        original_parameter = model.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones((1, 2))).sum().backward()
        optimizer.step()
        scheduler.step()
        generator = np.random.default_rng(15)
        generator.random()
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": 3, "bestDice": 0.6, "staleEpochs": 1,
            "numpyGeneratorState": generator.bit_generator.state, "torchRngState": torch.get_rng_state(),
        }
        expected_parameter = model.weight.detach().clone()
        expected_next_random = int(generator.bit_generator.random_raw())
        checkpoint_path = self.root / "training-resume.pt"
        torch.save(checkpoint, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        with torch.no_grad():
            model.weight.zero_()
        resumed_rng = np.random.default_rng(999)
        epoch, best, stale = _restore_training_state(torch, checkpoint, model, optimizer, scheduler, scaler, resumed_rng, "cpu")
        self.assertFalse(torch.equal(original_parameter, expected_parameter))
        self.assertTrue(torch.equal(model.weight, expected_parameter))
        self.assertEqual(epoch, 3)
        self.assertEqual(best, 0.6)
        self.assertEqual(stale, 1)
        self.assertEqual(int(resumed_rng.bit_generator.random_raw()), expected_next_random)
        self.assertTrue(optimizer.state)


if __name__ == "__main__":
    unittest.main()
