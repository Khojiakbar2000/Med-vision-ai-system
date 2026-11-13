import sys
import os
from pathlib import Path
import base64
import gzip
import io
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import nibabel as nib
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))
from medvision_ai.protocol import (  # noqa: E402
    CHANNEL_ORDER,
    InputProtocolError,
    region_masks_to_labels,
    segmentation_metrics,
    segmentation_volumes_mm3,
    validate_study,
)


class ProtocolTests(unittest.TestCase):
    def study(self, shape=(16, 16, 16)):
        affine = np.eye(4, dtype=np.float64)
        array = np.ones(shape, dtype=np.float32)
        return {role: (array.copy(), affine.copy()) for role in CHANNEL_ORDER}

    def test_accepts_only_aligned_four_channel_isotropic_study(self):
        validated = validate_study(self.study())
        self.assertEqual(validated.image.shape, (4, 16, 16, 16))
        self.assertEqual(validated.spacing_mm.tolist(), [1.0, 1.0, 1.0])

    def test_rejects_missing_sequence_and_misaligned_affine(self):
        volumes = self.study()
        volumes.pop("flair")
        with self.assertRaisesRegex(InputProtocolError, "MISSING_SEQUENCE"):
            validate_study(volumes)
        volumes = self.study()
        volumes["t2"] = (volumes["t2"][0], np.diag([1.0, 1.0, 1.0, 1.0]))
        volumes["t2"][1][0, 3] = 1.0
        with self.assertRaisesRegex(InputProtocolError, "MISMATCHED_GEOMETRY"):
            validate_study(volumes)

    def test_rejects_non_model_spacing_and_too_small_shape(self):
        volumes = self.study()
        affine = np.diag([0.8, 1.0, 1.0, 1.0])
        for role in CHANNEL_ORDER:
            volumes[role] = (volumes[role][0], affine)
        with self.assertRaisesRegex(InputProtocolError, "UNSUPPORTED_SPACING"):
            validate_study(volumes)
        volumes = {role: (np.ones((7, 16, 16), dtype=np.float32), np.eye(4)) for role in CHANNEL_ORDER}
        with self.assertRaisesRegex(InputProtocolError, "INVALID_DIMENSIONS"):
            validate_study(volumes)

    def test_rejects_non_ras_orientation(self):
        volumes = self.study()
        affine = np.diag([-1.0, 1.0, 1.0, 1.0])
        for role in CHANNEL_ORDER:
            volumes[role] = (volumes[role][0], affine)
        with self.assertRaisesRegex(InputProtocolError, "UNSUPPORTED_ORIENTATION"):
            validate_study(volumes)

    def test_api_reorients_lps_inputs_and_restores_prediction_to_source_grid(self):
        from medvision_ai.api import _restore_orientation, _to_ras

        source = np.arange(16 * 16 * 16, dtype=np.float32).reshape(16, 16, 16)
        affine = np.diag([-1.0, -1.0, 1.0, 1.0])
        ras, ras_affine = _to_ras(source, affine)
        restored = _restore_orientation(ras.astype(np.uint8), ras_affine, affine)
        self.assertEqual(ras.shape, source.shape)
        self.assertEqual(nib.aff2axcodes(ras_affine), ("R", "A", "S"))
        np.testing.assert_array_equal(restored, source.astype(np.uint8))

    def test_maps_tumor_regions_using_bundle_priority(self):
        probabilities = np.zeros((3, 8, 8, 8), dtype=np.float32)
        probabilities[1, 1, 1, 1] = 0.9  # WT only -> 2
        probabilities[0, 2, 2, 2] = 0.9  # TC -> 1
        probabilities[2, 3, 3, 3] = 0.9  # ET takes priority -> 4
        labels = region_masks_to_labels(probabilities)
        self.assertEqual([labels[1, 1, 1], labels[2, 2, 2], labels[3, 3, 3]], [2, 1, 4])

    def test_physical_volume_uses_voxel_spacing_and_nested_labels(self):
        labels = np.zeros((4, 4, 4), dtype=np.uint8)
        labels.flat[:4] = [1, 2, 4, 0]
        volumes = segmentation_volumes_mm3(labels, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(volumes["whole_tumor_mm3"], 18.0)
        self.assertEqual(volumes["tumor_core_mm3"], 12.0)
        self.assertEqual(volumes["enhancing_tumor_mm3"], 6.0)

    def test_metrics_known_overlap_empty_and_hausdorff_units(self):
        prediction = np.zeros((8, 8, 8), dtype=np.uint8)
        reference = prediction.copy()
        prediction[2, 2, 2] = reference[2, 2, 2] = 4
        prediction[3, 2, 2] = 2
        result = segmentation_metrics(prediction, reference, np.array([2.0, 1.0, 1.0]))
        self.assertEqual(result["whole_tumor"]["dice"], 2 / 3)
        self.assertEqual(result["whole_tumor"]["iou"], 0.5)
        self.assertAlmostEqual(result["whole_tumor"]["hausdorff95_mm"], 1.8)
        self.assertEqual(result["enhancing_tumor"]["dice"], 1.0)
        self.assertEqual(result["tumor_core"]["hausdorff95_mm"], 0.0)
        empty = np.zeros_like(prediction)
        self.assertEqual(segmentation_metrics(empty, empty, np.ones(3))["whole_tumor"]["dice"], 1.0)
        self.assertIsNone(segmentation_metrics(prediction, empty, np.ones(3))["whole_tumor"]["hausdorff95_mm"])

    def test_metrics_match_brats_2023_enhancing_label_three_to_model_label_four(self):
        prediction = np.zeros((8, 8, 8), dtype=np.uint8)
        reference = prediction.copy()
        prediction[2, 2, 2] = 4
        reference[2, 2, 2] = 3
        result = segmentation_metrics(prediction, reference, np.ones(3), "brats_2023")
        for values in result.values():
            self.assertEqual(values["dice"], 1.0)
            self.assertEqual(values["iou"], 1.0)
            self.assertEqual(values["hausdorff95_mm"], 0.0)

    def test_brats_2024_post_treatment_excludes_resection_cavity_from_tumor_regions(self):
        prediction = np.zeros((8, 8, 8), dtype=np.uint8)
        reference = prediction.copy()
        prediction[1, 1, 1] = 4
        reference[1, 1, 1] = 3
        reference[2, 2, 2] = 4  # Resection cavity is a separate class, not WT/TC/ET.
        result = segmentation_metrics(prediction, reference, np.ones(3), "brats_2024_gli_post_treatment")
        for values in result.values():
            self.assertEqual(values["dice"], 1.0)
            self.assertEqual(values["iou"], 1.0)
            self.assertEqual(values["hausdorff95_mm"], 0.0)

    def test_metrics_reject_unknown_reference_labels(self):
        prediction = np.zeros((8, 8, 8), dtype=np.uint8)
        reference = prediction.copy()
        reference[1, 1, 1] = 7
        with self.assertRaisesRegex(InputProtocolError, "INVALID_REFERENCE"):
            segmentation_metrics(prediction, reference, np.ones(3))

    def test_evaluation_api_compares_braTS_2023_label_three_without_mutating_reference(self):
        from medvision_ai.api import create_app

        prediction = np.zeros((8, 8, 8), dtype=np.uint8)
        reference = np.zeros_like(prediction)
        prediction[1, 1, 1], reference[1, 1, 1] = 1, 1
        prediction[2, 2, 2], reference[2, 2, 2] = 2, 2
        prediction[3, 3, 3], reference[3, 3, 3] = 4, 3
        reference_before = reference.copy()
        affine = np.diag([1.0, 1.0, 1.0, 1.0])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction_path, reference_path = root / "pred.nii.gz", root / "ref.nii.gz"
            nib.save(nib.Nifti1Image(prediction, affine), prediction_path)
            nib.save(nib.Nifti1Image(reference, affine), reference_path)
            files = {
                "prediction": ("synthetic-prediction.nii.gz", prediction_path.read_bytes(), "application/gzip"),
                "reference": ("synthetic-reference.nii.gz", reference_path.read_bytes(), "application/gzip"),
                "reference_convention": (None, "brats_2023"),
            }
            response = TestClient(create_app(type("Segmenter", (), {"network": None})())).post(
                "/api/evaluate", files=files, headers={"Origin": "http://127.0.0.1:5173"}
            )

        self.assertEqual(response.status_code, 200, response.text)
        metrics = response.json()["metrics"]
        self.assertTrue(response.json()["referenceIsUserSupplied"])
        for region in ("whole_tumor", "tumor_core", "enhancing_tumor"):
            self.assertEqual(metrics[region]["dice"], 1.0)
            self.assertEqual(metrics[region]["iou"], 1.0)
        np.testing.assert_array_equal(reference, reference_before)

    def test_msd_task01_label_order_is_evaluated_from_its_documented_convention(self):
        prediction = np.zeros((8, 8, 8), dtype=np.uint8)
        reference = np.zeros_like(prediction)
        prediction[1, 1, 1], reference[1, 1, 1] = 2, 1  # edema
        prediction[2, 2, 2], reference[2, 2, 2] = 1, 2  # non-enhancing core
        prediction[3, 3, 3], reference[3, 3, 3] = 4, 3  # enhancing tumor
        reference_before = reference.copy()

        metrics = segmentation_metrics(prediction, reference, np.ones(3), "msd_task01")

        for values in metrics.values():
            self.assertEqual(values["dice"], 1.0)
            self.assertEqual(values["iou"], 1.0)
        np.testing.assert_array_equal(reference, reference_before)

    def test_msd_task01_rejects_labels_outside_dataset_metadata(self):
        prediction = np.zeros((8, 8, 8), dtype=np.uint8)
        reference = prediction.copy()
        reference[1, 1, 1] = 4
        with self.assertRaisesRegex(InputProtocolError, "INVALID_REFERENCE"):
            segmentation_metrics(prediction, reference, np.ones(3), "msd_task01")

    def test_evaluation_api_requires_a_reference_label_convention(self):
        from medvision_ai.api import create_app

        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        affine = np.eye(4)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction_path, reference_path = root / "pred.nii.gz", root / "ref.nii.gz"
            nib.save(nib.Nifti1Image(mask, affine), prediction_path)
            nib.save(nib.Nifti1Image(mask, affine), reference_path)
            response = TestClient(create_app(type("Segmenter", (), {"network": None})())).post(
                "/api/evaluate",
                files={
                    "prediction": ("synthetic-prediction.nii.gz", prediction_path.read_bytes(), "application/gzip"),
                    "reference": ("synthetic-reference.nii.gz", reference_path.read_bytes(), "application/gzip"),
                },
            )
        self.assertEqual(response.status_code, 422)

    def test_evaluation_api_rejects_unknown_labels_and_affine_mismatch(self):
        from medvision_ai.api import create_app

        prediction = np.zeros((8, 8, 8), dtype=np.uint8)
        prediction[2, 2, 2] = 4
        invalid_reference = prediction.copy()
        invalid_reference[1, 1, 1] = 15
        affine = np.eye(4)
        client = TestClient(create_app(type("Segmenter", (), {"network": None})()))

        def post_masks(reference, reference_affine):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                prediction_path, reference_path = root / "pred.nii.gz", root / "ref.nii.gz"
                nib.save(nib.Nifti1Image(prediction, affine), prediction_path)
                nib.save(nib.Nifti1Image(reference, reference_affine), reference_path)
                return client.post(
                    "/api/evaluate",
                    files={
                        "prediction": ("synthetic-prediction.nii.gz", prediction_path.read_bytes(), "application/gzip"),
                        "reference": ("synthetic-reference.nii.gz", reference_path.read_bytes(), "application/gzip"),
                    },
                    data={"reference_convention": "brats_2018_2021"},
                    headers={"Origin": "http://127.0.0.1:5173"},
                )

        invalid_labels = post_masks(invalid_reference, affine)
        self.assertEqual(invalid_labels.status_code, 422)
        self.assertIn("incompatible", invalid_labels.json()["detail"])

        # A translated origin must be rejected even when the shapes and labels match.
        shifted_affine = affine.copy()
        shifted_affine[0, 3] = 1.0
        misaligned = post_masks(prediction, shifted_affine)
        self.assertEqual(misaligned.status_code, 422)
        self.assertIn("same spatial grid", misaligned.json()["detail"])

    def test_api_accepts_synthetic_study_and_returns_aligned_local_mask(self):
        from medvision_ai.api import create_app

        class TestSegmenter:
            network = object()
            provenance = {"name": "test-only fixture engine", "version": "fixture", "license": "test", "device": "cpu"}

            def predict(self, study):
                labels = np.zeros(study.shape, dtype=np.uint8)
                labels[2, 3, 4] = 4
                return labels, 2.5

        affine = np.eye(4)
        image_data = np.ones((16, 16, 16), dtype=np.float32)
        buffer = io.BytesIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "synthetic.nii.gz"
            nib.save(nib.Nifti1Image(image_data, affine), str(path))
            payload = path.read_bytes()
        app = create_app(TestSegmenter())
        client = TestClient(app)
        files = {role: ("SYNTHETIC_NO_PATIENT_INFO.nii.gz", payload, "application/gzip") for role in CHANNEL_ORDER}
        response = client.post("/api/segment", files=files, headers={"Origin": "http://127.0.0.1:5173"})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["shape"], [16, 16, 16])
        self.assertEqual(result["regions"]["enhancingTumor"], 1)
        self.assertEqual(result["volumes"]["enhancing_tumor_mm3"], 1.0)
        self.assertEqual(result["model"]["name"], "test-only fixture engine")
        self.assertNotIn("SYNTHETIC_NO_PATIENT_INFO", response.text)

    def test_api_accepts_lps_synthetic_study_and_returns_native_orientation_mask(self):
        from medvision_ai.api import create_app

        class TestSegmenter:
            network = object()
            provenance = {"name": "test-only fixture engine", "version": "fixture", "license": "test", "device": "cpu"}

            def predict(self, study):
                labels = np.zeros(study.shape, dtype=np.uint8)
                labels[2, 3, 4] = 4
                return labels, 2.5

        affine = np.diag([-1.0, -1.0, 1.0, 1.0])
        image_data = np.ones((16, 16, 16), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "synthetic.nii.gz"
            nib.save(nib.Nifti1Image(image_data, affine), str(path))
            payload = path.read_bytes()
        client = TestClient(create_app(TestSegmenter()))
        files = {role: ("synthetic.nii.gz", payload, "application/gzip") for role in CHANNEL_ORDER}
        response = client.post("/api/segment", files=files, headers={"Origin": "http://127.0.0.1:5173"})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        mask = nib.Nifti1Image.from_bytes(gzip.decompress(base64.b64decode(result["mask"])))
        self.assertEqual(mask.shape, image_data.shape)
        np.testing.assert_allclose(mask.affine, affine)
        self.assertEqual(nib.aff2axcodes(mask.affine), ("L", "P", "S"))
        restored = np.asarray(mask.dataobj)
        self.assertEqual(np.count_nonzero(restored), 1)
        self.assertEqual(restored[13, 12, 4], 4)

    def test_api_rejects_cross_site_posts_before_processing(self):
        from medvision_ai.api import create_app

        class TestSegmenter:
            network = None

        response = TestClient(create_app(TestSegmenter())).post(
            "/api/segment",
            headers={"Origin": "https://example.invalid"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("example.invalid", response.text)

    def test_public_sample_development_route_is_disabled_by_default(self):
        from medvision_ai.api import create_app

        class TestSegmenter:
            network = None

        with patch.dict(os.environ, {}, clear=True):
            response = TestClient(create_app(TestSegmenter())).get("/api/dev/public-brats/result")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("/private/tmp", response.text)

    def test_public_sample_route_adapts_stored_result_to_api_geometry_schema(self):
        from medvision_ai.api import create_app

        class TestSegmenter:
            network = None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            affine = np.diag([-1.0, -1.0, 1.0, 1.0])
            source_data = np.ones((16, 16, 16), dtype=np.float32)
            labels = np.zeros((16, 16, 16), dtype=np.uint8)
            labels[2, 3, 4] = 4
            nib.save(nib.Nifti1Image(source_data, affine), root / "flair.nii.gz")
            nib.save(nib.Nifti1Image(labels, affine), root / "prediction.nii.gz")
            result_path = root / "result.json"
            result_path.write_text(
                '{"model":{"name":"test-only fixture engine","version":"fixture",'
                '"license":"test","device":"cpu"},"inferenceMs":2.5,'
                '"volumesCm3":{"whole_tumor_mm3":0.001,"tumor_core_mm3":0.001,'
                '"enhancing_tumor_mm3":0.001},"limitations":["synthetic test fixture"],'
                '"referenceConvention":"msd_task01","metrics":{"whole_tumor":{"dice":1.0,"iou":1.0}},'
                '"maskShape":[999,999,999],"maskSpacingMm":[2,2,2]}'
            )
            with (
                patch.dict(os.environ, {"MEDVISION_ENABLE_PUBLIC_BRATS_SAMPLE": "1"}),
                patch("medvision_ai.api.PUBLIC_SAMPLE_DIR", root),
                patch("medvision_ai.api.PUBLIC_SAMPLE_MASK", root / "prediction.nii.gz"),
                patch("medvision_ai.api.PUBLIC_SAMPLE_RESULT", result_path),
            ):
                response = TestClient(create_app(TestSegmenter())).get("/api/dev/public-brats/result")

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["shape"], [16, 16, 16])
        self.assertEqual(result["spacingMm"], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(result["affine"], affine)
        self.assertEqual(result["volumes"]["whole_tumor_mm3"], 1.0)
        self.assertEqual(result["elapsedMs"], 2.5)
        self.assertEqual(result["referenceConvention"], "msd_task01")
        self.assertEqual(result["metrics"]["whole_tumor"]["dice"], 1.0)
        mask = nib.Nifti1Image.from_bytes(gzip.decompress(base64.b64decode(result["mask"])))
        self.assertEqual(mask.shape, tuple(result["shape"]))
        np.testing.assert_allclose(mask.affine, result["affine"])
        self.assertEqual(int(np.asarray(mask.dataobj)[2, 3, 4]), 4)

    def test_public_sample_route_rejects_prediction_on_a_different_grid(self):
        from medvision_ai.api import create_app

        class TestSegmenter:
            network = None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = nib.Nifti1Image(np.ones((16, 16, 16), dtype=np.float32), np.eye(4))
            mask = nib.Nifti1Image(np.zeros((16, 16, 16), dtype=np.uint8), np.diag([1, 1, 1, 1]))
            mask.affine[0, 3] = 1
            nib.save(source, root / "flair.nii.gz")
            nib.save(mask, root / "prediction.nii.gz")
            result_path = root / "result.json"
            result_path.write_text("{}")
            with (
                patch.dict(os.environ, {"MEDVISION_ENABLE_PUBLIC_BRATS_SAMPLE": "1"}),
                patch("medvision_ai.api.PUBLIC_SAMPLE_DIR", root),
                patch("medvision_ai.api.PUBLIC_SAMPLE_MASK", root / "prediction.nii.gz"),
                patch("medvision_ai.api.PUBLIC_SAMPLE_RESULT", result_path),
            ):
                response = TestClient(create_app(TestSegmenter())).get("/api/dev/public-brats/result")

        self.assertEqual(response.status_code, 503)
        self.assertIn("could not be read safely", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
