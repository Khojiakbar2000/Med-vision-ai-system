import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).parents[1] / "tools" / "deidentify_audit.py"
spec = importlib.util.spec_from_file_location("deidentify_audit", MODULE_PATH)
audit_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_tool)


class Dataset(dict):
    def __getattr__(self, name):
        return self.get(name, "")


class AuditTests(unittest.TestCase):
    def run_audit(self, original, scrubbed):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "original"
            output = Path(directory) / "scrubbed"
            source.mkdir()
            output.mkdir()
            original_read = audit_tool.read_headers
            def fake_read(root):
                if root == source:
                    return original, 0
                return scrubbed, 0
            audit_tool.read_headers = fake_read
            capture = io.StringIO()
            try:
                with contextlib.redirect_stdout(capture):
                    result = audit_tool.audit(source, output)
            finally:
                audit_tool.read_headers = original_read
            return result, capture.getvalue()

    def test_audit_detects_reused_values_without_printing_them(self):
        secret = "SYNTHETIC_ONLY_IDENTIFIER"
        result, output = self.run_audit(
            [Dataset(SOPClassUID="1.2.3", PatientName=secret)],
            [Dataset(SOPClassUID="1.2.3", PatientName=secret)],
        )
        self.assertEqual(result, 1)
        self.assertIn("reusing checked original identifier values: 1", output)
        self.assertNotIn(secret, output)

    def test_audit_rejects_empty_folders(self):
        result, output = self.run_audit([], [])
        self.assertEqual(result, 1)
        self.assertIn('Original DICOM files checked: 0', output)

    def test_audit_accepts_nonmatching_synthetic_headers(self):
        result, output = self.run_audit(
            [Dataset(SOPClassUID="1.2.3", PatientID="SYNTHETIC_ORIGINAL")],
            [Dataset(SOPClassUID="1.2.3", PatientID="SYNTHETIC_SCRUBBED", PatientIdentityRemoved="YES")],
        )
        self.assertEqual(result, 0)
        self.assertIn("declaring PatientIdentityRemoved=YES: 1", output)


if __name__ == "__main__":
    unittest.main()

class CopyTests(unittest.TestCase):
    def test_synthetic_copy_preserves_geometry_and_class_not_identity(self):
        try:
            from pydicom import dcmread
        except ImportError:
            self.skipTest('pydicom integration dependency unavailable')
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'input'; output=Path(root)/'output'
            subprocess.run([sys.executable, str(Path(__file__).parent/'browser'/'generate_fixture.py'), str(source)], check=True, capture_output=True)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit_tool.create_copy(source,output),0)
                self.assertEqual(audit_tool.audit(source,output),0)
            original=dcmread(next(source.glob('*.dcm')))
            clean=dcmread(next(output.glob('*.dcm')))
            self.assertTrue(original.SOPClassUID == clean.SOPClassUID, "Image class must be preserved")
            self.assertTrue(original.SeriesInstanceUID != clean.SeriesInstanceUID, "Instance identity must be remapped")
            self.assertFalse(str(clean.PatientName))
            self.assertFalse(str(clean.PatientID))
            self.assertEqual(list(original.PixelSpacing),list(clean.PixelSpacing))
            self.assertEqual(len(original.PixelData),len(clean.PixelData))
