"""Separate, best-effort DICOM de-identification and count-only header audit.

Run this tool in a trusted local terminal, outside the coding agent workspace.
It never prints input/output paths, filenames, or identifier values.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings
import logging

IDENTIFIER_FIELDS = (
    "PatientName", "PatientID", "PatientBirthDate", "PatientAddress",
    "PatientTelephoneNumbers", "OtherPatientIDs", "InstitutionName",
    "InstitutionAddress", "ReferringPhysicianName", "AccessionNumber",
)


def _directory(path: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_dir():
        raise ValueError("A selected folder is unavailable.")
    return result


def create_copy(source: Path, output: Path) -> int:
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Input and output folders must be separate, non-nested folders.")
    from pydicom import dcmread
    from pydicom.uid import generate_uid
    workspace = Path(__file__).resolve().parents[1]
    if source == workspace or workspace in source.parents or output == workspace or workspace in output.parents:
        raise ValueError("Medical-data folders must be outside the coding workspace.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output folder must be empty.")
    output.mkdir(parents=True, exist_ok=True)
    uid_map = {}
    written = skipped = 0
    def scrub(dataset, element):
        if element.VR == "PN" or element.keyword in IDENTIFIER_FIELDS or element.VR in ("DA", "DT", "TM") or element.keyword in ("StudyDescription", "SeriesDescription", "ProtocolName", "ImageComments", "PatientComments", "StudyID", "StationName", "DeviceSerialNumber", "OperatorsName", "PerformingPhysicianName"):
            element.value = ""
        elif element.VR == "UI" and element.keyword not in ("SOPClassUID", "MediaStorageSOPClassUID", "TransferSyntaxUID", "ReferencedSOPClassUID", "CodingSchemeUID", "ImplementationClassUID"):
            values = list(element.value) if element.VM > 1 else [element.value]
            mapped = [uid_map.setdefault(str(value), generate_uid()) for value in values if value]
            element.value = mapped if element.VM > 1 else (mapped[0] if mapped else "")
    for path in source.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            dataset = dcmread(path)
            if "PixelData" not in dataset or "SOPClassUID" not in dataset:
                skipped += 1
                continue
            dataset.remove_private_tags()
            dataset.walk(scrub)
            dataset.file_meta.walk(scrub)
            dataset.PatientIdentityRemoved = "YES"
            dataset.DeidentificationMethod = "Best-effort header scrub; pixels and uncommon fields unreviewed"
            dataset.save_as(output / f"image-{written + 1:06d}.dcm", enforce_file_format=True)
            written += 1
        except Exception:
            skipped += 1
    print(f"Best-effort scrubbed images produced: {written}")
    print(f"Unreadable or unsupported objects skipped: {skipped}")
    print("Pixel text and uncommon attributes are not certified de-identified.")
    return 0 if written and not skipped else 2


def read_headers(root: Path):
    try:
        from pydicom import dcmread
    except ImportError as exc:
        raise RuntimeError("Install the pinned pydicom dependency before running the audit.") from exc
    datasets = []
    skipped = 0
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            dataset = dcmread(path, stop_before_pixels=True, force=False)
        except Exception:
            skipped += 1
            continue
        # The pinned deid recipe removes every UID, including SOP Class UID.
        # Recognize its remaining image headers by dimensions and transfer syntax.
        file_meta = getattr(dataset, "file_meta", None)
        has_sop_class = "SOPClassUID" in dataset or (file_meta and "MediaStorageSOPClassUID" in file_meta)
        has_image_geometry = all(key in dataset for key in ("Rows", "Columns", "PhotometricInterpretation", "BitsAllocated"))
        has_transfer_syntax = bool(file_meta and "TransferSyntaxUID" in file_meta)
        if has_sop_class or (has_image_geometry and has_transfer_syntax):
            datasets.append(dataset)
        else:
            skipped += 1
    return datasets, skipped


def audit(source: Path, cleaned: Path) -> int:
    if source == cleaned or source in cleaned.parents or cleaned in source.parents:
        raise ValueError("Original and scrubbed folders must be separate, non-nested folders.")
    original, original_skipped = read_headers(source)
    scrubbed, scrubbed_skipped = read_headers(cleaned)
    source_values = {
        field: {str(getattr(ds, field, "")).strip() for ds in original
                if str(getattr(ds, field, "")).strip()}
        for field in IDENTIFIER_FIELDS
    }
    reused = sum(
        any(str(getattr(ds, field, "")).strip() in source_values[field]
            for field in IDENTIFIER_FIELDS if str(getattr(ds, field, "")).strip())
        for ds in scrubbed
    )
    removed_yes = sum(str(getattr(ds, "PatientIdentityRemoved", "")).strip().upper() == "YES" for ds in scrubbed)
    print(f"Original DICOM files checked: {len(original)}")
    print(f"Scrubbed DICOM files checked: {len(scrubbed)}")
    print(f"Scrubbed files reusing checked original identifier values: {reused}")
    print(f"Scrubbed files declaring PatientIdentityRemoved=YES: {removed_yes}")
    print(f"Original unreadable/non-DICOM files skipped: {original_skipped}")
    print(f"Scrubbed unreadable/non-DICOM files skipped: {scrubbed_skipped}")
    print("Narrow comparison only; this is not a de-identification certification.")
    return 0 if original and scrubbed and reused == 0 else 1


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description="Local best-effort DICOM copy and count-only audit")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-copy")
    create.add_argument("original_folder")
    create.add_argument("output_folder")
    check = sub.add_parser("audit")
    check.add_argument("original_folder")
    check.add_argument("scrubbed_folder")
    args = parser.parse_args()
    try:
        source = _directory(args.original_folder)
        target = (Path(args.output_folder).expanduser().resolve() if args.command == "create-copy"
                  else _directory(args.scrubbed_folder))
        return create_copy(source, target) if args.command == "create-copy" else audit(source, target)
    except (ValueError, RuntimeError, OSError, ImportError):
        print("Privacy workflow stopped. Verify the selected folders and local dependencies; details are suppressed.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
