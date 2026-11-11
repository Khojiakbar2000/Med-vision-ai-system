# Browser and privacy notes

The current compatibility matrix is in [DICOM_COMPATIBILITY.md](DICOM_COMPATIBILITY.md). Executed proof and remaining gaps are in [TEST_RESULTS.md](TEST_RESULTS.md).

All medical file processing is local to browser memory after explicit selection. A Content Security Policy restricts connections to this origin; no scan uploads, analytics or persistence are implemented. Library logging is disabled before selected files are decoded. This is educational/research software, not a diagnostic device.

The separate copy/audit CLI is a best-effort header scrub. It preserves DICOM image class and geometry, remaps instance UIDs, and excludes private attributes. It does not inspect or redact pixels, certify every attribute, or establish that an output is safe to share. Keep all inputs and outputs outside the project.
