# Historical implementation plan

This initial record is superseded by ARCHITECTURE.md, DICOM_COMPATIBILITY.md and TEST_RESULTS.md. Earlier verification claims below predate the browser rendering fixes.

# Private DICOM Explorer implementation plan

## Tutorial architecture

The NextWork tutorial builds a Vite browser application with Cornerstone3D (`@cornerstonejs/core`, `@cornerstonejs/dicom-image-loader`, and `@cornerstonejs/tools`) plus `dicom-parser`. A browser folder picker supplies `File` objects directly to the local DICOM loader. The app inspects only fields needed to form a compatible single-frame series, groups by Series UID, Frame of Reference UID, orientation, rows, and columns, sorts slices, preloads image IDs, then creates one shared volume for axial, sagittal, coronal, and experimental 3D viewports. MPR interactions include wheel scrolling, window/level drag, and length measurement; 3D supports trackball rotation and Cornerstone volume presets. The tutorial's displayed summary is limited to modality and geometry.

A separate Python workflow uses `deid` to create a best-effort scrubbed copy outside the source tree and a `pydicom` audit that compares a narrow set of common identifying fields without printing values. It explicitly does not certify de-identification, burned-in pixels, filenames, private tags, or regulatory compliance.

## Implementation sequence

1. Establish repository privacy rules and document the workspace boundary. Keep original and scrubbed scans outside this repository; never use scan data in coding or tests.
2. Create a Vite app with the tutorial's pinned direct dependencies, local-only folder loading, generic status/error reporting, and responsive four-viewport layout.
3. Split DICOM candidate parsing/series selection from viewer orchestration so future MONAI/PyTorch/RAG/model services can be added behind separate interfaces without changing viewport code.
4. Add Cornerstone tool groups, rendering presets, safe geometry statistics, and a separate standalone Python de-identification/audit workflow.
5. Add automated tests that synthesize DICOM bytes in memory or in temporary test fixtures. No patient or sample scan files are checked in.
6. Document setup, privacy limitations, and browser compatibility. Build the production bundle and run the requested automated test suite.

## Privacy and safety constraints

- The app makes no network requests and has no upload, telemetry, remote-storage, or AI paths.
- DICOM selection happens only after explicit browser folder selection. The app does not print filenames, headers, identifiers, UIDs, parser errors, or loader errors.
- Scan folders and the audit's inputs/outputs must remain outside the repository. Tests use synthetic data only.
- The audit operates as a separate command-line tool, prints aggregate counts only, and is described as a narrow best-effort check.
- Educational/research use only; not for diagnosis or treatment decisions.

## Missing resources and constraints found before implementation

- The repository had no source files, manifest, lockfile, dependencies, tests, or browser support documentation.
- The tutorial dependency pins were retained. Verification used Node 24.22.1 (the project requires Node >=24; `.nvmrc` records the tutorial's Node 26.13.0 target) and Python 3.13.0 for the optional audit dependencies.
- No authorized DICOM scan or synthetic fixture existed. Automated fixtures are generated synthetically and live outside the repository when used for the browser smoke test.
- Browser support is documented based on `webkitdirectory` and WebGL; the app shell initialized in Chrome, but full volume rendering was not confirmed in the native file picker smoke test.

## Verification results

- `npm install` completed with the tutorial's exact direct package pins and created `package-lock.json`. npm reported transitive engine requirements of Node >=22.13 and >=24. Verification now uses Node 24.22.1; `.nvmrc` records the tutorial's Node 26.13.0 target and `package.json` requires Node >=24.
- `npm test`: 4 Node tests and 2 Python tests passed. The Node suite includes a synthetic two-slice MR series; all fixtures are generated and contain no patient data.
- `npm run build`: passed. Vite emitted expected Cornerstone codec Node-builtin externalization and large-chunk warnings; no compile errors.
- `python3 -m py_compile tools/deidentify_audit.py`: passed. The separate workflow was exercised against a generated synthetic 12-slice CT series: it produced 12 scrubbed headers and audited 12/12 with zero reused checked identifier values. The configured `deid` policy removes all UIDs, so its result may not load in the viewer.
- Chrome initialized the local app and showed its ready state; clicking the folder control opened the native picker. The native picker interaction did not complete a synthetic series selection, so the browser's full file-loading, MPR, and 3D rendering path remains unverified end to end. The MR test verifies parsing/grouping only, not image rendering.
- Source scan found no runtime network calls, external URLs, console logging, or PHI-bearing diagnostics in the application.
