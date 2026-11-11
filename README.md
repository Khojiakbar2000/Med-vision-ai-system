# Private DICOM Explorer

Local educational/research DICOM viewer, based on the [NextWork project](https://nextwork.ai/projects/4b262351-63f1-453e-afbf-985f9ba383b2). Not for diagnosis. AI interpretation, RAG and training are outside the current implementation.

## Run

Use Node >=24 (verified with 24.22.1), npm, and desktop Chrome with WebGL. Direct Cornerstone packages remain pinned at 5.15.2; dicom-parser at 1.8.21. The tutorial's runtime targets are recorded in `.nvmrc` and `.python-version`; they were not the verification runtimes.

```sh
npm ci
npm run dev
```

Open the printed localhost URL and select **Upload DICOM folder**. Selection stays in browser memory. The largest compatible single-frame spatial series is used. The optional `?preview=stack` URL verifies the tutorial’s temporary first-slice stage. Existing enhanced multi-frame support is preserved only behind `?enhanced=1` and is outside tutorial acceptance. After loading, the same control reloads the page to choose another folder. Maximum selection: 4096 files / 1 GiB; maximum frames per instance: 4096.

Primary drag adjusts window/level or draws a distance after choosing that tool. Wheel scrolls slices. Middle drag pans; right drag zooms. Primary drag rotates the experimental 3D view. Rendering presets affect only 3D. Measurements require positive pixel spacing and validated spatial geometry.

## Verify

```sh
npm test
npm run build
python3 -m venv /private/tmp/dicom-validation-env
/private/tmp/dicom-validation-env/bin/pip install -r requirements-audit.txt
/private/tmp/dicom-validation-env/bin/python -m unittest discover -s tests -p 'test_*.py'
```

The Python copy integration test explicitly skips when pydicom is unavailable. Run it with the environment above for complete privacy-workflow coverage.

For browser verification, start the development server on port 5180 using `npm run dev -- --port 5180`, then:

```sh
DICOM_TEST_PYTHON=/private/tmp/dicom-validation-env/bin/python npm run test:browser
DICOM_TEST_MODE=scrubbed DICOM_TEST_PYTHON=/private/tmp/dicom-validation-env/bin/python npm run test:browser
DICOM_TEST_MODE=stack DICOM_TEST_PYTHON=/private/tmp/dicom-validation-env/bin/python npm run test:browser
DICOM_TEST_MODE=multi DICOM_TEST_PYTHON=/private/tmp/dicom-validation-env/bin/python npm run test:browser
```

The harness uses installed macOS Google Chrome in an isolated temporary profile and software WebGL. It generates external temporary synthetic fixtures, explicitly selects them, asserts visible pixels in all four viewports, volume dimensions, spacing, MPR camera orientation, a drawn 20 mm measurement, gestures, preset changes, no external requests, and no synthetic patient identifier in UI/console. It writes mode-specific synthetic screenshots such as `/private/tmp/dicom-single-proof.png` and `/private/tmp/dicom-scrubbed-proof.png`. No patient files are involved.

## Separate privacy workflow

Run only in a trusted local terminal with authorized input and output folders outside this coding workspace:

```sh
python3 tools/deidentify_audit.py create-copy /external/original /external/empty-copy
python3 tools/deidentify_audit.py audit /external/original /external/empty-copy
```

The output folder must be empty. Originals are never overwritten. The tool removes private tags, name fields, common identifiers and dates, and consistently remaps instance/reference UIDs while preserving SOP classes, transfer syntax and geometry. It prints counts only. The audit compares a narrow set of common identifier fields. Neither operation certifies de-identification: pixels, uncommon fields, and clinical context may identify a person and require independent review. `PatientIdentityRemoved=YES` records the header operation, not a guarantee about pixels.

See [architecture](ARCHITECTURE.md), [compatibility](DICOM_COMPATIBILITY.md), [real DICOM rendering evidence](docs/REAL_DATA_VALIDATION.md), and [executed test results and remaining work](TEST_RESULTS.md). A public de-faced brain MR series has been verified in desktop Chrome; Safari and other browsers remain unverified.

Current tutorial requirements and manual Safari acceptance: [completion plan](docs/NEXTWORK_COMPLETION_PLAN.md) and [final verification](docs/NEXTWORK_FINAL_VERIFICATION.md). Stop Vite with Control-C in its Terminal when finished; source and external medical files remain in place.
