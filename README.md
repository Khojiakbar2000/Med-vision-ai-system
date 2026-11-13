# Private DICOM Explorer

Local educational/research DICOM viewer, based on the [NextWork project](https://nextwork.ai/projects/4b262351-63f1-453e-afbf-985f9ba383b2). Not for diagnosis. An experimental local MRI segmentation workflow is developed separately from the DICOM viewer and is not clinically validated.

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

## Experimental local MRI segmentation

The optional research panel accepts four NIfTI volumes assigned by the user to T1c, T1, T2, and FLAIR. It supports co-registered, skull-stripped, axis-aligned 1 mm isotropic BraTS-like inputs on a matching voxel grid. The API losslessly reorients supported NIfTI axes to RAS for inference and restores the prediction to the source affine and voxel order. Oblique or mismatched inputs are rejected; this version does not register or resample images. The browser sends selected volumes only to the service bound to `127.0.0.1` after **Run local segmentation** is pressed. Predictions are unconfirmed research outputs, not diagnoses or treatment recommendations.

The backend uses Python 3.13 (`backend/.python-version`) and the pinned packages in `backend/requirements.txt`. Set up its repository-scoped environment and install the pinned packages with:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
```

`npm test` and `npm run test:ai` use this environment. It requires the four already co-registered, skull-stripped, 1 mm isotropic NIfTI volumes; it does not convert DICOM, register, or resample scans. Provision the official MONAI bundle weights at `~/.cache/medvision/models/brats_mri_segmentation-v0.5.4-model.pt` and verify the SHA-256 listed in [the implementation plan](docs/BRAIN_TUMOR_AI_PLAN.md). Weights are not downloaded automatically. Start both services in separate terminals with `npm run dev` and `npm run ai:serve`. Without checksum-verified weights, inference fails safely. Reference-mask comparison is local and does not establish clinical performance.

For the fixed public BraTS development sample only, local sample routes are enabled with `MEDVISION_ENABLE_PUBLIC_BRATS_SAMPLE=1`; the Vite app shows **Load public sample result** only in development at `?publicBrats=1`. Routes use explicitly configured fixed paths, defaulting outside the repository, and do not accept caller-provided paths. The verified MSD Task01 result and ground-truth metrics are documented in [the implementation plan](docs/BRAIN_TUMOR_AI_PLAN.md); no sample data or screenshots are stored in Git.

## Local medical research assistant

The research panel retrieves from a small curated corpus of attributed notes stored under `backend/medvision_ai/rag_corpus/`. It can read local PDF, Markdown, and text source documents; PDF extraction uses `pypdf` from the pinned backend requirements. Retrieval, answer assembly, and validated study-context lookup run in the loopback FastAPI process. By default the deterministic extractive answerer stays local. An optional hosted adapter requires a server-side API key and two explicit opt-ins; the UI discloses when it is enabled, never sends images or DICOM identifiers, and withholds study results unless separately enabled. Obtain approval before configuring a paid provider or allowing question/evidence transfer. Sources, architecture, test results, and current limits are documented in [the Phase 3 plan](docs/MEDICAL_RAG_ARCHITECTURE.md).

## Phase 4 research evaluation tools

The collapsed evaluation dashboard reads operator-generated result JSON locally in the browser. Multi-case evaluation, safe device selection, patient-level split checks, fine-tuning, experiment artifacts, and paired comparison commands are documented in [MODEL_TRAINING_AND_EVALUATION.md](docs/MODEL_TRAINING_AND_EVALUATION.md). Infrastructure and synthetic regression checks are in place, but there has been no Phase 4 multi-case evaluation, independent-cohort evaluation, GPU benchmark, or fine-tuning run. The single-case baseline remains the only real model evaluation and is not an independent benchmark.

See [architecture](ARCHITECTURE.md), [compatibility](DICOM_COMPATIBILITY.md), [real DICOM rendering evidence](docs/REAL_DATA_VALIDATION.md), and [executed test results and remaining work](TEST_RESULTS.md). A public de-faced brain MR series has been verified in desktop Chrome; Safari and other browsers remain unverified.

Current tutorial requirements and manual Safari acceptance: [completion plan](docs/NEXTWORK_COMPLETION_PLAN.md) and [final verification](docs/NEXTWORK_FINAL_VERIFICATION.md). Stop Vite with Control-C in its Terminal when finished; source and external medical files remain in place.
