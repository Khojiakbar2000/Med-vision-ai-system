# NextWork final verification

Date: 2026-11-11. **Overall: NOT COMPLETE — real public anatomy is verified in Chrome; native Safari acceptance remains UNVERIFIED.**

Primary source: [Build a Private DICOM Explorer](https://nextwork.ai/projects/4b262351-63f1-453e-afbf-985f9ba383b2). All tutorial implementation sections were retrieved. The linked [completion plan](NEXTWORK_COMPLETION_PLAN.md) records exact files/functions, missing work and dependency-ordered milestones. No private scan directory was read, listed, searched or passed to a subprocess.

## Requirement-by-requirement evidence

COMPLETE below means the stated implementation and its relevant automated criterion passed. Automated fixtures are synthetic. Actual public anatomical imaging was separately verified in local Chrome as documented in [REAL_DATA_VALIDATION.md](REAL_DATA_VALIDATION.md). R23 tracks the still-unverified Safari acceptance.

| ID | Original requirement or explicitly separated extra | Status | Evidence / remaining gate |
|----|--------------------------------------------------|--------|---------------------------|
| R01 | Required: source/scan separation, restricted agent access | PARTIAL | AGENTS/CLAUDE rules and ignore safeguards inspected; user must confirm external storage and active agent permissions. No prohibited read attempted. |
| R02 | Required setup: runtime/package compatibility | PARTIAL | Compatible Node24.22.1/Python3.13.0 executed. Exact tutorial Node26.13.0/Python3.15.2 targets are not installed/verified. |
| R03 | Required: private Vite package and exact imaging pins | COMPLETE | Synthetic browser and/or unit evidence below; file/function mapping in completion plan. |
| R04 | Required: dark four-panel shell, toolbar, safe summary, research note | COMPLETE | Synthetic browser and/or unit evidence below; file/function mapping in completion plan. |
| R05 | Required: browser-controlled local folder selection | COMPLETE | Chrome filechooser selected both synthetic fixtures and the public 384-slice MR folder; native Safari picker remains unverified. |
| R06 | Required intermediate stage: first local slice STACK proof | COMPLETE | Synthetic browser and/or unit evidence below; file/function mapping in completion plan. |
| R07 | Required: local geometry parsing, no displayed identifiers | COMPLETE | Synthetic browser and/or unit evidence below; file/function mapping in completion plan. |
| R08 | Required: same series/frame-reference/orientation/dimensions grouping, largest multi-slice group | COMPLETE | Synthetic browser and/or unit evidence below; file/function mapping in completion plan. |
| R09 | Required limitation: exclude multi-frame from tutorial path | COMPLETE | Synthetic browser and/or unit evidence below; file/function mapping in completion plan. |
| R10 | Required: ordered slice IDs before volume creation | COMPLETE | Synthetic browser and/or unit evidence below; file/function mapping in completion plan. |
| R11 | Required: preload every image before shared streaming volume creation | COMPLETE | Synthetic browser and/or unit evidence below; file/function mapping in completion plan. |
| R12 | Required: axial/sagittal/coronal ORTHOGRAPHIC shared volume | COMPLETE | Synthetic assertions and public MR render show coherent anatomy in all three planes; screenshots and geometry are recorded in REAL_DATA_VALIDATION.md. |
| R13 | Required: experimental VOLUME_3D shared volume | COMPLETE | Synthetic assertions and public MR render show the volume built from decoded scan voxels; the 3D view remains experimental. |
| R14 | Required: primary MPR window/level, wheel scrolling | COMPLETE | Synthetic browser and/or unit evidence below; file/function mapping in completion plan. |
| R15 | Required: calibrated millimetre ruler and active control state | COMPLETE | Synthetic 20 mm test plus a public MR 20 mm browser annotation measured 20.37 mm. |
| R16 | Required: primary 3D rotation and modality-appropriate presets | COMPLETE | Synthetic preset/rotation assertions; public MR screenshot confirms automatic `MR-Default` selection. |
| R17 | Required: only modality, slices, matrix, spacing, thickness summary | COMPLETE | Synthetic browser and/or unit evidence below; file/function mapping in completion plan. |
| R18 | Required: safe failures, progress, reload before changing series | COMPLETE | Synthetic browser and/or unit evidence below; file/function mapping in completion plan. |
| R19 | Required: no medical uploads, analytics, remote image URLs, identifying logs | COMPLETE | Synthetic browser and/or unit evidence below; file/function mapping in completion plan. |
| R20 | Required: separate best-effort copy, pinned deid/pydicom, count-only audit | PARTIAL | deid --version reports 0.4.13; pydicom3.0.2 installed. Compatible UID-preserving copy used. Exact default deid recipe is not the verified copy path; see deviation below. |
| R21 | Required: audit fails for empty/reused identifiers; disclose limits | COMPLETE | Synthetic browser and/or unit evidence below; file/function mapping in completion plan. |
| R22 | Required: scrubbed copy still renders and controls work | MISSING | Actual synthetic copy produced by create_copy and audited successfully then passed full browser pixel/control/measurement tests. Real scrubbed copy in Safari pending manual C. |
| R23 | Required final: authorized selected study demonstrated in Safari | PARTIAL | Public de-faced MR study displayed in Chrome with MPR, 3D and calibrated ruler; Safari-specific rendering remains for manual verification. |
| O01 | Optional: Secret Mission denied agent read and browser access | UNVERIFIED | Optional challenge not attempted because AGENTS forbids accessing protected folders. User may independently inspect active permission rules; never send scan data. |
| O02 | Optional: pause/cleanup local resources | COMPLETE | Synthetic browser and/or unit evidence below; file/function mapping in completion plan. |
| X01 | Existing extension outside tutorial: enhanced multi-frame, pan/zoom | COMPLETE | Synthetic browser and/or unit evidence below; file/function mapping in completion plan. |
| X02 | Advanced outside scope: AI diagnosis/RAG/training | COMPLETE | No AI processing services introduced. Excluded from tutorial acceptance. |

## Commands actually executed in this audit run

- `/Users/aaaaa/.nvm/versions/node/v24.22.1/bin/node --test tests/dicom-series.test.js`: **5 passed**, zero failures.
- `/private/tmp/dicom-validation-env/bin/python -m unittest discover -s tests -p 'test_*.py'`: **4 passed**, zero failures/skips. Includes actual synthetic copy/audit integration, empty-input rejection, reused-value rejection without disclosure, and nonmatching-value acceptance.
- `pip install -r requirements-audit.txt` in external temporary environment: succeeded; `deid --version`: **0.4.13**; pydicom **3.0.2**.
- Node24 `node node_modules/vite/bin/vite.js build`: **passed**, with codec builtin-externalization and large-bundle warnings.
- `git diff --check -- .`: **passed** when run after implementation.
- Browser commands use `DICOM_TEST_PYTHON=/private/tmp/dicom-validation-env/bin/python` and Node24 `tests/browser/render.mjs`, against Vite on port5180. Each run launches an isolated headless Chrome with software WebGL, generates temporary synthetic files outside the repository, and deletes those files afterward.

| Browser mode | Executed result | Evidence |
|--------------|-----------------|----------|
| `DICOM_TEST_MODE=stack` | PASS | Real folder-button chooser, one native DICOM decoded, visible isolated STACK image; no sagittal canvas |
| `DICOM_TEST_MODE=single` | PASS | Four nonuniform rendered images, 64×64×16 volume, 1×1×2mm spacing, expected MPR camera normals |
| `DICOM_TEST_MODE=scrubbed` | PASS | Actual generated-copy output, successful count-only audit, same volume and full interaction assertions |
| `DICOM_TEST_MODE=reject` | PASS | Default tutorial single-frame path rejects enhanced multi-frame input; retry enabled; no page error |
| `DICOM_TEST_MODE=multi` | PASS | Preserved `?enhanced=1` extension; outside tutorial requirements |
| `DICOM_TEST_MODE=malformed` | PASS | Non-DICOM bytes safely rejected; chooser retry available; no page errors |
| `DICOM_TEST_MODE=production` on port5181 | PASS | Built production bundle decodes native synthetic data and visibly renders all four viewports; no external requests or page errors |

Full volume modes assert a drawn 20mm Length annotation within 0.5mm, window/level VOI change, middle-pan camera change, right-zoom scale change, wheel slice change, primary 3D rotation camera change, preset screenshot change, no page errors, no external requests, and no synthetic patient sentinel in UI/console. These are real Cornerstone rendering and tool assertions, not mocks.

Synthetic screenshots are outside the repository:

- `/private/tmp/dicom-stack-proof.png`
- `/private/tmp/dicom-single-proof.png`
- `/private/tmp/dicom-scrubbed-proof.png` (visually inspected during this run: MPR, 3D and a millimetre annotation visible)
- `/private/tmp/dicom-production-proof.png`
- `/private/tmp/dicom-multi-proof.png` (existing optional extension)

## Preserved limitations and compatibility deviations

- Default final path remains single-frame, largest compatible multi-slice series only. The earlier enhanced-frame extension is explicit opt-in via `?enhanced=1` and does not satisfy any tutorial requirement. `?preview=stack` is the tutorial's intermediate one-slice proof.
- Physical position/orientation ordering remains from the user's earlier requirement, rather than the tutorial's instance-number/filename tie-break. Unit tests and the publicly licensed MR series verify this path in Chrome; orientation is documented without retaining raw header values.
- Worker dependency transforms and `useLegacyMetadataProvider: true` are necessary compatibility fixes for pinned Cornerstone5.15.2 local files, retained from the verified implementation.
- The literal deid default policy can remove image-class/series UIDs needed by the viewer. The existing compatible pydicom copy remains in place, with class/transfer syntax retained and instance/reference UIDs remapped. This preserves working code and the intended separate-copy/audit behavior, but is not a claim that the literal deid command reproduces a renderable copy. The pinned deid tool remains available for the tutorial workflow in a separate user Terminal.
- Header scrub is best effort. Pixels, unusual attributes, clinical context, filenames of originals and complete DICOM confidentiality-profile compliance are not certified. `PatientIdentityRemoved=YES` records a header operation, not a guarantee about pixels.
- Native Explicit VR Little Endian is verified with synthetic MR fixtures; native Implicit VR Little Endian is verified with the public structural T1 MR series. Bundled codecs do not establish compressed-syntax verification. Tilted/sheared, duplicate-position, irregularly spaced or missing-geometry inputs are rejected. Missing slices at perfectly regular intervals cannot be detected.
- Native Safari/GPU behavior remains unverified; real anatomy is verified in Chrome. 3D is experimental. No AI diagnosis/RAG/training is implemented.
- Earlier dependency audit found 10 affected packages (4 moderate, 6 high). No incompatible forced downgrade was applied. This is a recorded security limitation, not a clean-security claim.

## Exact manual acceptance steps

### A — Privacy boundary and original study

1. Keep both original and scrubbed medical folders outside `/Users/aaaaa/Desktop/medical-vision` and outside the coding workspace. Do not copy scans into the repository. Confirm your active coding-agent access restrictions independently; do not trigger a protected read through Codex.
2. For a repeatable non-private check, use the de-faced, anonymized, PDDL-licensed series listed in [REAL_DATA_VALIDATION.md](REAL_DATA_VALIDATION.md). Do not select protected scans through Codex.
3. In Safari, open `http://127.0.0.1:5182/` if the local server is still running. Confirm the four-panel shell, disabled tools before load, and Ready status.
4. Click **Upload DICOM folder**. In Safari's native picker, select the extracted `dicoms` folder yourself and confirm. No scan data is sent to an upload service.
5. Confirm all three MPR images show the expected planes of the same study; scroll each plane and visually inspect orientation/order. Confirm the safe summary has only the five allowed geometry fields. Chrome verification already shows anatomical correspondence; this step checks Safari specifically.

### B — Real-study controls

1. With Window / Level active, drag away from any existing ruler handles and confirm contrast/brightness change.
2. Use the wheel in each MPR panel. Select **Distance**, drag two points, and confirm a distance annotation in mm. Calibration measured a 20 mm test line as 20.37 mm on the public MR series; this is not a clinical accuracy guarantee.
3. Drag the experimental 3D view to rotate it, then change presets. Report a generic 3D failure separately from MPR success if GPU/data compatibility prevents rendering.
4. In Safari's developer tools, independently confirm there are no requests carrying medical files and no identifying log values. Do not paste console output into this conversation.

### C — Separate copy and audit, then Safari reload

1. In a separate trusted Terminal, activate an environment with `requirements-audit.txt` installed. Run commands yourself; do not ask Codex to execute a command pointing at protected scan folders.
2. From the project source directory, run `python3 tools/deidentify_audit.py create-copy /absolute/external-original /absolute/external-empty-copy`, replacing placeholders locally. The output folder must be empty and non-nested. Originals are not overwritten.
3. Run `python3 tools/audit_deid.py /absolute/external-original /absolute/external-copy`. Proceed only if both readable counts are positive, reused checked values are zero, and exit status is zero. Treat this only as a narrow header audit.
4. Reload Safari, select the external scrubbed copy yourself, and repeat A/B. Do not share scans or scrubbed output on the strength of this audit alone.

### D — Optional Secret Mission / cleanup

The tutorial's denied-read challenge is optional and is not attempted under this workspace's prohibition on accessing scan folders. Review permission settings independently. Stop Vite/preview with Control-C in their Terminal when done. No source or medical files are deleted automatically.

## Smallest remaining action

User must report only these pass/fail categories: external-folder/privacy boundary confirmed; Safari original-study MPR and controls pass/fail; experimental 3D pass/fail; separate audit pass/fail; Safari scrubbed-copy MPR and controls pass/fail. Do not send filenames, headers, identifiers, medical screenshots or loader output.

Until the Safari-specific checks pass, M10 and full tutorial reproduction remain **PARTIAL / incomplete**. Real anatomical rendering has passed in Chrome; exact tutorial-runtime and raw-deid deviations above remain disclosed rather than silently counted as reproduced.

## Literal tutorial deid compatibility experiment

Executed the pinned `deid --overwrite --outfolder … identifiers --action all --input …` only against an externally generated synthetic 16-file series, capturing tool output without disclosing file values. Command exit: success. Output headers: 16. Headers missing required image-class field: 16. Headers missing required series-grouping field: 16. Thus command success cannot satisfy the tutorial’s scrubbed-copy render gate. The working compatible `create-copy` workflow is retained and passed actual scrubbed-output rendering/calibration tests; literal command compatibility remains PARTIAL rather than falsely COMPLETE.
