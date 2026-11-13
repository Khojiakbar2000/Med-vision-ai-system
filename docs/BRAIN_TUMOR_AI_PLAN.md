# Brain Tumor Segmentation Plan

## Scope and safety

This is an educational and research feature, not a diagnostic medical device. It preserves the existing browser-based DICOM viewer and uses an explicit, user-initiated inference path to a service bound to `127.0.0.1`. No scan is sent to a remote service. The viewer never presents a prediction as a confirmed tumor or clinical finding. Phase 2 single-case technical acceptance is finalized below. Phase 3 adds local evidence retrieval and research explanations; it does not add clinical advice or fine-tuning.

Original and de-identified scans remain outside this repository. Automated tests use synthetic fixtures. Public research data used for a manual evaluation will be kept outside the repository and coding workspace. Patient information, UIDs, source filenames, pixels, and raw parser/model errors must not enter logs.

## Selected pretrained model

| Item | Selection |
|---|---|
| Model | MONAI Model Zoo `brats_mri_segmentation`, 3D SegResNet bundle, model metadata version 0.5.4 |
| Weights | Official MONAI Hugging Face bundle `models/model.pt`; reported size 18.8 MB; pin and verify SHA-256 `860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae` before loading |
| License | Bundle code and weights are Apache-2.0 according to the bundle. BraTS dataset terms are separate and must be checked for every evaluation source. |
| Training source | BraTS 2018, 285 cases (per the model card). The model-card benchmark is not a result of this application. |
| Input channels | Four co-registered, skull-stripped MRI volumes in the exact order T1c, T1, T2, FLAIR. A missing sequence or incompatible geometry must stop inference. |
| Output | Three sigmoid channels for tumor core (TC), whole tumor (WT), and enhancing tumor (ET). The bundle's reference discretization maps to mutually exclusive BraTS labels 0 background, 1 core, 2 edema/whole-tumor remainder, and 4 enhancing tumor, with ET taking priority. Region volume calculations must use the corresponding nested binary masks, not sum label codes. |

Primary sources: [MONAI model bundle](https://huggingface.co/MONAI/brats_mri_segmentation), [bundle inference config](https://huggingface.co/MONAI/brats_mri_segmentation/blob/main/configs/inference.json), [bundle metadata](https://github.com/Project-MONAI/model-zoo/blob/dev/models/brats_mri_segmentation/configs/metadata.json), and [bundle LICENSE](https://huggingface.co/MONAI/brats_mri_segmentation/blob/main/LICENSE). Metadata identifies bundle 0.5.4, trained with PyTorch 2.4 / MONAI 1.4 on BraTS 2018. The actual checkpoint downloaded from the official bundle matched the pinned SHA-256 and loaded successfully in the backend runtime.

## Input and preprocessing contract

- Initial supported input should be a single study with all four required sequences. Accept a folder of DICOM series or aligned NIfTI volumes only through an explicit local user action; do not infer sequence roles from private filenames or descriptions. The user assigns each sequence role in the UI.
- DICOM inputs must be single-frame, scalar MR images with valid pixel data, orientation, position, pixel spacing, and slice spacing. Group and sort slices by physical position projected onto the orientation-derived slice normal. Reject missing, duplicate, irregular, or incompatible geometry. All four channels must share a frame of reference and match in dimensions, spacing, orientation, and physical coverage, or undergo a documented and tested spatial resampling step.
- NIfTI support is useful for BraTS research data. Validate image affine, shape, finite voxels, and correspondence of the four image affines. Validate skull-stripped/co-registered model assumptions; do not silently register or skull-strip with an undocumented substitute.
- Match the bundle inference config: load T1c, T1, T2, FLAIR in that order; apply channel-wise `NormalizeIntensityd(nonzero=True)`; then use `SlidingWindowInferer` ROI `[240, 240, 160]`, batch size 1, overlap `0.5`. The published inference config applies sigmoid, thresholds at 0.5, then assigns ET first, TC second, and WT third to labels 4, 1, and 2. It specifies no orientation transform or resampling. The app accepts aligned, axis-aligned RAS or LPS 1 mm NIfTI, flips/reorders axes losslessly to RAS for inference, and restores the mask to the source shape and affine. Oblique and mismatched geometry is rejected. No interpolation or resize-based coordinate approximation is used.
- Return the segmentation with the source spatial affine and dimensions. Overlay must be aligned to the actual viewer volume in axial, sagittal, and coronal planes. Reject a result if its geometry does not match the displayed source.
- Use 1 mm isotropic spacing when required by the pinned bundle configuration; verify the exact orientation convention and transform order from the pinned config before coding. Never assume array axis order equals DICOM patient axes.

## Architecture

1. Preserve `src/core/dicom-series.js` and `src/services/local-viewer.js` for existing DICOM display.
2. Add a Python inference service under a distinct backend directory, pinned dependencies, and a loopback-only FastAPI entrypoint. The service owns model initialization, validation, preprocessing, inference, inverse mapping, postprocessing, and volume calculation.
3. Add a frontend analysis workflow that lets the user explicitly assign the four local series and start inference. Only then may the browser POST the selected image bytes to the local loopback service. Provide clear local-processing status and actionable errors. No remote URLs, analytics, telemetry, or scan content in console output.
4. Return a compact mask plus explicit shape, affine/geometry, region-volume, model provenance, and elapsed-time metadata. Keep model output ephemeral by default; do not write source scans or predictions into the repository.
5. Integrate overlays through Cornerstone3D labelmap/segmentation APIs after confirming exact API compatibility with the pinned Cornerstone version. Preserve slice navigation, window/level, zoom, pan, and distance measurement behavior.
6. Show model/version, expected protocol, predicted regions and physical volumes, inference time, research-use limitations, and failures in a dedicated analysis panel.

## Dataset selection and current access status

The selected alternative is Medical Segmentation Decathlon Task01_BrainTumour. The official MSD site states CC BY-SA 4.0 and that the labels were expert verified. MONAI's own `DecathlonDataset` loader points to the direct HTTPS archive on the MONAI-managed AWS Open Data bucket. The complete tar is 7,608,266,240 bytes, but HTTP range requests let us retrieve one paired training image and label without downloading the archive: 9,893,368 bytes for the four-channel image, 22,339 bytes for the reference, and 10,367,954 bytes total range traffic. Before sampling, 12 GiB of free disk space was confirmed; no full archive was downloaded. The sample is stored outside the repository at `/private/tmp/medvision-msd-one-case`; no archive or patient data is checked in. Only one case was selected; its public case identifier is not copied into logs or documentation under this workspace's privacy rules. Sources: [official MSD dataset page and license](https://medicaldecathlon.com/), [AWS Registry listing](https://registry.opendata.aws/msd/), and [MONAI dataset loader and official archive URL/checksum](https://github.com/Project-MONAI/MONAI/blob/dev/monai/apps/datasets.py).

The official Task01 `dataset.json` documents channel order `FLAIR`, `T1w`, `t1gd`, `T2w` and labels 0 background, 1 edema, 2 non-enhancing tumor, 3 enhancing tumor. The selected image is float32 with shape 240×240×155×4; the uint8 reference is 240×240×155. Both have the same identity affine, RAS orientation, and 1 mm isotropic spacing. Gzip integrity passed for both retrieved NIfTI members, and all range responses and tar member lengths were checked. An archive-wide MD5 was not claimed because the full tar was deliberately not downloaded. The sample is compatible with the model's 1 mm RAS grid after channel reordering only: T1c=index 2, T1=index 1, T2=index 3, FLAIR=index 0. No spatial resampling, flips, interpolation, or ground-truth label changes were applied. This is an MSD training case, and the available provenance does not establish whether it overlaps the BraTS 2018 model training cohort; results are a single-case technical evaluation, not an independent generalization estimate.

The previously attempted alternatives remain unavailable: the smaller official Synapse BraTS-SSA archive returned access denied; the TCIA BraTS-Africa download requires Aspera, whose official macOS installers failed signature validation. No Gatekeeper bypass or unverified installer execution was used, and no BraTS-Africa transfer was started. The larger BraTS-GLI archives and the 436.4 MB Zenodo upload were not downloaded. The MSD single-case HTTP-range approach avoids repeating the Aspera blocker and avoids downloading the full archive.

## Evaluation strategy

- Use one authorized, de-identified, held-out research case with four inputs and expert ground truth for manual end-to-end demonstration; ensure it is not a training example for this model if that can be established from its provenance.
- Compare predicted and reference WT, TC, and ET masks in the same physical coordinate system. Report Dice and IoU per region. Report a symmetric Hausdorff distance (preferably HD95) in millimeters with valid voxel spacing; define empty-mask behavior explicitly.
- Evaluation requires an explicit reference convention selected from the dataset documentation: BraTS 2018/2021 or model labels (0,1,2,4), BraTS 2023 (0,1,2,3), or MSD Task01 (0,1,2,3 with label 1=edema, 2=non-enhancing, 3=enhancing). The API rejects requests without a convention. WT/TC/ET are derived as binary regions using the selected documented convention. Source masks are never relabeled or mutated; unsupported labels and affine mismatch are rejected.
- The evaluation helper and API have synthetic regression coverage for all three conventions, including the distinct MSD label-1/label-2 semantics, unchanged references, invalid labels, and affine mismatch.
- A real Task01 sample passed geometry and label validation. The actual one-case WT/TC/ET Dice, IoU, and HD95 results are recorded below. The sample is from the MSD training split and the available provenance does not prove independence from the BraTS 2018 training cohort, so these scores are a technical demonstration only.
- Sources for the convention: [BraTS 2023 GLI discussion (ET label 3)](https://www.synapse.org/Synapse:syn51156910/discussion/threadId=10189) and [MONAI bundle metadata (BraTS 2018 training/output labels)](https://github.com/Project-MONAI/model-zoo/blob/dev/models/brats_mri_segmentation/configs/metadata.json).
- Confirm input decoding, channel assignment, preprocessing dimensions, inference, mask inverse mapping, overlay alignment, physical volumes, and metrics. Keep scan data outside git and do not log identifiers or image contents.
- Automated tests use synthetic images and masks only: sequence validation, normalization, spatial transforms/inversion, label mapping, physical volume, metrics including empty masks, API validation, privacy/error behavior, and inference integration with a tiny test-only network. Add a real-weight smoke test only if the model weight artifact is provisioned and the test can run locally without external data.
- Browser verification must demonstrate the overlay on actual anatomical research MRI, with screenshots that contain no identifiers. Synthetic geometry is not evidence of segmentation performance.

## Hardware and dependency compatibility

The current workspace is Apple Silicon (`arm64`). The backend virtual environment contains Python 3.13, PyTorch, MONAI, FastAPI, NiBabel, NumPy, and SciPy. The official checkpoint SHA-256 is `860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae`; inference ran on CPU (MPS and CUDA unavailable). On the selected 240×240×155 case, direct model inference took 562,206.3 ms; inference through `POST /api/segment` took 616,061.9 ms, with a 646,892.2 ms total test-client request. Both returned the same mask voxel-for-voxel. Peak memory was not measured. The repository-root `.python-version` remains 3.15.2; the AI backend uses a separate Python 3.13 runtime.

## Dependency-ordered implementation milestones

### M1 — Model, runtime, and dependency compatibility

- **Already works:** Existing Vite/Cornerstone viewer remains independent from Python.
- **Implemented:** Separate Python dependency pins; official checksum validation; SegResNet initialization. The requirements installed under Python 3.13, the checkpoint SHA-256 matched, and real checkpoint weights loaded on CPU.
- **Tasks:** Confirm a supported Python/PyTorch/MONAI combination on Apple Silicon; pin FastAPI, Uvicorn, NiBabel, pydicom, NumPy, PyTorch, and MONAI; add a model artifact cache outside git with strict checksum validation; initialize the exact bundle architecture and weights.
- **Dependencies:** Official bundle files and compatible package wheels.
- **Acceptance:** Backend starts on loopback, loads checksum-verified weights, reports only safe health/model-version metadata, and handles absent or invalid weights without leaking paths or parser details.
- **Verification:** Dependency installation and real-weight load passed. No persistent checkpoint installer/cache command or checksum-mismatch automated test exists yet; backend default expects `~/.cache/medvision/models/brats_mri_segmentation-v0.5.4-model.pt`.

### M2 — Local sequence input and validation

- **Already works:** Browser folder selection and local single-series DICOM loading.
- **Implemented:** Explicit T1c/T1/T2/FLAIR assignment for local NIfTI volumes, loopback-only request, byte limits, grid validation, temporary decoding, and safe errors. DICOM series cannot be assigned into the AI path.
- **Tasks:** Decide whether a separately validated DICOM-to-NIfTI preparation workflow is in scope; current input is NIfTI only.
- **Dependencies:** M1; existing folder input/viewer.
- **Acceptance:** No inference starts unless exactly four valid assigned channels meet the model protocol; no remote network request occurs.
- **Verification:** Synthetic API tests, browser request interception, malformed/oversized/missing-sequence cases.

### M3 — Geometry-aware preprocessing

- **Already works:** DICOM series ordering and geometry checks exist for the viewer's single volume.
- **Implemented:** Four-channel grid/affine validation, axis-aligned 1 mm acceptance, reversible LPS/RAS orientation mapping, and nonzero per-channel MONAI intensity normalization.
- **Tasks:** Add explicit synthetic normalization tests; decide whether to expand protocol beyond already-preprocessed BraTS-like NIfTI.
- **Dependencies:** M1–M2.
- **Acceptance:** Identical protocol inputs map deterministically to the expected model tensor and can be inverted to the source geometry within tested tolerance.
- **Verification:** Synthetic orientation/spacing/coordinate round-trip tests pass. Transform definitions were compared against the official inference config.

### M4 — Real inference and mask postprocessing

- **Implemented:** MONAI/PyTorch SegResNet, official sliding-window settings, sigmoid/threshold and BraTS label mapping, timing, nested region volume helper, metric helper. Real weights load successfully.
- **Verified:** The official checksum-verified 0.5.4 SegResNet ran through `BrainTumorSegmenter.predict` and through the real `POST /api/segment` route on the retrieved MSD Task01 case, using the documented channel reorder and MONAI normalization. The API returned HTTP 200; its mask matched direct inference voxel-for-voxel and preserved the source affine. Direct inference took 562,206.3 ms; API model inference took 616,061.9 ms and the whole request took 646,892.2 ms. WT volume was 83.945 cm³, TC 3.467 cm³, and ET 2.980 cm³. The API mask was then served through the fixed development sample route and compared by `/api/evaluate` with `msd_task01` selected explicitly.
- **Remaining from Phase 2:** Peak-memory measurement. The subsequent three-case evaluation is documented in the Phase 4 section below; it remains a training-split result and does not establish held-out generalization or clinical performance. TCIA Aspera was not used because its installers failed signature verification; no bypass was attempted.
- **Dependencies:** M1–M3 and checksum-verified weights.
- **Acceptance:** Real pretrained weights produce finite masks on a compatible four-sequence MRI; no fallback or synthetic prediction is possible.
- **Verification:** Real model smoke test; threshold/label mapping, physical-volume, and failure tests.

### M5 — Viewer overlay and 3D segmentation display

- **Already works:** Axial/sagittal/coronal and experimental volume viewports render the base DICOM volume.
- **Implemented:** Labelmap representations on axial, sagittal, and coronal viewports, plus a color-coded 3D label volume actor built from the same model mask. The MRI actor remains visible while the tumor mask is rendered. Visibility and opacity controls apply to 2D and 3D.
- **Root cause found and fixed:** The active Cornerstone `VolumeViewport3D` inherits `BaseVolumeViewport.addVolumes()`, which sets an actor invisible when its `visibility` option is omitted. The research mask volume input omitted it, so the 3D actor was hidden. The mask input now explicitly sets `visibility: true`; a focused Node regression test covers this option and nearest-neighbor label rendering.
- **Verified:** Headless local Chrome loaded the real MSD MRI and MONAI mask through the development-only sample route. MRI and mask matched exactly at 240×240×155, 1 mm spacing, origin `[0,0,0]`, and direction `[-1,0,0,0,-1,0,0,0,1]`; the 3D tumor actor was visible. The real predicted region was made visible in axial, sagittal, and coronal slices by navigating to tumor-bearing slices. Wheel navigation changed all three MPR slice positions; opacity changed to 15% and back to 65%. No page errors were observed. Screenshots were saved outside the repository.
- **Evidence:** `/private/tmp/medvision-msd-real-all-mpr-3d.png` shows all three MPR overlays and the same tumor in 3D; `/private/tmp/medvision-msd-real-3d.png` is a close 3D view. Plane-specific screenshots are `/private/tmp/medvision-msd-axial-overlay.png`, `/private/tmp/medvision-msd-sagittal-overlay.png`, and `/private/tmp/medvision-msd-coronal-overlay.png`. These show the public research sample only. The combined screenshot displays the direct backend run's 562.2 s; the later `/api/segment` mask was proven identical voxel-for-voxel, with 616.1 s model inference time.
- **Remaining:** The rendering check is manual browser validation against one research case. Automated DICOM browser tests remain synthetic-only.
- **Dependencies:** M4; Cornerstone API compatibility check.
- **Acceptance:** Overlay slices align in all planes; 3D shows the predicted segmentation derived from the real mask; empty outputs show no invented regions.
- **Verification:** Browser assertions and identifier-free screenshots using authorized research MRI.

### M6 — Research analysis panel

- **Implemented in source:** Model/protocol/results/volumes/timing/limitations panel and safe local errors.
- **Verified:** Manual local browser check with the real model output showed the results panel, overlays, and 3D mask. Automated browser tests remain synthetic-fixture-only.
- **Dependencies:** M4–M5.
- **Acceptance:** All requested fields are present and never claim confirmation or clinical accuracy.
- **Verification:** Browser assertions for success, missing input, unsupported input, and empty-mask states.

### M7 — Ground-truth evaluation and test coverage

- **Implemented:** Explicit reference conventions in the backend evaluator and UI; MSD Task01 mapping is WT={1,2,3}, TC={2,3}, ET={3}, compared to model labels WT={1,2,4}, TC={1,4}, ET={4}. The original reference labelmap is not rewritten.
- **Verified:** The real reference has uint8 labels exactly 0,1,2,3; spatial shape and affine match the model prediction; 1 mm isotropic RAS geometry. `/api/evaluate` returned 200 for `msd_task01`. Real single-case scores: WT Dice 0.896637, IoU 0.812641, HD95 4.358899 mm; TC Dice 0.834522, IoU 0.716034, HD95 2.000000 mm; ET Dice 0.831881, IoU 0.712155, HD95 2.449490 mm. Reference data remained unchanged.
- **Regression coverage at Phase 2 validation:** 25 Python tests and seven Node tests passed for inference, label conventions, geometry, evaluation, and volume helpers. After the Phase 3 addition, the full current suite is 40 Python tests and eight Node tests; the synthetic browser suite remains a separate run using `.venv/bin/python` for its declared `pydicom` dependency.
- **Limitations:** This is one case from the Task01 training split. Dataset provenance does not establish independence from the BraTS 2018 training cohort; no population-level or clinical accuracy claim is made.
- **Dependencies:** M1–M6.
- **Acceptance:** One valid research case has real inference, correctly mapped ground-truth metrics, physical volumes, and visible aligned viewer output, with source data kept outside the repository.
- **Verification:** `npm test`, `npm run build`, `npm run test:browser`, direct inference, `/api/evaluate`, browser screenshots.

### M8 — Documentation and final acceptance

- **Already works:** Existing DICOM compatibility and viewer documentation.
- **Implemented:** AI protocol, runtime, model limitations, performance, dataset findings, failure cases, and exact manual verification steps are documented in this plan and the README.
- **Verified:** A compatible public MSD Task01 sample now supplies a valid reference. Actual metrics, volumes, inference latency, geometry, viewer overlays, and screenshot evidence are recorded above. Phase 2's documented single-case technical acceptance criteria are met; generalization and clinical validation remain outside scope.
- **Dependencies:** M1–M7.
- **Acceptance:** No unverified item is called complete; actual rendering and evaluation evidence is available before declaring completion.
- **Verification:** Review docs against runtime config, model card, test logs, and browser evidence.

## Immediate implementation checklist

- [x] Inspect the existing viewer and preserve its current architecture.
- [x] Select a documented MONAI BraTS segmentation bundle and identify its input protocol.
- [x] Add a Python backend dependency pin set and confirm installation on Python 3.13 / Apple Silicon.
- [x] Keep the root `.python-version` unchanged and document the backend's separate Python 3.13 runtime.
- [x] Download the official artifact to `/private/tmp`, verify its expected SHA-256, and load the real bundle checkpoint on CPU.
- [x] Use the authorized small BraTS-GLI fastlane sample without downloading multi-gigabyte challenge archives. Its reference mask is not compatible with the accepted label encoding, so it cannot support evaluation.
- [x] Add an explicit local NIfTI assignment flow, loopback API, basic aligned-grid validation, MONAI inference path, mask/volume and reference-metric helpers, and Cornerstone labelmap overlay source code.
- [x] Add reversible orientation mapping; validate its coordinate round trip with synthetic inputs.
- [x] Verify that the returned labelmap geometry is accepted and overlays appear in axial, sagittal, and coronal views.
- [x] Verify 3D tumor visibility with the MRI actor left visible; capture an identifier-free screenshot outside the repository.
- [x] Execute all Node and Python tests, the synthetic DICOM browser suite, and the production build after the 3D visibility fix.
- [x] Retrieve one official MSD Task01 MRI and reference by HTTP range requests; validate the documented modality order, labels, geometry, and gzip integrity.
- [x] Run real 0.5.4 MONAI inference on the selected Task01 case and record actual latency and volumes.
- [x] Evaluate real WT/TC/ET Dice, IoU, and HD95 through `/api/evaluate` with the explicit MSD label convention; confirm the reference is unchanged.
- [x] Verify the same prediction in axial, sagittal, coronal, and 3D views, plus scrolling and opacity controls; capture screenshots outside the repository.
- [x] Run Node/Python tests, the synthetic DICOM browser suite, and the production build.
- [x] Update the plan with actual dataset provenance, model/checkpoint, measurements, scores, screenshots, and limitations.

## Current implementation snapshot

The official MONAI 0.5.4 checkpoint ran on one official MSD Task01 research case. Checkpoint SHA-256: `860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae`. Inputs were reordered from FLAIR/T1w/T1gd/T2w to T1c/T1/T2/FLAIR and normalized as configured; no spatial resampling was required. Direct CPU inference took 562.2 seconds; the real `POST /api/segment` route returned HTTP 200 in 646.9 seconds total, with 616.1 seconds attributed to model inference. Its response mask matched the direct result voxel-for-voxel and preserved the source affine. Predicted WT/TC/ET volumes were 83.945/3.467/2.980 cm³. Using Task01's documented labels without modifying the source, WT/TC/ET Dice were 0.896637/0.834522/0.831881 and IoU were 0.812641/0.716034/0.712155. Browser validation showed visible mask overlays in all three MPR planes, slice scrolling and opacity changes, and the same prediction visible in 3D. Screenshots remain outside Git at the paths listed in M5. The sample is a training-split case and independent overlap with the model's BraTS 2018 training cohort is unknown. Phase 2's documented single-case technical acceptance is met; it does not establish generalization or clinical performance. Phase 3 architecture and implementation status are tracked in [MEDICAL_RAG_ARCHITECTURE.md](MEDICAL_RAG_ARCHITECTURE.md).

## Phase 4 — model evaluation and fine-tuning platform (infrastructure status)

Phase 4 infrastructure is implemented in [`MODEL_TRAINING_AND_EVALUATION.md`](MODEL_TRAINING_AND_EVALUATION.md): explicit-manifest multi-case evaluation, aggregate JSON/CSV, safe CPU/MPS/CUDA device selection and CPU fallback, patient-group splits, pinned-config fine-tuning, local experiment artifacts, candidate checkpoint hash verification, paired baseline comparison, and a local-only results dashboard. The original checkpoint and all Phase 2/3 behavior remain the defaults.

The three-case MSD Task01 CPU pilot has now completed. This does **not** complete real Phase 4 research validation: no fine-tuning experiment, GPU inference/benchmark, or independent cohort evaluation has been executed. MSD Task01 is derived from the BraTS 2016/2017 challenge cases; the model checkpoint metadata reports BraTS 2018 training, but case-level overlap is unresolved. The three-case training-split metrics are potentially overlapping, not independent. The current Apple Silicon runtime has CPU available, MPS unavailable, no CUDA device, and 8 GiB physical RAM. No paid compute was used.

## Phase 4 real-experiment status — 2026-11-13

- **Dataset choice and restored pilot:** the initial three-case pilot was unavailable from its temporary location, so it was restored into `~/MedVision-Research/datasets/MSD-Task01/` using the official MONAI-managed HTTPS tar and byte ranges. Exactly three image/reference pairs were saved (30,549,259 payload bytes); tar-header index reads added 654,336 bytes. The total transfer was 31,203,595 bytes, under the 52 MiB cap. The 7.6 GB archive was not downloaded. Each range response, member length, gzip stream, four-channel image, geometry, spacing, and MSD label convention passed validation. The manifest is `~/MedVision-Research/manifests/msd-task01-three-case.json`.
- **Partial integrity:** only selected tar members were retrieved; the full-archive MD5 `240a19d752f0d9e9101544901065d872` was not claimed or checked. The selected data are Task01_BrainTumour training cases, licensed CC BY-SA 4.0 per the AWS Open Data Registry. The sample is potentially overlapping with the MONAI model's BraTS 2018 training cohort because MSD Task01 derives from BraTS 2016/2017 and case-level overlap is unresolved.
- **Overlap:** MSD Task01 is BraTS 2016/2017-derived; the pinned MONAI model reports BraTS 2018 training. No case-level cross-release overlap proof is available here. Classify MSD pilot results as potentially overlapping/exploratory. Confirmed independent cases: none. Do not assume that a later BraTS release is independent.
- **Alternatives reviewed:** the official BraTS 2018 data request requires CBICA IPP approval and provides four MRI modalities and ground truth for training cases, but is part of the checkpoint's own named training source and cannot serve as held-out evaluation. TCIA UCSF-PDGM is listed as 495 subjects / 142 GB; its relationship to later BraTS cohorts prevents assuming independence, and it is not the small direct-range pilot. BraTS-Africa is listed with CC BY 4.0 plus NIH controlled-access terms; the previously attempted Aspera installation remained blocked by signature verification. No installer security controls were bypassed.
- **Checkpoint readiness:** the official MONAI bundle 0.5.4 checkpoint was restored to `~/MedVision-Research/models/MONAI/` and matched pinned SHA-256 `860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae` before loading. CUDA and MPS are unavailable on this host.
- **Fine-tuning:** no real-data training was executed. There is no authorized patient-separated training/validation manifest; three evaluation-only cases do not establish a viable training cohort. Phase 5 later ran synthetic pipeline sanity checks (including one optimizer update and checkpoint restore), but these are not patient-data training or performance evidence.
- **GPU:** user-controlled Colab is the first no-provisioning option, but GPU type/availability and free-tier limits vary; any use requires intentional research-data transfer. No Google Cloud account credits are visible to this workspace, and no paid infrastructure was provisioned. The actual memory/time requirements must be measured on a granted GPU; the 224×224×144 training crop is only a configured size, not a verified VRAM estimate.
- **Resumable execution:** `backend/medvision_ai/research.py` now atomically writes JSON/CSV after every case and skips successful cases when resuming with the same manifest/checkpoint hashes. An output lock prevents duplicate processes. Synthetic interruption coverage confirms a completed case is not repeated. The run log, PID, start time, session handle, and exact resume command are stored in `~/MedVision-Research/logs/`.
- **Existing BraTS-GLI fastlane:** the user-provided archive and extracted directory were inspected in `~/Downloads/` before the MSD restoration. The subset contains five four-modality studies on a 240×240×155, 1 mm LPS grid with matching affine geometry. Its source is confirmed as the official Synapse BraTS-GLI fastlane compatibility demo (`syn52276402`); the archive SHA-1 matches the hash published in the official challenge test instructions. Its purported reference files each contain 11,263 distinct integer values, and all five arrays are identical; the existing loader rejects them as `INVALID_REFERENCE_LABELS`. The archive contains no separate license document. It is the only verified small package found, but it is not usable for training or evaluation. Normalized exact identifier matching against the reconstructed MSD pilot found zero matches, but no cross-release case map was available and the original pilot identifiers were lost; independence is not established.
- **Completed execution:** the three-case CPU evaluation ran from `2026-11-12T17:18:06Z` to `2026-11-12T17:41:43Z` (PID 10015), with all three cases successful and no failures. Mean WT/TC/ET Dice was 0.900586/0.888279/0.845939 and mean IoU was 0.820268/0.801125/0.733749. Mean inference latency was 454.083 seconds per case (total 1,362.248 seconds). Detailed per-case scores and predicted/reference volumes are in [`MODEL_TRAINING_AND_EVALUATION.md`](MODEL_TRAINING_AND_EVALUATION.md); result JSON/CSV, PID, log, and exact resume command are under `~/MedVision-Research/`. Phase 4 remains incomplete: no actual fine-tuning, accelerator benchmark, or independent-cohort evaluation has been performed.

## Phase 5 — fine-tuning audit — 2026-11-13

The persistent data audit found only the three MSD Task01 cases already used for baseline evaluation. Their manifest explicitly excludes them from training. Patient-level separation from the MONAI bundle's BraTS 2018 cohort is unresolved, and no verified independent training/evaluation cohort is available. The previously inspected BraTS-GLI fastlane subset was rejected because its segmentation labels are incompatible with the documented loader. Its source is confirmed as an official Synapse demo, but no separate license document is packaged with it. No evaluation cases were reused for training or model selection.

The fine-tuning pipeline now requires distinct training and final held-out manifests with stable patient-group identifiers, rejects overlap, supports configurable patch batches/crops/optimizer settings, and validates manifest, split, and configuration identity when resuming. The final held-out manifest is never consumed by the training loop. Default patch batch size is one. The verified original checkpoint remains unchanged.

The local Apple arm64 host has 8 CPU cores, PyTorch 2.15.0, MONAI 1.6.1, no CUDA or MPS, and 8 GiB RAM per the prior hardware audit. It is unsuitable for a credible full fine-tuning run on CPU. No real fine-tuning, overfit-on-research-sample, or paired evaluation was executed because there are no eligible training cases. Only synthetic code-level pipeline checks are appropriate here; they are not model performance evidence. See [`MODEL_TRAINING_AND_EVALUATION.md`](MODEL_TRAINING_AND_EVALUATION.md) for safeguards, the reproducible command, data limitations, and the minimum next requirement: a provenance-verified patient-separated training cohort plus final held-out cohort. Phase 5 remains exploratory infrastructure work; no independent validation or improvement claim is made.

See [MODEL_TRAINING_AND_EVALUATION.md](MODEL_TRAINING_AND_EVALUATION.md) for detailed source, access, integrity, device, and reproduction notes.

## Phase 5.1 — BraTS data access decision — 2026-11-13

The user's signed-in Synapse session was not available to this workspace, so registration and file permissions could not be verified. The provided 2026 challenge URL is a multi-task challenge overview, not the BraTS2024 adult-glioma MRI access page. For four-modality glioma segmentation, the most compatible target is the official BraTS2024 adult glioma post-treatment release. The relevant files require access through the separate BraTS2024 Data Access team and are subject to its post-challenge terms. Validation labels remain private. The organizer has not provided an official 5–10-case package; the reported training archive size is about 34.9 GB (participant estimate), so no download was started without approval.

The small official BraTS-GLI fastlane demo package (`syn52276402`) is the same package already present locally, verified by matching its published SHA-1. All five segmentation arrays were previously found to be duplicates with 11,263 distinct values and to match an MRI volume, rather than a valid label mask. The original files were left untouched and were not used. No smaller official labeled BraTS2024 GLI package was found; the 2024 training archive is distributed as one large ZIP, and the organizer says it will not provide individual files. A participant reported a 34.9 GB size. The explicit BraTS2024 label convention was added to training/evaluation mappings with cavity label 4 excluded from WT/TC/ET. This code path has synthetic regression coverage only; no BraTS case has been loaded or evaluated.

Any future split must group repeated observations by patient, reserve a labeled patient-disjoint holdout before training, and keep the prior three MSD evaluation cases excluded. Since released BraTS2024 validation labels are private and patient-level overlap with the pretrained checkpoint's BraTS2018 cohort cannot be checked, a local labeled holdout would be exploratory; independent validation is not established. The user now reports Synapse permission and has opened the separate post-challenge page; the local browser surface does not expose its file listing for verification. No archive transfer has begun. The reported ~34.9 GB full archive needs separate approval under the user's earlier download-size constraint. See [MODEL_TRAINING_AND_EVALUATION.md](MODEL_TRAINING_AND_EVALUATION.md) for access links, source notes, and the exact next step.

## Phase 5.2 — Small-cohort acquisition feasibility — 2026-11-13

The BraTS2024 training entity is a single ~34.9 GB reported ZIP. Synapse's official Python/CLI clients download whole file entities; direct ZIP-entry retrieval would require authenticated HTTP Range support, but the signed-in tab and Synapse API were unavailable to this workspace. A one-byte/tail-range test could not be completed because outbound requests failed. Do not assume range access, resume, or remote ZIP mounting works until tested against the protected entity. No data transfer occurred.

The best smaller licensed alternative is the preprocessed BraTS-Africa public NIfTI collection: 146 subjects, four MRI modalities, expert-reviewed masks, 1 mm isotropic SRI24 registration, and CC BY 4.0; the advertised public labeled package is 1.6 GB. Ten subjects would average about 110 MB by simple division, but TCIA case-level download of this particular NIfTI package was not verified. The existing small BraTS fastlane is official but its masks fail validation, so it remains excluded. If approved and selected-subject access is confirmed, use only glioma cases and a patient-separated exploratory split; preserve the prior MSD cases for baseline only.

No GPU compute is attached to Synapse. Direct S3/STS access requires an STS-configured location, which has not been verified for the BraTS project. Free Colab GPUs are not guaranteed and free runtimes are limited; no dataset was uploaded to a third party. The Mac CPU remains impractical for this 3D fine-tuning workload. See [MODEL_TRAINING_AND_EVALUATION.md](MODEL_TRAINING_AND_EVALUATION.md) for source links, size estimates, access findings, and the approval gate.
