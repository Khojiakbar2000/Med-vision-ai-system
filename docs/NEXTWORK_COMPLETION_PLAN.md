# NextWork completion plan

Source: https://nextwork.ai/projects/4b262351-63f1-453e-afbf-985f9ba383b2 (accessed 2026-11-11). All sections were retrieved, including workspace, temporary slice preview, final volume loader, controls/statistics, scrub/audit, Safari acceptance, Secret Mission and cleanup. The full retrieved HTML/text is temporary outside the repository.

This matrix distinguishes source inspection from executed proof. COMPLETE never means code presence alone. Public anatomical rendering has now been verified in desktop Chrome; the older Safari-specific acceptance check remains open. No protected user scan folder will be inspected by the agent.

## Requirements matrix

| ID | Tutorial requirement | Existing implementation | Status | Missing work | Verification |
|----|----------------------|-------------------------|--------|--------------|--------------|
| R01 | Required: source/scan separation, restricted agent access | AGENTS.md; CLAUDE.md; .gitignore | PARTIAL | Rules exist; native permission enforcement and user folder separation need manual confirmation | Source review; manual checklist A |
| R02 | Required setup: runtime/package compatibility | package.json; vite.config.js; .nvmrc; .python-version | PARTIAL | Recheck build. Exact tutorial Node26/Python3.15 not installed; compatible Node24/Python3.13 used | M1 commands |
| R03 | Required: private Vite package and exact imaging pins | package.json scripts/dependencies | COMPLETE | No remaining automated task; native/private-study acceptance tracked separately in R23. | Executed synthetic browser/unit proof; see NEXTWORK_FINAL_VERIFICATION.md. |
| R04 | Required: dark four-panel shell, toolbar, safe summary, research note | index.html; src/style.css | COMPLETE | No remaining automated task; native/private-study acceptance tracked separately in R23. | Executed synthetic browser/unit proof; see NEXTWORK_FINAL_VERIFICATION.md. |
| R05 | Required: browser-controlled local folder selection | index.html #folder-input; src/main.js change handler | COMPLETE | Native Safari picker still needs manual validation | M2; browser chooser plus public series in Chrome |
| R06 | Required intermediate stage: first local slice STACK proof | LocalDicomViewer.initialize/loadFiles currently volume-only | COMPLETE | No remaining automated task; native/private-study acceptance tracked separately in R23. | Executed synthetic browser/unit proof; see NEXTWORK_FINAL_VERIFICATION.md. |
| R07 | Required: local geometry parsing, no displayed identifiers | src/core/dicom-series.js parseCandidate; LocalDicomViewer.#updateStats | COMPLETE | No remaining automated task; native/private-study acceptance tracked separately in R23. | Executed synthetic browser/unit proof; see NEXTWORK_FINAL_VERIFICATION.md. |
| R08 | Required: same series/frame-reference/orientation/dimensions grouping, largest multi-slice group | chooseLargestCompatibleSeries; LocalDicomViewer.loadFiles | COMPLETE | No remaining automated task; native/private-study acceptance tracked separately in R23. | Executed synthetic browser/unit proof; see NEXTWORK_FINAL_VERIFICATION.md. |
| R09 | Required limitation: exclude multi-frame from tutorial path | LocalDicomViewer.loadFiles currently expands frames automatically | COMPLETE | No remaining automated task; native/private-study acceptance tracked separately in R23. | Executed synthetic browser/unit proof; see NEXTWORK_FINAL_VERIFICATION.md. |
| R10 | Required: ordered slice IDs before volume creation | chooseLargestCompatibleSeries; projectedPosition | COMPLETE | No remaining automated task; native/private-study acceptance tracked separately in R23. | Executed synthetic browser/unit proof; see NEXTWORK_FINAL_VERIFICATION.md. |
| R11 | Required: preload every image before shared streaming volume creation | LocalDicomViewer.loadFiles | COMPLETE | No remaining automated task; native/private-study acceptance tracked separately in R23. | Executed synthetic browser/unit proof; see NEXTWORK_FINAL_VERIFICATION.md. |
| R12 | Required: axial/sagittal/coronal ORTHOGRAPHIC shared volume | LocalDicomViewer.initialize/loadFiles | COMPLETE | Safari-specific rendering remains unverified | Synthetic assertions plus public MR screenshot and geometry evidence in REAL_DATA_VALIDATION.md |
| R13 | Required: experimental VOLUME_3D shared volume | LocalDicomViewer.initialize/loadFiles | COMPLETE | Safari-specific rendering remains unverified; 3D stays experimental | Synthetic assertions plus public MR volume screenshot in REAL_DATA_VALIDATION.md |
| R14 | Required: primary MPR window/level, wheel scrolling | LocalDicomViewer.#configureTools/selectMprTool | COMPLETE | No remaining automated task; native/private-study acceptance tracked separately in R23. | Executed synthetic browser/unit proof; see NEXTWORK_FINAL_VERIFICATION.md. |
| R15 | Required: calibrated millimetre ruler and active control state | LengthTool; selectMprTool; validateVolumeGeometry; src/main.js | COMPLETE | Broader acquisitions remain unverified | Synthetic 20 mm test and public MR 20 mm measurement (20.37 mm) |
| R16 | Required: primary 3D rotation and modality-appropriate presets | TrackballRotateTool; #populatePresets/#selectModalityPreset/setPreset | COMPLETE | Other modality-specific choices remain user-selectable | Synthetic assertions; public MR uses MR-Default and renders its volume |
| R17 | Required: only modality, slices, matrix, spacing, thickness summary | LocalDicomViewer.#updateStats; index.html | COMPLETE | No remaining automated task; native/private-study acceptance tracked separately in R23. | Executed synthetic browser/unit proof; see NEXTWORK_FINAL_VERIFICATION.md. |
| R18 | Required: safe failures, progress, reload before changing series | src/main.js setStatus/change handler; SeriesSelectionError | COMPLETE | No remaining automated task; native/private-study acceptance tracked separately in R23. | Executed synthetic browser/unit proof; see NEXTWORK_FINAL_VERIFICATION.md. |
| R19 | Required: no medical uploads, analytics, remote image URLs, identifying logs | index.html CSP; LocalDicomViewer.initialize logging; browser tests | COMPLETE | No remaining automated task; native/private-study acceptance tracked separately in R23. | Executed synthetic browser/unit proof; see NEXTWORK_FINAL_VERIFICATION.md. |
| R20 | Required: separate best-effort copy, pinned deid/pydicom, count-only audit | tools/deidentify_audit.py create_copy/read_headers/audit/main; requirements-audit.txt | PARTIAL | Restore documented tutorial dependency/entrypoint; preserve working compatible copy; explain deid recipe UID loss | M8 |
| R21 | Required: audit fails for empty/reused identifiers; disclose limits | tools/deidentify_audit.py audit; tests/test_deidentify_audit.py | COMPLETE | No remaining automated task; native/private-study acceptance tracked separately in R23. | Executed synthetic browser/unit proof; see NEXTWORK_FINAL_VERIFICATION.md. |
| R22 | Required: scrubbed copy still renders and controls work | create_copy; browser fixture tests | PARTIAL | Synthetic scrubbed copy render is proven; real Safari copy remains unverified | M8/M9; manual C |
| R23 | Required final: authorized selected study demonstrated in Safari | Local public MR series verified in Chrome | PARTIAL | Repeat compatible public-series rendering in native Safari if Safari parity is required | M10 manual A-C |
| O01 | Optional: Secret Mission denied agent read and browser access | AGENTS.md rules exist | UNVERIFIED | Do not attempt protected access; user reviews permission summary independently | Manual D; not completion blocker |
| O02 | Optional: pause/cleanup local resources | README.md run instructions | COMPLETE | Existing extension preserved / instructions documented. | Executed synthetic browser/unit proof; see NEXTWORK_FINAL_VERIFICATION.md. |
| X01 | Existing extension outside tutorial: enhanced multi-frame, pan/zoom | expandFrames; PanTool; ZoomTool | COMPLETE | Existing extension preserved / instructions documented. | Executed synthetic browser/unit proof; see NEXTWORK_FINAL_VERIFICATION.md. |
| X02 | Advanced outside scope: AI diagnosis/RAG/training | No processing services implemented | COMPLETE | Excluded from baseline; no AI feature work | Source review |

## Dependency-ordered milestones

### M1 — R02 R03

- [x] Existing: pinned packages and worker fixes. Missing: fresh compatibility evidence. Tasks: inspect versions; run Node unit suite and Vite build. Depends: none. Acceptance: compatible installed pins build successfully, runtime deviations documented.
- Verification/results: source audit, 5 Node tests, 4 Python tests and browser assertions executed; see final report. Native Safari portions remain open under M10.

### M2 — R04 R05 R18

- [x] Existing: explicit folder button, ready/error status. Tasks: verify chooser event, disabled controls and failure retry. Depends: M1. Acceptance: browser chooser supplies only selected File objects; no identity text.
- Verification/results: source audit, 5 Node tests, 4 Python tests and browser assertions executed; see final report. Native Safari portions remain open under M10.

### M3 — R06

- [x] Existing: final axial volume slice renders in prior synthetic proof. Tasks: add explicit temporary stack-preview mode and pixel assertion. Depends: M2. Acceptance: one selected synthetic slice renders in axial only without volume creation.
- Verification/results: source audit, 5 Node tests, 4 Python tests and browser assertions executed; see final report. Native Safari portions remain open under M10.

### M4 — R07 R08 R09 R10 R11

- [x] Existing: parser, physical sorting, preload, geometry guard. Tasks: restore tutorial single-frame default, retain enhanced extension opt-in, verify mixed/reversed inputs and geometry. Depends: M3. Acceptance: largest compatible single-frame volume decoded after preload; unsupported objects safely rejected.
- Verification/results: source audit, 5 Node tests, 4 Python tests and browser assertions executed; see final report. Native Safari portions remain open under M10.

### M5 — R12

- [x] Existing: three shared orthographic viewports. Tasks: rerun pixels/dimensions/spacing/camera assertions. Depends: M4. Acceptance: synthetic assertions and public MR anatomical render passed in Chrome; Safari-specific acceptance remains open.
- Verification/results: source audit, 5 Node tests, 4 Python tests and browser assertions executed; see final report. Native Safari portions remain open under M10.

### M6 — R13 R16

- [x] Existing: experimental 3D viewport/presets/rotation tool. Tasks: assert visible pixels, camera change on rotation and preset screenshot change. Depends: M5. Acceptance: synthetic assertions and public MR anatomical volume render passed in Chrome; experimental limitation retained.
- Verification/results: source audit, 5 Node tests, 4 Python tests and browser assertions executed; see final report. Native Safari portions remain open under M10.

### M7 — R14 R15 R17

- [x] Existing: window/level, scroll, ruler, safe stats; extra pan/zoom. Tasks: recheck gesture properties, active state, 20mm calibration and all summary fields. Depends: M6. Acceptance: browser assertions pass; no identifying values in UI.
- Verification/results: source audit, 5 Node tests, 4 Python tests and browser assertions executed; see final report. Native Safari portions remain open under M10.

### M8 — R19 R20 R21 R22

- [ ] Existing: local-only CSP, muted library logging, copy/audit CLI. Tasks: restore tutorial-named audit entrypoint/pins; add empty-audit test; render actual synthetic scrubbed copy; document UID-preserving compatibility adjustment. Depends: M7. Acceptance: aggregate-only audit rejects empty/reused fields, scrubbed output remains renderable; real copy requires user Terminal/Safari.
- Verification/results: source audit, 5 Node tests, 4 Python tests and browser assertions executed; see final report. Native Safari portions remain open under M10.

### M9 — R05 R12 R13 R18 R19 R22

- [x] Existing: unit/Python/browser harness. Tasks: actual chooser event, safe unsupported/malformed errors, stack/raw/scrubbed synthetic runs and build. Depends: M8. Acceptance: executed tests with screenshots outside repo; no parser-only rendering claim.
- Verification/results: source audit, 5 Node tests, 4 Python tests and browser assertions executed; see final report. Native Safari portions remain open under M10.

### M10 — R01 R23 O02

- [ ] Existing: README/architecture/compatibility/test reports. Tasks: final verification report, exact private Safari steps, safe confirmation checklist and stop instructions. Depends: M9. Acceptance: every required row verified, including user Safari original/scrubbed study confirmation; otherwise remain incomplete.
- Verification/results: source audit, 5 Node tests, 4 Python tests and browser assertions executed; see final report. Native Safari portions remain open under M10.

## Deliberate adaptations and exclusions

- Preserve the working modular architecture; tutorial main.js responsibilities are in core/service/main modules.
- Physical orientation/position sorting is retained from the earlier user requirement; the tutorial sorts instance numbers. Real-study ordering must still be visually confirmed.
- Multi-frame is not part of tutorial acceptance. Existing tested extension is preserved but disabled by default. Pan/zoom are retained extras, not missing tutorial features.
- The tutorial deid default can remove class/series UIDs and break its own final load gate. The existing UID-preserving local copy remains a documented compatible alternative; no de-identification certification is claimed.
- Claude-specific permission/install commands are not copied as purported Codex enforcement. User must verify their active agent permission controls. Exact tutorial runtime pins are not falsely reported as installed.
- Automated fixtures and screenshots are synthetic only. Real protected-study verification happens solely through user-controlled Safari selection and a separate user Terminal.
- AI/RAG/fine-tuning/clinical diagnosis are excluded. Cleanup deletion is optional and never run automatically.

## Run log

- Initial audit: worker/metadata fixes and synthetic tests exist, but no real Safari confirmation, isolated stack preview, rotation assertion or scrubbed-output browser test. Plan created before new implementation.

- M1–M7: missing stack proof and default restriction implemented; existing rendering and interaction code preserved.
- M8: tutorial-named audit entrypoint and pinned tools restored; synthetic scrubbed output passes real browser rendering and controls. Exact default deid recipe remains a documented compatibility deviation; user external-folder workflow remains pending.
- M9: stack/raw/scrubbed/default-rejection/opt-in browser runs passed; malformed rejection and production four-viewport pixel smoke also passed.
- M10: public de-faced real MR series verified in local Chrome and documented. Native Safari parity and manual private-copy workflow remain unverified. No protected scan directory accessed.

## Final milestone gate status

| Milestone | Status | Remaining gate |
|---|---|---|
| M1 | PARTIAL | Compatible runtime verified; literal tutorial Node/Python runtime targets differ |
| M2 | PARTIAL | Chrome chooser proven; native Safari/user external-folder selection pending |
| M3 | COMPLETE | Isolated synthetic STACK decoding and visible pixels proven |
| M4 | COMPLETE | Single-frame restriction, grouping, ordering, preload and volume verified automatically |
| M5 | PARTIAL | Synthetic and real-study MPR verified in Chrome; Safari rendering remains unverified |
| M6 | PARTIAL | Synthetic and real-study 3D verified in Chrome; 3D remains experimental and Safari is unverified |
| M7 | PARTIAL | Synthetic controls and calibrated real MR ruler verified in Chrome; Safari remains unverified |
| M8 | PARTIAL | Synthetic copy/audit/reload proven; exact deid recipe deviation and user copy/audit pending |
| M9 | COMPLETE | Unit, privacy, slice, volume, scrubbed, rejection and production browser tests passed |
| M10 | PARTIAL | Native Safari acceptance and any user-run private-copy workflow remain open |

## Literal tutorial deid compatibility experiment

Executed the pinned `deid --overwrite --outfolder … identifiers --action all --input …` only against an externally generated synthetic 16-file series, capturing tool output without disclosing file values. Command exit: success. Output headers: 16. Headers missing required image-class field: 16. Headers missing required series-grouping field: 16. Thus command success cannot satisfy the tutorial’s scrubbed-copy render gate. The working compatible `create-copy` workflow is retained and passed actual scrubbed-output rendering/calibration tests; literal command compatibility remains PARTIAL rather than falsely COMPLETE.
