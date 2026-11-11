# Verification notes

The current authoritative tutorial audit and executed results are [NEXTWORK_FINAL_VERIFICATION.md](docs/NEXTWORK_FINAL_VERIFICATION.md). The notes below describe earlier runs and may predate the tutorial default restriction and production proof.

Latest loading-diagnostic regression: six Node tests, four Python tests, and the synthetic missing-Series-UID Chrome scenario passed. When every selected DICOM lacks Series Instance UID, the viewer now names that missing grouping field and explains the known tutorial deid incompatibility without displaying header values.

# Current verification status

Actual anatomical rendering has now been verified in local Chrome with a public, de-faced, anonymized MR series. See [REAL_DATA_VALIDATION.md](docs/REAL_DATA_VALIDATION.md) for dataset details, screenshots and the limits of this evidence. Support remains bounded by the compatibility notes and is not a claim of clinical or universal DICOM validation.

## Audit findings

- Existing parsing/grouping tests passed, but had no pixels and could not validate decoding or rendering.
- Local decoding stalled: Vite did not transform the optimized loader's module Worker correctly; codec CommonJS imports also required optimization.
- Cornerstone 5.15.2's default naturalized metadata path failed to supply native pixel frame data for the synthetic local fixture. The legacy metadata provider is now explicitly selected.
- Multi-frame input was rejected unconditionally; enhanced spatial frame expansion is implemented but needs decoding proof.
- Pan and zoom bindings were absent; middle/right mouse bindings are now added.
- Spatial geometry, positive spacing, duplicates and irregular gaps were not validated before measurement/reconstruction; validation is now added.
- Folder switching still requires reload. Single-image stacks do not produce a volume.
- The literal tutorial `deid` command may strip SOP Class and series UIDs, which would make its output unusable in this viewer. The separate `create-copy` path preserves those required fields and remaps identifiers; its synthetic output passed rendering and audit. No private scrubbed study was accessed.
- Dependency audit reports 10 affected packages (4 moderate, 6 high), including transitive dcmjs/adm-zip/fflate/uuid/vtk dependencies. Automatic suggested downgrade is incompatible; no forced downgrade was applied. Do not treat the dependency tree as security certified.

## Executed checks

- Existing unit tests: four Node and two Python tests passed after frame expansion changes.
- Browser harness now installs and launches an isolated Chrome profile. Early runs exposed actual module worker and native-pixel metadata errors. These failures are recorded rather than counted as rendering success.
- Real-data decoding, MPR orientation, calibration and 3D proof are recorded in [REAL_DATA_VALIDATION.md](docs/REAL_DATA_VALIDATION.md).

No user/private scan folder was accessed. The only real scan inspected was the public de-faced research sample documented below. No external image service or AI service was used.

## Subsequent executed evidence

The worker and metadata fixes now pass browser decoding and rendering for **synthetic** native Explicit VR Little Endian MR data, in both 16 separate files and one enhanced 16-frame object:

- All four viewport screenshots contain visible nonuniform image pixels.
- Volume dimensions are 64 × 64 × 16; voxel spacing is 1 × 1 × 2 mm.
- Axial, sagittal and coronal camera normals align with the expected phantom patient axes.
- A distance drawn using the real Length tool measures a known 20 mm displacement within 0.5 mm.
- Primary window/level drag, middle pan, right zoom and wheel slice scrolling change the corresponding viewport properties.
- Changing the experimental 3D preset changes rendered screenshot pixels.
- No external requests, browser page errors, or synthetic identifying sentinel in UI/console were observed.
- Screenshot artifact: `/private/tmp/dicom-synthetic-render.png` (synthetic data only).
- Five Node unit tests passed. Three Python tests passed with pydicom installed, including actual synthetic copy/audit output preserving class/spacing and remapping identity.
- Production build passed under Node 24.22.1, with codec Node-builtin externalization and large-bundle warnings. Browser assertions target the development server, not the production bundle.

The old deid CLI invocation has been replaced by a local pydicom header-copy workflow. SOP classes and transfer syntax are retained; instance/reference identities are remapped consistently. The corresponding earlier audit defect is resolved for the synthetic fixture. It remains best effort and is not a DICOM confidentiality-profile implementation.

## Real imaging acceptance evidence

- Public research series: DataLad example structural T1 MR, described by its source as de-faced and anonymized; PDDL data license.
- Browser-loaded 384 single-frame DICOM slices from a directory chosen through the local folder picker.
- All four viewports rendered varied pixel values; the three MPR views showed corresponding brain anatomy. The 3D panel showed the decoded head volume.
- Reconstructed dimensions: 274 × 384 × 384; pixel spacing approximately 0.667 × 0.700 mm; measured slice interval approximately 0.667 mm; mild axial obliquity; native Implicit VR Little Endian.
- Distance tool: 20 mm test line measured 20.37 mm.
- MR preset selected automatically as `MR-Default`, correcting the unsuitable CT-Bone default for MR.
- Browser run: zero page errors and zero external requests.
- Public-series screenshots are outside the workspace under `/private/tmp/medical-vision-public-...png`; no public DICOM data or public-series screenshot was committed. Separate synthetic preview screenshots remain in `docs/images/`.
- Automated unit tests continue to use synthetic fixtures only. The real series was used for local browser rendering evidence and is not part of the automated test suite.

This does not validate other transfer syntaxes, acquisitions, orientations, browsers or clinical use. The 3D view remains experimental.

## Latest executed commands

- `env PYTHONPATH=/private/tmp/medical-vision-test-python npm test` with Node 24.22.1: **6 Node tests and 4 Python tests passed**. The temporary Python target contains the already-declared `pydicom==3.0.2`; it is outside the repository.
- `env PYTHONPATH=/private/tmp/medical-vision-test-python npm run test:browser` with Node 24.22.1: **passed** the synthetic MR browser test, including 4 rendered panels, 20 mm calibration, no external requests and no page errors.
- Local Chrome public-data verification: **384-slice real MR series rendered in all four panels**; measured 20 mm as 20.37 mm; MR preset selected; no external requests or page errors.
- `/Users/aaaaa/.nvm/versions/node/v24.22.1/bin/npm run build`: **passed**. Vite still reports codec Node-module externalization and large-bundle warnings.
