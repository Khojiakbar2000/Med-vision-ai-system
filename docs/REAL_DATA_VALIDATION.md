# Real DICOM rendering validation

Educational and research software. Not a diagnostic medical device.

## Dataset

- **Source:** [DataLad example structural DICOM dataset](https://github.com/datalad/example-dicom-structural), a minimal excerpt of the [StudyForrest dataset](https://www.nature.com/articles/sdata20143).
- **Data license:** Open Data Commons Public Domain Dedication and License (PDDL), as supplied by the dataset repository.
- **Privacy:** The source documents this volume as created from a de-faced NIfTI image and its DICOM headers anonymized. The source data stayed in `/private/tmp`, outside this repository and coding workspace.
- **Series:** 384 single-frame MR images; 384 × 274 image matrix; Implicit VR Little Endian; 0.667 × 0.700 mm in-plane pixel spacing; 0.667 mm measured slice spacing; axial acquisition with mild obliquity.
- **Reconstructed volume:** 274 × 384 × 384 voxels; approximately 0.700 × 0.667 × 0.667 mm voxel spacing.

Exact anonymized identifiers and raw DICOM header values are intentionally not recorded.

## Verification evidence

The real series was selected through the browser's local folder chooser and decoded by the bundled Cornerstone DICOM image loader. It grouped all 384 slices, sorted by physical image position, created a volume, and displayed nonuniform image pixels in all four viewports. Axial, sagittal, and coronal screenshots showed the same recognizable brain anatomy in the corresponding planes. The 3D viewport displayed the head volume from these same decoded voxels.

The detected MR modality now selects Cornerstone's `MR-Default` volume preset before rendering. This corrected the initial CT bone preset, which made the MR volume dark and red. The selector remains available for changing presets.

A 20 mm distance was drawn with the viewer's real length tool on the axial viewport. The annotation measured **20.37 mm**, within 0.5 mm of the expected distance using the dataset's DICOM geometry. This confirms calibration for this series and does not certify measurement accuracy for other acquisitions.

The local Chrome run reported zero page errors and zero external requests. The selected files were handled only in the browser; no scan data were uploaded. The screenshot artifact is stored outside the repository at:

`/private/tmp/medical-vision-public-brain-mr-viewer.png`

Individual panel captures are `/private/tmp/medical-vision-public-axial.png`, `-sagittal.png`, `-coronal.png`, and `-volume-3d.png`.

## Repeat manually

1. Download the [source archive](https://api.github.com/repos/datalad/example-dicom-structural/tarball/f077bcc8d502ce8155507bd758cb3b7ccc887f40) (about 15.8 MB compressed) and extract it outside this repository. The extracted data directory in this session is `/private/tmp/medical-vision-public-brain-mr/datalad-example-dicom-structural-f077bcc/dicoms`.
2. Start this application with `npm run dev`.
3. Choose the extracted `dicoms` folder with the application's folder button.
4. Wait for the selected-slice count to reach 384. Confirm the axial, sagittal, coronal and experimental 3D panels show the scan and that the preset selector reads `MR-Default`.
5. Select **Distance** and draw a line across the axial viewport. The reported length should use the scan's spatial calibration.

Automated test fixtures remain synthetic to satisfy this workspace's privacy rule. The public research series was used only for local browser rendering evidence and is not included in the repository.

## Scope and limitations

This establishes successful rendering for this single, anonymized structural T1 MR series and native Implicit VR Little Endian encoding. It does not establish universal DICOM compatibility, correctness for every scanner orientation, clinical validity, or general diagnostic accuracy. The 3D renderer remains experimental and depends on the browser's WebGL implementation and the selected transfer function.
