# DICOM compatibility

Educational/research software. Not a diagnostic device.

## Current spatial requirements

At least two monochrome spatial slices in one compatible series, with consistent dimensions, orientation and positive pixel spacing. Sorting uses ImagePositionPatient projected onto the normal derived from ImageOrientationPatient. Duplicate positions and slice gaps differing by more than 2% (minimum 0.01 mm tolerance) are rejected. Missing slices at a perfectly regular interval cannot be detected. Tilted/sheared, temporal, diffusion and mixed acquisition data require additional validation and are not certified supported.

The default tutorial path excludes multi-frame input. Only with the optional `?enhanced=1` extension, enhanced multi-frame input is expanded into one-based Cornerstone frame IDs only when each frame has explicit position and orientation, including shared functional groups. Cine without spatial positions is rejected. Limit: 4096 frames per instance. Synthetic native Explicit VR Little Endian enhanced MR has passed pixel rendering and 20 mm calibration assertions; real enhanced acquisitions remain unverified.

## Transfer syntax status

Bundled Cornerstone 5.15.2 decoder code includes native Implicit VR Little Endian (1.2.840.10008.1.2), Explicit VR Little Endian (.1), Explicit VR Big Endian (.2), deflated Explicit VR (.1.99), encapsulated uncompressed (.1.98), deflated frames (.8.1), RLE (.5), JPEG (.4.50/.51/.53/.55/.57/.70), JPEG-LS (.4.80/.81), JPEG 2000 (.4.90/.91), HTJ2K (.4.201/.202/.203), and JPEG XL (.4.110/.111/.112). Prefix for abbreviated suffixes: 1.2.840.10008.1.2.

Decoder availability is not a verification claim. Native Implicit VR Little Endian has been verified on synthetic fixtures and one public, anonymized, defaced structural T1 MR series. Compressed syntaxes, native big endian and deflated files have not been demonstrated in this application. MPEG/video, arbitrary private encodings, segmentation, radiotherapy objects, PDF, ECG, and non-spatial cine are unsupported by the volume UI. See [REAL_DATA_VALIDATION.md](docs/REAL_DATA_VALIDATION.md) for actual-image evidence.

## Browser status

Desktop Chrome is the automated target using an isolated headless profile and software WebGL. Safari and Edge need manual rendering validation. Mobile and Firefox are unverified. WebGL, directory input, module Workers and adequate browser memory are required. Experimental 3D depends on GPU/driver and transfer-function suitability. The viewer selects `MR-Default` for MR data and retains CT presets for CT volumes.
