# Architecture

The Vite application passes explicitly selected browser File objects to Cornerstone's local file manager. No HTTP image loader, upload endpoint, AI service, or persistence is exposed by the interface.

`src/core/dicom-series.js` parses minimal geometry, expands enhanced frames with explicit frame geometry, groups compatible images, projects positions onto the orientation normal, and rejects invalid spatial volumes. `src/services/local-viewer.js` owns image decoding, the shared volume, MPR and experimental 3D viewports, calibrated length tools, gestures, and geometry statistics. `src/main.js` handles controls and fixed, non-identifying errors.

Cornerstone 5.15.2 defaults to a naturalized metadata provider which failed the local native-pixel fixture. The local viewer uses the library's legacy metadata provider explicitly. Vite excludes the image loader from dependency optimization so its module Worker URL can be transformed; codec CommonJS entry points are explicitly optimized. Two decoding workers bound concurrency.

Future processing services should consume an explicitly approved in-memory volume through a separate adapter. They must not own browser controls or rendering state. No such processing service is currently implemented.

The Python privacy CLI is a separate local workflow, outside the viewer. Its audit is a narrow header comparison, not certification; pixel text and private attributes need independent review.
