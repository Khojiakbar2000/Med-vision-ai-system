# Privacy rules for this workspace

- Keep original and de-identified medical files outside this repository and outside the coding workspace.
- Never read, list, glob, search, attach, summarize, or request contents from the user's scan folders. The application receives files only after explicit selection in the browser.
- Never print or log filenames, DICOM headers, patient identifiers, UIDs, image pixels, or raw loader/parser errors.
- Do not add analytics, telemetry, remote storage, uploads, remote image URLs, or network calls to the viewer.
- Automated tests must use synthetic DICOM fixtures only.
- Describe this project as educational/research software, not a diagnostic medical device.
