# Codex handoff: BraTS2024 range probe

**Status as of 2026-11-13:** Synapse browser access and metadata authentication work, but the official file endpoint denied the one-byte request. Do not download the full archive.

## Verified

- The Synapse Data page `syn64952546` contains a BraTS-GLI folder; the official GLI training archive is `syn60086071`.
- Chrome now exposes the user's Synapse session through native UI automation. The file page shows view and download permission. Do not inspect browser credentials/cookies or read tokens from browser storage.
- The official archive is V2, 34.89 GB, with MD5 `1d910b17d6cd32e38aa6296b8dfb7c77` (the page metadata matched the pinned official value).
- The official `synapseclient` package was installed into `/private/tmp/synapse-client-probe` after escalation approval. It was not installed in the repository environment.
- The user configured `SYNAPSE_AUTH_TOKEN` in the repository `.env`. The probe read that variable into process memory without displaying it. Synapse login and the metadata/file-handle lookup succeeded.
- A one-byte tail request (`Range` for the final byte) returned HTTP 403 with the sanitized storage error code `AccessDenied`. No archive bytes were received. Byte-range support and ZIP central-directory access remain unverified.
- The PAT may lack the `download` scope even though the browser account page shows download permission; this is a hypothesis, not a confirmed cause. Do not print or request the token.
- The existing BraTS GLI fastlane package was already inspected and failed the project's segmentation-label validation. Do not reuse it or redownload it as training data.
- BraTS-Africa's official public processed NIfTI item is a 1.6 GB package. TCIA per-series APIs are documented for imaging downloads, but individual-subject retrieval from this packaged NIfTI item has not been verified. Do not download it under the current 250 MB case-pilot cap.
- No valid new cases, manifests, preprocessing run, or training sanity test were produced. The MSD baseline cases remain excluded from training.

## Next action

Ask the user to verify that the PAT in `.env` has both `view` and `download` scopes. If needed, they should update it locally and tell Codex when ready. Never ask them to paste the token into chat. If the PAT has both scopes and the file endpoint still returns 403, stop and report the result or seek official Synapse support; do not bypass access controls.

After the scope check, rerun the single-byte tail request. Print only status and range headers; never print the URL or raw exception. HTTP 206 verifies byte-range support at the tail. It does not itself parse the ZIP directory. If that succeeds, fetch only the minimal ZIP/ZIP64 tail metadata needed to locate the central directory, then estimate selected-entry transfer sizes before extracting any subjects. Do not fetch the full archive or exceed an approved transfer scope.

The previous probe ran inline and suppressed the signed URL and response body. It made only metadata calls plus one-byte tail requests. The current temporary client path is `/private/tmp/synapse-client-probe`; it may not persist indefinitely.

## After a successful range check

1. Verify ZIP directory access with a tail-only request and inspect entry names/sizes without extracting.
2. Confirm the archive's modality suffixes, segmentation files, subject IDs, and per-case compressed sizes from directory metadata.
3. Select 10–15 complete labeled subjects only if the transfer estimate is within the authorized scope; keep the existing three MSD cases out of all training/selection splits.
4. Extract outside the repository, validate labels and image/reference geometry, then create patient-separated training, validation, and untouched evaluation manifests.
5. Run only a minimal real-data preprocessing/training sanity check. Label any evaluation exploratory unless cohort independence from the original MONAI training data is established.

## Git workflow for this repository

Make and push task commits on `develop` first. Then switch to `master`, merge `develop` into `master`, and push `master`. Keep unrelated working-tree changes out of the commits. Do not push directly to `master` unless the user explicitly asks for a different workflow.

## Workspace note

The original checkout had numerous unrelated modified and untracked files. This handoff should be committed on `develop`, then carried into `master` by merging `develop`; do not stage unrelated files.
