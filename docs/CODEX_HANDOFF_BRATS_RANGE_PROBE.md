# Codex handoff: BraTS2024 range probe

**Status as of 2026-11-13:** The authenticated BraTS2024 archive probe is blocked on local Synapse authentication. Do not download the full archive.

## Verified

- The Synapse Data page `syn64952546` contains a BraTS-GLI folder; the official GLI training archive is `syn60086071`.
- The user's Chrome window was open to Synapse, but the browser-control provider did not expose its tabs. Native Chrome inspection showed no readable accessibility tree or screenshot. Do not attempt to bypass the account's verification step or inspect browser credentials/cookies.
- The official `synapseclient` package was installed into `/private/tmp/synapse-client-probe` after escalation approval. It was not installed in the repository environment.
- A metadata-only client attempt returned `SynapseNoCredentialsError` before making an HTTP request. No archive bytes were fetched. There is no Synapse config/token to use in this workspace; do not inspect credential files.
- The existing BraTS GLI fastlane package was already inspected and failed the project's segmentation-label validation. Do not reuse it or redownload it as training data.
- BraTS-Africa's official public processed NIfTI item is a 1.6 GB package. TCIA per-series APIs are documented for imaging downloads, but individual-subject retrieval from this packaged NIfTI item has not been verified. Do not download it under the current 250 MB case-pilot cap.
- No valid new cases, manifests, preprocessing run, or training sanity test were produced. The MSD baseline cases remain excluded from training.

## Next action

When the user can access Synapse again, ask them to create a local Synapse personal access token with view/download access and configure it locally with the Synapse client. Never ask them to paste the token into chat. If account verification remains unavailable, stop at this blocker; do not bypass it.

Then use the configured client to get metadata for `syn60086071` without downloading file content, obtain its temporary download URL in memory, and request exactly one byte from the archive tail (`Range: bytes=-1`). Print only the HTTP status, `Content-Range`, and `Accept-Ranges`; never print the URL or raw exception. HTTP 206 verifies byte-range support at the tail. It does not itself parse the ZIP directory. If that succeeds, fetch only the minimal ZIP/ZIP64 tail metadata needed to locate the central directory, then estimate selected-entry transfer sizes before extracting any subjects. Do not fetch the full archive or exceed an approved transfer scope.

Example probe (run only after the user has configured local Synapse authentication):

```python
import requests
from synapseclient import Synapse
from synapseclient.api.file_services import get_file_handle_for_download

response = None
try:
    syn = Synapse()
    syn.login(silent=True)
    entity = syn.get("syn60086071", downloadFile=False)
    handle = get_file_handle_for_download(
        str(entity.dataFileHandleId), "syn60086071", synapse_client=syn
    )
    response = requests.get(
        handle["preSignedURL"],
        headers={"Range": "bytes=-1"},
        stream=True,
        timeout=30,
    )
    print(response.status_code)
    print(response.headers.get("Content-Range"))
    print(response.headers.get("Accept-Ranges"))
except Exception as error:
    print(type(error).__name__)
finally:
    if response is not None:
        response.close()
```

Do not print the signed URL or raw exceptions. The current temporary client path is `/private/tmp/synapse-client-probe`; it may not persist indefinitely.

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
