# MedVision Phase 3 — Medical Research Assistant

## Scope and status

This is an educational/research assistant attached to the existing MRI viewer. It explains segmentation methods, metrics, and validated measurements; it does not diagnose, recommend treatment, confirm tumors, or claim clinical accuracy. Phase 2 single-case technical acceptance remains finalized in [BRAIN_TUMOR_AI_PLAN.md](BRAIN_TUMOR_AI_PLAN.md).

The first complete Phase 3 slice is implemented locally: curated source ingestion, PDF/text extraction, source/topic-aware sentence-boundary chunking, sparse TF-IDF vector generation and cosine retrieval, citation-bound extractive answers, current-study context, and an integrated viewer panel. It uses the existing vanilla JavaScript/Vite frontend and FastAPI service. No React migration, hosted LLM call, external vector database, remote document fetch, or new background service is introduced.

## Architecture

```text
Curated local Markdown / text / PDF
          │ manifest + provenance
          ▼
   local ingestion → sentence-aware chunks → sparse TF-IDF vectors
                                                │ cosine retrieval
                                                ▼
User question + opaque study-context ID → local evidence answerer
                                                │
                    citations (manifest allowlist) + structured study values
                                                ▼
                       FastAPI loopback → Vite viewer assistant panel
```

- **Frontend:** `index.html`, `src/main.js`, and `src/style.css` add a research-oriented assistant panel. Suggested prompts cover WT/TC/ET, Dice/IoU, physical volume, and model limitations. The UI shows loading/error/insufficient-evidence states, source links, the fields of validated study context used, and a research-only notice. Current responses are returned as a complete answer; this local deterministic generator does not stream.
- **Backend:** `backend/medvision_ai/rag.py` owns ingestion, chunking, retrieval, evidence selection, and the generator interface. `POST /api/rag/query` validates the question and optional opaque context ID. Errors are returned in safe, fixed messages. The app remains loopback-only and rejects remote browser origins.
- **Ingestion and citations:** `backend/medvision_ai/rag_corpus/sources.json` is the source allowlist. Local `.md`, `.txt`, and `.pdf` files are supported; PDF text extraction uses pinned `pypdf==7.0.1`. The checked-in corpus contains short, original paraphrase notes with metadata and canonical source links; it does not contain copied article full text. Unknown citation IDs and non-HTTPS URLs are rejected. Citation URLs are rendered as links only after the UI verifies HTTPS.
- **Embedding/retrieval:** corpus-local sparse TF-IDF feature vectors plus cosine ranking are computed at process startup. This is a real vector representation and similarity search, but it is lexical rather than a neural semantic embedding model. The six-question evaluation fixture measures whether it is adequate for the current small corpus. Retrieval is in memory because six short notes do not justify PostgreSQL/pgvector operations or another service. No user scans or study identifiers enter the index.
- **Generation:** a provider-independent `Generator` protocol is implemented by `LocalEvidenceGenerator` and an optional OpenAI-compatible chat-completions adapter. The default is deterministic local extractive generation. The hosted adapter is inactive unless configured server-side; it requires `MEDVISION_RAG_PROVIDER=openai-compatible`, a server-side `MEDVISION_RAG_API_KEY`, `MEDVISION_RAG_MODEL`, and the explicit `MEDVISION_RAG_ALLOW_REMOTE_QUERIES=1` opt-in. It never receives raw images or identifiable DICOM metadata. Structured study context is withheld unless a separate `MEDVISION_RAG_ALLOW_REMOTE_STUDY_CONTEXT=1` opt-in is also set. API keys never reach the frontend. Hosted credentials may involve cost and query/evidence transfer; obtain explicit user approval before configuring or enabling them. Provider behavior is tested with a mocked transport; no real provider request or credential was used. A local generative-model download was not justified for this initial workflow.
- **Context:** `/api/segment` registers actual model name/version, required modalities, inference duration, and model-computed WT/TC/ET volumes in a capped in-memory registry. The random context ID contains no patient identifier, filename, affine, pixel data, or DICOM header. `/api/evaluate` updates the context only if the submitted prediction matches the registered prediction digest; missing reference metrics remain absent and are described as unavailable. Context expires when the service restarts or is evicted after the 100-entry cap.
- **Privacy:** images remain in the local viewer and local inference path and are never included in an LLM request. By default, question, evidence, and study context stay local. If an operator separately configures and explicitly enables the hosted adapter, user questions and retrieved source-note text are sent to that provider; structured study context remains withheld unless its separate flag is enabled. The UI discloses the active mode before a question is submitted. The hosted adapter never forwards image pixels, scan filenames, affine geometry, or DICOM identifiers. No remote document fetches, uploads, analytics, telemetry, or remote storage are used. Source links open only after a user click.

## Curated source set and provenance

The corpus is intentionally small and stores original paraphrase summaries, not source PDFs/full text.

| Source ID | Subject | Canonical source | Rights handling |
|---|---|---|---|
| `brats-regions` | BraTS release evolution, regional definitions, and protocol limits | [Review of BraTS datasets, 2012–2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11945730/) | CC BY 4.0; the local note is an attributed summary, not copied full text. |
| `monai-bundle` | Four-channel bundle protocol and documented limitations | [MONAI model-zoo bundle](https://github.com/Project-MONAI/model-zoo/tree/dev/models/brats_mri_segmentation) | Bundle code/weights identify Apache-2.0; citation points to official project documentation. |
| `msd-task01` | Task01 modalities, label mapping, and single-case limits | [MONAI Task01 dataset loader](https://github.com/Project-MONAI/MONAI/blob/dev/monai/apps/datasets.py) | Dataset terms are CC BY-SA 4.0 as recorded in the Phase 2 plan; only original summary text is included. |
| `dice-metric` | Dice overlap behavior and interpretation | [MONAI DiceMetric documentation](https://monai.readthedocs.io/en/latest/metrics.html) | Documentation is linked, not copied. |
| `jaccard-metric` | IoU/Jaccard definition | [scikit-learn Jaccard documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html) | Documentation is linked, not copied. |
| `nibabel-affine` | Affine and voxel-spacing basis for physical volume | [NiBabel affine documentation](https://nipy.org/nibabel/reference/nibabel.affines.html) | Documentation is linked, not copied. |

The ingestion utility reads local files and does not itself verify publication access or reuse rights, crawl links, or fetch source content. The bundled notes are curated summaries with recorded licenses/citations; anyone adding a source must first confirm its access and reuse terms and update the manifest. Retrieved content is treated as untrusted data; no source text is executed or allowed to alter system behavior. The local generator also filters instruction-like injected text.

## Study context semantics

The assistant distinguishes:

- **Observed protocol/context:** model name/version and required sequence names, supplied by the backend.
- **Model prediction:** predicted regional volumes and inference time, labeled as model output.
- **Ground-truth evaluation:** Dice/IoU values and the explicit dataset convention, included only after validated reference evaluation.
- **Retrieved evidence:** source passages summarized in answers and cited to the original documents.

When there is no reference mask, the assistant says metrics are unavailable; it never reports zero scores. When a context is absent or stale, general literature questions remain available, while a stale context ID receives a safe retry message.

## Verification and small RAG evaluation

Automated evaluation inputs are in `tests/fixtures/rag-evaluation.json`. `python tools/evaluate_rag.py` runs entirely offline against the bundled corpus. In the current run (six questions):

| Measure | Observed |
|---|---:|
| Supporting-source recall@3 | 0.9167 |
| Citation precision (citations are retrieved and allowlisted) | 1.0000 |
| Expected-fact grounding rate on the fixture | 1.0000 |
| Mean local query + answer latency | 0.410 ms |
| P95 local query + answer latency | 0.760 ms |

These values are a tiny deterministic corpus smoke check, not an external benchmark, clinical evaluation, or estimate of answer quality over open-ended medical questions. Latency is on the current development computer and excludes browser rendering. The test suite also covers ingestion (including a synthetic blank PDF), citation allowlisting, insufficient evidence, clinical-question refusal, prompt-injection text, extra request fields, safe context serialization, and absent ground-truth metrics. A browser integration test exercises the real local UI and FastAPI route, asserts source links, and verifies that it makes no external requests.

## Phase 3 roadmap

| Milestone | Status | Remaining work / acceptance |
|---|---|---|
| M1 — Audit and finalize Phase 2 | Complete | Verified single-case metrics, checkpoint, dataset provenance, preprocessing, latency, screenshots, and limitations are retained in the Phase 2 plan. |
| M2 — Minimal architecture | Complete | Existing Vite/FastAPI stack and in-process storage selected; no database or framework migration. |
| M3 — Local ingestion and retrieval | Complete for curated starter corpus | Extend only after new sources are license-checked; consider neural local embeddings only with a separately reviewed, sized model artifact. |
| M4 — Evidence answer generation | Complete for deterministic local answers; optional hosted adapter implemented but inactive | No credentials or remote provider call configured. Streaming and a local generative model remain future work. |
| M5 — Segmentation context | Complete for registered local predictions and validated evaluation | Context is ephemeral and numeric/model-only; restart requires rerunning analysis. |
| M6 — Viewer panel | Complete for initial workflow | Manual visual polish may follow user feedback; current controls and status/citation states are functional. |
| M7 — Testing and documentation | Complete for current local slice | Small offline fixture only; broaden questions and add blinded human review before treating answer quality as mature. |

### Known limitations

- Sparse TF-IDF cannot reliably handle paraphrases, long-tail terminology, or broad medical questions as well as a validated neural retriever.
- The deterministic extractive generator is deliberately conservative and may provide terse or incomplete answers; it is not a general medical language model.
- PDF support extracts embedded text only; scanned/image-only PDFs need a separately reviewed OCR pipeline and are not supported.
- The corpus contains six short curated source notes. It is not an exhaustive or continuously updated medical literature database.
- No hosted LLM credentials, local generative model, response streaming, reranker, persistent vector database, or multi-user context store is configured. The hosted adapter was not tested against a real provider.
- The Phase 2 underlying model validation remains one case; RAG explanations do not strengthen the segmentation evidence or validate clinical performance.
