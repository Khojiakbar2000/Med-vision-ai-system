import './style.css';
import { LengthTool, WindowLevelTool } from '@cornerstonejs/tools';
import { SeriesSelectionError } from './core/dicom-series.js';
import { pairedResearchDeltas, validateEvaluationResults } from './core/research-evaluation.js';
import { LocalDicomViewer } from './services/local-viewer.js';

const $ = (id) => document.getElementById(id);
const folder = $('folder-input');
const folderButton = $('folder-button');
const status = $('status');
const controls = { windowLevel: $('window-button'), distance: $('ruler-button'), preset: $('preset-select') };
const aiInput = $('ai-input');
const aiAssignment = $('ai-assignment');
const aiRunButton = $('ai-run-button');
const publicSampleButton = $('ai-public-sample-button');
const aiStatus = $('ai-status');
const aiResults = $('ai-results');
const aiRoles = ['t1c', 't1', 't2', 'flair'];
let selectedNiftiFiles = [];
let predictionMaskBase64 = '';
let activeStudyContextId = '';
const researchEvaluationResults = { baseline: null, candidate: null };
const parameters = new URLSearchParams(location.search);
const viewer = new LocalDicomViewer(controls, { preview: parameters.get('preview') === 'stack', enhanced: parameters.get('enhanced') === '1' });

if (import.meta.env.DEV && parameters.get('publicBrats') === '1') publicSampleButton.hidden = false;

folderButton.addEventListener('click', () => folder.disabled ? location.reload() : folder.click());

function setStatus(message, state = 'ready') {
  status.dataset.state = state;
  status.querySelector('.status-dot').hidden = state === 'ready';
  status.lastChild.textContent = ` ${message}`;
}

folder.addEventListener('change', async ({ target }) => {
  const files = [...target.files];
  if (!files.length) return;
  activeStudyContextId = '';
  updateRagContextStatus();
  folder.disabled = true;
  folderButton.disabled = true;
  setStatus(`Checking ${files.length} selected files in this browser…`, 'busy');
  try {
    const count = await viewer.loadFiles(files, (stage, done, total) => {
      setStatus(stage === 'headers'
        ? `Inspecting local headers: ${done} of ${total}`
        : `Decoding local images: ${done} of ${total}`, 'busy');
    });
    folderButton.textContent = 'Reload to choose another folder';
    folderButton.disabled = false;
    setStatus(`Loaded ${count} slices. This app did not upload scan data.`);
  } catch (error) {
    // Only a fixed category reaches the page or console; parser/loader text may contain PHI.
    const reason = error instanceof SeriesSelectionError ? error.code : '';
    const geometryErrors = { VOLUME_LIMIT: 'Selection exceeds the 4096-file or 1 GiB browser safety limit. Select one smaller series.', MISSING_SERIES_UID: 'These DICOM images are missing their Series Instance UID, which this viewer needs to group slices. If you selected a copy made by the tutorial deid command, it removes that UID and the copy cannot load here. Select the original series in the browser or create a compatibility-preserving copy in a separate trusted Terminal.', UNSUPPORTED_SYNTAX: 'This transfer syntax is not supported by the installed decoder libraries.', DECODE_TIMEOUT: 'Image decoding timed out. Check browser Worker support, available memory, and compression compatibility.', INVALID_GEOMETRY: 'Reconstruction requires valid image orientation, positions, and positive pixel spacing. Measurement is unavailable.', IRREGULAR_SERIES: 'The series contains duplicate positions or uneven slice gaps. Choose a complete, single spatial volume.', MULTIFRAME_GEOMETRY: 'This multi-frame object lacks explicit spatial metadata for each frame, or exceeds the 4096-frame limit. Cine and non-spatial objects cannot form a volume.' };
    const message = geometryErrors[reason] ?? (reason === 'NO_DICOM_IMAGES'
      ? 'No DICOM images were recognized. Choose the folder containing the DICOM series.'
      : reason === 'MULTIFRAME_UNSUPPORTED'
        ? 'This experimental viewer supports single-frame DICOM images; the selected images are multi-frame.'
        : reason === 'NO_MULTISLICE_SERIES'
          ? 'No compatible multi-slice series found. Choose at least two single-frame DICOM images from one series.'
          : 'Loading stopped. Check that the selected folder contains a supported DICOM series with a supported transfer syntax.');
    setStatus(message, 'error');
    folder.disabled = false;
    folderButton.disabled = false;
    folder.value = '';
  }
});

$('ai-select-button').addEventListener('click', () => aiInput.click());
publicSampleButton.addEventListener('click', async () => {
  let stage = 'Fetching the local public sample';
  publicSampleButton.disabled = true;
  $('ai-select-button').disabled = true;
  setAiStatus('Loading the locally staged public research sample and its real model result…', 'busy');
  try {
    const [flairResponse, resultResponse] = await Promise.all([
      fetch('/api/dev/public-brats/flair'),
      fetch('/api/dev/public-brats/result'),
    ]);
    if (!flairResponse.ok || !resultResponse.ok) throw new Error('The public development sample or its verified model result is unavailable.');
    const result = await resultResponse.json();
    activeStudyContextId = result.studyContextId || '';
    updateRagContextStatus();
    stage = 'Loading the original MRI in the viewer';
    const flair = new File([await flairResponse.blob()], 'flair.nii.gz', { type: 'application/gzip' });
    await viewer.loadResearchVolume(flair);
    stage = 'Loading the prediction mask';
    await viewer.showResearchSegmentation(result.mask, result.shape, result.spacingMm);
    stage = 'Rendering the prediction';
    renderAnalysisResult(result);
    if (result.metrics) renderMetrics(result.metrics);
    aiResults.hidden = false;
    $('ai-evaluation').hidden = true;
    setAiStatus(result.metrics
      ? `Displayed the real local prediction and its ${result.referenceConvention} reference comparison.`
      : 'Displayed the actual local model output. This development sample has no validated reference comparison.', 'ready');
  } catch (error) {
    setAiStatus(`${stage} failed. ${safeAiError(error instanceof Error ? error.message : '')}`, 'error');
  } finally {
    publicSampleButton.disabled = false;
    $('ai-select-button').disabled = false;
  }
});
aiInput.addEventListener('change', ({ target }) => {
  selectedNiftiFiles = [...target.files].filter(file => /\.nii(?:\.gz)?$/i.test(file.name));
  predictionMaskBase64 = '';
  activeStudyContextId = '';
  updateRagContextStatus();
  aiResults.hidden = true;
  if (selectedNiftiFiles.length < 4) {
    aiAssignment.hidden = true;
    setAiStatus('Select at least four NIfTI MRI volumes, then assign T1c, T1, T2, and FLAIR.', 'error');
    return;
  }
  for (const role of aiRoles) {
    const select = $(`ai-${role}`);
    select.replaceChildren(...selectedNiftiFiles.map((_, index) => {
      const option = document.createElement('option');
      option.value = String(index);
      option.textContent = `Volume ${index + 1}`;
      return option;
    }));
    const suggestedIndex = selectedNiftiFiles.findIndex(file => filenameMatchesRole(file.name, role));
    if (suggestedIndex >= 0) select.value = String(suggestedIndex);
  }
  aiAssignment.hidden = false;
  setAiStatus(`${selectedNiftiFiles.length} local NIfTI volumes selected. Confirm the four sequence assignments before analysis.`);
});

aiRunButton.addEventListener('click', async () => {
  const assignment = Object.fromEntries(aiRoles.map(role => [role, Number($(`ai-${role}`).value)]));
  if (new Set(Object.values(assignment)).size !== aiRoles.length) {
    setAiStatus('Assign a different volume to each of T1c, T1, T2, and FLAIR.', 'error');
    return;
  }
  const chosen = Object.fromEntries(aiRoles.map(role => [role, selectedNiftiFiles[assignment[role]]]));
  aiRunButton.disabled = true;
  $('ai-select-button').disabled = true;
  setAiStatus('Loading the selected FLAIR volume into the local viewer…', 'busy');
  try {
    await viewer.loadResearchVolume(chosen.flair);
    setAiStatus('Sending the four selected volumes to the loopback service for local preprocessing and inference…', 'busy');
    const body = new FormData();
    for (const role of aiRoles) {
      const file = chosen[role];
      body.append(role, file, `${role}${file.name.toLowerCase().endsWith('.nii.gz') ? '.nii.gz' : '.nii'}`);
    }
    const response = await fetch('/api/segment', { method: 'POST', body });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.detail || 'The local segmentation service rejected this study.');
    await viewer.showResearchSegmentation(result.mask, result.shape, result.spacingMm);
    predictionMaskBase64 = result.mask;
    activeStudyContextId = result.studyContextId || '';
    updateRagContextStatus();
    renderAnalysisResult(result);
    aiResults.hidden = false;
    $('ai-evaluation').hidden = false;
    setAiStatus('Prediction overlay rendered on the selected FLAIR volume. This is an unconfirmed research result.', 'ready');
  } catch (error) {
    setAiStatus(error instanceof TypeError
      ? 'Cannot reach the local inference service. Start it with “npm run ai:serve” after installing backend/requirements.txt.'
      : safeAiError(error.message), 'error');
  } finally {
    aiRunButton.disabled = false;
    $('ai-select-button').disabled = false;
  }
});

$('ai-ground-truth-button').addEventListener('click', () => {
  if (!$('ai-reference-convention').value) {
    setAiStatus('Select the reference mask label convention from its dataset documentation before evaluation.', 'error');
    return;
  }
  $('ai-ground-truth-input').click();
});
$('ai-ground-truth-input').addEventListener('change', async ({ target }) => {
  const reference = target.files?.[0];
  if (!reference || !predictionMaskBase64) return;
  if (!$('ai-reference-convention').value) {
    setAiStatus('Select the reference mask label convention from its dataset documentation before evaluation.', 'error');
    target.value = '';
    return;
  }
  setAiStatus('Comparing the prediction with the selected reference mask locally…', 'busy');
  try {
    const bytes = Uint8Array.from(atob(predictionMaskBase64), character => character.charCodeAt(0));
    const prediction = new File([bytes], 'prediction.nii.gz', { type: 'application/gzip' });
    const body = new FormData();
    body.append('prediction', prediction, 'prediction.nii.gz');
    body.append('reference', reference, 'reference.nii.gz');
    body.append('reference_convention', $('ai-reference-convention').value);
    if (activeStudyContextId) body.append('study_context_id', activeStudyContextId);
    const response = await fetch('/api/evaluate', { method: 'POST', body });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.detail || 'The selected reference mask could not be evaluated.');
    renderMetrics(result.metrics);
    setAiStatus('Reference comparison completed locally. Metrics describe only this selected research case.', 'ready');
  } catch (error) {
    setAiStatus(error instanceof TypeError ? 'Cannot reach the local evaluation service.' : safeAiError(error.message), 'error');
  }
});

const ragForm = $('rag-form');
const ragQuestion = $('rag-question');
const ragSubmit = $('rag-submit');
const ragStatus = $('rag-status');
const ragAnswer = $('rag-answer');
const ragSuggestionButtons = [...document.querySelectorAll('[data-rag-question]')];
function updateRagContextStatus() {
  $('rag-context-status').textContent = activeStudyContextId
    ? 'A validated local segmentation result is available as optional study context.'
    : 'No active segmentation context. General research questions are available.';
}
updateRagContextStatus();
fetch('/api/rag/status').then(async (response) => {
  if (!response.ok) throw new Error('unavailable');
  return response.json();
}).then((provider) => {
  if (provider.provider === 'openai-compatible') {
    $('rag-provider-badge').textContent = 'HOSTED · OPTED IN';
    $('rag-provider-disclosure').textContent = provider.remoteStudyContextEnabled
      ? 'Questions, retrieved notes, and numeric study results are sent to the configured hosted provider. MRI images and DICOM identifiers are never sent.'
      : 'Questions and retrieved notes are sent to the configured hosted provider. Study results and MRI images are not sent.';
  } else {
    $('rag-provider-disclosure').textContent = 'Questions, retrieval, and answers stay on this computer. No image or study context is sent to an external model.';
  }
  ragSubmit.disabled = false;
  ragSuggestionButtons.forEach(button => { button.disabled = false; });
}).catch(() => {
  $('rag-provider-disclosure').textContent = 'The local assistant service is not available; no question has been sent.';
});
ragForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = ragQuestion.value.trim();
  if (!question) {
    ragStatus.textContent = 'Enter a research question first.';
    return;
  }
  ragSubmit.disabled = true;
  ragStatus.textContent = 'Searching the local curated research library…';
  ragStatus.dataset.state = 'busy';
  ragAnswer.hidden = true;
  try {
    const response = await fetch('/api/rag/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, studyContextId: activeStudyContextId || null }),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.detail || 'The local research assistant could not answer this question.');
    ragAnswer.replaceChildren();
    const answer = document.createElement('p');
    answer.className = 'rag-answer-text';
    answer.textContent = result.answer;
    ragAnswer.append(answer);
    if (result.studyContextUsed?.length) {
      const used = document.createElement('p');
      used.className = 'rag-context-used';
      used.textContent = `Study results used: ${result.studyContextUsed.join(', ')}.`;
      ragAnswer.append(used);
    }
    if (result.citations?.length) {
      const sourceTitle = document.createElement('h4');
      sourceTitle.textContent = 'Retrieved sources';
      const sources = document.createElement('ol');
      sources.className = 'rag-citations';
      for (const citation of result.citations) {
        if (!citation.url?.startsWith('https://')) continue;
        const item = document.createElement('li');
        const link = document.createElement('a');
        link.href = citation.url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = citation.title;
        const license = document.createElement('span');
        license.textContent = citation.license;
        item.append(link, license);
        sources.append(item);
      }
      if (sources.childElementCount) ragAnswer.append(sourceTitle, sources);
    }
    ragStatus.textContent = result.evidenceStatus === 'insufficient'
      ? 'Insufficient evidence in the curated library. No source was cited.'
      : 'Answer assembled from retrieved sources and, when shown, validated local study results.';
    ragStatus.dataset.state = result.evidenceStatus === 'insufficient' ? 'error' : 'ready';
    ragAnswer.hidden = false;
  } catch (error) {
    ragStatus.textContent = error instanceof TypeError
      ? 'Cannot reach the local assistant service. Start it with “npm run ai:serve”.'
      : 'The local research assistant could not complete this question safely.';
    ragStatus.dataset.state = 'error';
  } finally {
    ragSubmit.disabled = false;
  }
});
ragSuggestionButtons.forEach((button) => {
  button.addEventListener('click', () => {
    ragQuestion.value = button.dataset.ragQuestion;
    ragForm.requestSubmit();
  });
});

function filenameMatchesRole(filename, role) {
  const stem = filename.toLowerCase().replace(/\.nii(?:\.gz)?$/i, '');
  const suffix = stem.split(/[_-]/).at(-1);
  if (role === 't1c') return ['t1c', 't1ce', 't1gd'].includes(suffix);
  if (role === 't1') return ['t1', 't1n'].includes(suffix);
  if (role === 't2') return ['t2', 't2w'].includes(suffix);
  return ['flair', 't2f'].includes(suffix);
}

function setAiStatus(message, state = 'ready') {
  aiStatus.dataset.state = state;
  aiStatus.textContent = message;
}

function safeAiError(message) {
  const safeShape = '(?:\\d+×\\d+×\\d+|unavailable)';
  const geometryDiagnostic = message.match(new RegExp(`^Prediction dimensions differ: expected ${safeShape}, mask ${safeShape}, displayed ${safeShape}\\.$`));
  if (geometryDiagnostic) return geometryDiagnostic[0];
  const safeMessages = [
    'Select one volume for each required MRI sequence: T1c, T1, T2, and FLAIR.',
    'Each input must be a 3D MRI volume with a supported spatial shape.',
    'An input contains invalid or non-finite image values.',
    'A volume has invalid spatial geometry.',
    'This model requires BraTS-preprocessed 1 mm isotropic MRI volumes.',
    'The four volumes are not co-registered on the same voxel grid. No prediction was produced.',
    'The selected volumes contain no nonzero image data.',
    'The selected research study exceeds the local processing size limit.',
    'An input is not a readable NIfTI volume.',
    'The local pretrained model weights are missing. Install the official checksum-verified MONAI bundle first.',
    'The local pretrained model weights could not be read.',
    'The local pretrained model weights failed integrity verification.',
    'The pretrained model could not be initialized with the installed local runtime.',
    'Local model inference failed. Check available memory and supported hardware.',
    'Prediction geometry does not match the displayed MRI volume. Overlay was stopped.',
    'The displayed MRI reference grid is unavailable. Overlay was stopped.',
    'Prediction dimensions do not match the displayed MRI grid. Overlay was stopped.',
    'Prediction spacing does not match the displayed MRI grid. Overlay was stopped.',
    'Prediction origin does not match the displayed MRI grid. Overlay was stopped.',
    'Prediction orientation does not match the displayed MRI grid. Overlay was stopped.',
    'Both evaluation inputs must be readable NIfTI masks.',
    'Prediction and reference masks do not share the same spatial grid.',
  ];
  return safeMessages.includes(message) ? message : 'Local analysis could not be completed. Check that the inputs meet the documented protocol.';
}

function renderAnalysisResult(result) {
  const volume = (mm3) => `${(mm3 / 1000).toFixed(2)} cm³`;
  const regions = result.volumes;
  aiResults.replaceChildren();
  const header = document.createElement('div');
  header.className = 'ai-result-heading';
  header.textContent = `${result.model.name} · version ${result.model.version} · ${result.model.device} · ${(result.elapsedMs / 1000).toFixed(1)} s`;
  aiResults.append(header);
  const overlayControls = document.createElement('div');
  overlayControls.className = 'ai-overlay-controls';
  const visibilityLabel = document.createElement('label');
  visibilityLabel.className = 'ai-overlay-toggle';
  const visibility = document.createElement('input');
  visibility.type = 'checkbox';
  visibility.checked = true;
  visibility.setAttribute('aria-label', 'Show segmentation overlay');
  visibility.addEventListener('change', () => viewer.setResearchSegmentationVisible(visibility.checked));
  visibilityLabel.append(visibility, document.createTextNode('Show segmentation overlay'));
  const opacityLabel = document.createElement('label');
  opacityLabel.className = 'ai-overlay-opacity';
  opacityLabel.htmlFor = 'ai-overlay-opacity';
  opacityLabel.textContent = 'Overlay opacity';
  const opacity = document.createElement('input');
  opacity.id = 'ai-overlay-opacity';
  opacity.type = 'range';
  opacity.min = '0.1';
  opacity.max = '1';
  opacity.step = '0.05';
  opacity.value = '0.65';
  opacity.setAttribute('aria-label', 'Segmentation overlay opacity');
  const opacityValue = document.createElement('output');
  opacityValue.value = '65%';
  opacity.addEventListener('input', () => {
    const value = Number(opacity.value);
    opacityValue.value = `${Math.round(value * 100)}%`;
    viewer.setResearchSegmentationOpacity(value);
  });
  opacityLabel.append(opacity, opacityValue);
  overlayControls.append(visibilityLabel, opacityLabel);
  aiResults.append(overlayControls);
  viewer.setResearchSegmentationOpacity(Number(opacity.value));
  const grid = document.createElement('div');
  grid.className = 'ai-result-grid';
  [
    ['Whole tumor', volume(regions.whole_tumor_mm3)],
    ['Tumor core', volume(regions.tumor_core_mm3)],
    ['Enhancing tumor', volume(regions.enhancing_tumor_mm3)],
    ['Input protocol', 'T1c · T1 · T2 · FLAIR'],
  ].forEach(([label, value]) => grid.append(resultItem(label, value)));
  aiResults.append(grid);
  const notes = document.createElement('p');
  notes.className = 'ai-limitations';
  notes.textContent = result.limitations.join(' ');
  aiResults.append(notes);
}

function renderMetrics(metrics) {
  const grid = $('ai-metric-grid');
  grid.replaceChildren();
  for (const [region, values] of Object.entries(metrics)) {
    const row = document.createElement('div');
    row.className = 'ai-metric-row';
    const title = document.createElement('strong');
    title.textContent = region.replaceAll('_', ' ');
    const summary = document.createElement('span');
    summary.textContent = `Dice ${values.dice.toFixed(3)} · IoU ${values.iou.toFixed(3)} · HD95 ${values.hausdorff95_mm === null ? 'N/A' : `${values.hausdorff95_mm.toFixed(2)} mm`}`;
    row.append(title, summary);
    grid.append(row);
  }
  $('ai-metrics').hidden = false;
}

function resultItem(label, value) {
  const item = document.createElement('div');
  item.className = 'ai-result-item';
  const title = document.createElement('span');
  title.textContent = label;
  const text = document.createElement('strong');
  text.textContent = value;
  item.append(title, text);
  return item;
}

for (const [slot, inputId] of [['baseline', 'research-baseline-file'], ['candidate', 'research-candidate-file']]) {
  $(inputId).addEventListener('change', async ({ target }) => {
    const file = target.files?.[0];
    if (!file) return;
    const statusNode = $('research-evaluation-status');
    try {
      if (file.size > 10 * 1024 * 1024) throw new TypeError('RESULT_FILE_TOO_LARGE');
      const parsed = validateEvaluationResults(JSON.parse(await file.text()));
      researchEvaluationResults[slot] = parsed;
      statusNode.textContent = 'Local result file validated. Its contents were not sent to the backend.';
      renderResearchEvaluation();
    } catch {
      researchEvaluationResults[slot] = null;
      statusNode.textContent = 'This file is not a valid MedVision evaluation result. No file contents were transmitted.';
      renderResearchEvaluation();
    } finally {
      target.value = '';
    }
  });
}

function renderResearchEvaluation() {
  const content = $('research-evaluation-content');
  content.replaceChildren();
  const loaded = Object.entries(researchEvaluationResults).filter(([, result]) => result);
  content.hidden = loaded.length === 0;
  for (const [slot, result] of loaded) {
    const block = document.createElement('section');
    block.className = 'research-result-block';
    const title = document.createElement('h3');
    title.textContent = `${slot === 'baseline' ? 'Baseline' : 'Candidate'} · ${result.datasetProvenance.dataset?.name || 'Research dataset'}`;
    block.append(title);
    const model = result.datasetProvenance.model || {};
    const provenance = document.createElement('p');
    provenance.textContent = `${model.name || 'Model'} ${model.version || ''} · ${result.datasetProvenance.runtime?.device || 'device not recorded'} · checkpoint ${model.checkpointSha256 || 'hash not recorded'}`;
    block.append(provenance);
    const datasetInfo = result.datasetProvenance.dataset || {};
    const source = document.createElement('p');
    source.textContent = `Dataset ${datasetInfo.version || 'version unspecified'} · ${datasetInfo.license || 'license unspecified'} · independence: ${datasetInfo.independenceAssessment || 'not established'} · source: ${datasetInfo.source || 'unspecified'}`;
    block.append(source);
    const summary = document.createElement('p');
    const totals = result.summary;
    const peakRss = result.datasetProvenance.runtime?.processPeakRssBytes;
    summary.textContent = `${totals.successfulCaseCount ?? 0} successful · ${totals.failedCaseCount ?? 0} failed · load ${(result.datasetProvenance.runtime?.modelLoadMs ?? 0).toFixed(0)} ms · peak process RSS ${Number.isFinite(peakRss) ? `${peakRss} bytes` : 'not measured'}`;
    block.append(summary);
    const aggregates = totals.metrics || {};
    const aggregateTable = document.createElement('table');
    aggregateTable.className = 'research-case-table';
    const aggregateHead = document.createElement('thead');
    aggregateHead.innerHTML = '<tr><th>Aggregate measure</th><th>n</th><th>Mean</th><th>Median</th><th>SD</th><th>Min</th><th>Max</th></tr>';
    aggregateTable.append(aggregateHead);
    const aggregateBody = document.createElement('tbody');
    for (const [measure, values] of Object.entries(aggregates)) {
      if (!values || !Number.isFinite(values.n)) continue;
      const tr = document.createElement('tr');
      for (const value of [measure, String(values.n), ...['mean', 'median', 'stddev', 'min', 'max'].map(key => Number.isFinite(values[key]) ? values[key].toFixed(4) : '—')]) {
        const cell = document.createElement('td'); cell.textContent = value; tr.append(cell);
      }
      aggregateBody.append(tr);
    }
    aggregateTable.append(aggregateBody);
    block.append(aggregateTable);
    const table = document.createElement('table');
    table.className = 'research-case-table';
    const head = document.createElement('thead');
    head.innerHTML = '<tr><th>Case alias</th><th>Status</th><th>WT Dice / IoU</th><th>TC Dice / IoU</th><th>ET Dice / IoU</th><th>Inference</th><th>Peak device memory</th><th>Volume WT / TC / ET cm³</th></tr>';
    table.append(head);
    const body = document.createElement('tbody');
    for (const row of result.cases) {
      const tr = document.createElement('tr');
      const values = row.status === 'success'
        ? [row.case, 'Success', ...['whole_tumor', 'tumor_core', 'enhancing_tumor'].map(region => `${row.metrics[region].dice.toFixed(4)} / ${row.metrics[region].iou.toFixed(4)}`), `${row.inferenceMs.toFixed(0)} ms`, Number.isFinite(row.peakDeviceMemoryBytes) ? `${row.peakDeviceMemoryBytes} bytes` : 'not measured', `${row.predictedWholeCm3?.toFixed(2) ?? '—'} / ${row.predictedCoreCm3?.toFixed(2) ?? '—'} / ${row.predictedEnhancingCm3?.toFixed(2) ?? '—'}`]
        : [row.case, `Failed · ${row.errorCode || 'unspecified'}`, '—', '—', '—', '—', '—', '—'];
      for (const value of values) {
        const cell = document.createElement('td');
        cell.textContent = value;
        tr.append(cell);
      }
      body.append(tr);
    }
    table.append(body);
    block.append(table);
    content.append(block);
  }
  if (researchEvaluationResults.baseline && researchEvaluationResults.candidate) {
    const deltas = pairedResearchDeltas(researchEvaluationResults.baseline, researchEvaluationResults.candidate);
    const note = document.createElement('p');
    note.className = 'research-delta-note';
    const failedOrUnpaired = Math.max(researchEvaluationResults.baseline.cases.length, researchEvaluationResults.candidate.cases.length) - deltas.length;
    note.textContent = `Candidate minus baseline: ${deltas.length} paired successful case(s), ${failedOrUnpaired} failed or unpaired. Positive metric deltas indicate a higher overlap score; they do not establish model superiority.`;
    content.append(note);
    if (deltas.length) {
      const aggregateDelta = document.createElement('p');
      const deltaFields = ['whole_tumor.dice', 'whole_tumor.iou', 'tumor_core.dice', 'tumor_core.iou', 'enhancing_tumor.dice', 'enhancing_tumor.iou'];
      const metricMeans = deltaFields.map(key => `${key} ${(deltas.reduce((sum, row) => sum + row[key], 0) / deltas.length).toFixed(4)}`);
      const meanLatency = deltas.reduce((sum, row) => sum + row.inferenceMs, 0) / deltas.length;
      const measuredMemory = deltas.map(row => row.peakDeviceMemoryBytes).filter(Number.isFinite);
      metricMeans.push(`inference ms ${meanLatency.toFixed(1)}`);
      if (measuredMemory.length) metricMeans.push(`device memory bytes ${Math.round(measuredMemory.reduce((sum, value) => sum + value, 0) / measuredMemory.length)}`);
      aggregateDelta.textContent = `Mean paired deltas: ${metricMeans.join(' · ')}`;
      content.append(aggregateDelta);
      const table = document.createElement('table');
      table.className = 'research-case-table';
      const header = document.createElement('thead');
      header.innerHTML = '<tr><th>Case alias</th><th>WT Δ Dice / IoU</th><th>TC Δ Dice / IoU</th><th>ET Δ Dice / IoU</th><th>Inference Δ ms</th><th>Device memory Δ bytes</th></tr>';
      table.append(header);
      const body = document.createElement('tbody');
      for (const row of deltas) {
        const tr = document.createElement('tr');
        for (const value of [row.case, ...['whole_tumor', 'tumor_core', 'enhancing_tumor'].map(region => `${row[`${region}.dice`].toFixed(4)} / ${row[`${region}.iou`].toFixed(4)}`), row.inferenceMs.toFixed(1), Number.isFinite(row.peakDeviceMemoryBytes) ? String(Math.round(row.peakDeviceMemoryBytes)) : 'N/A']) {
          const cell = document.createElement('td'); cell.textContent = value; tr.append(cell);
        }
        body.append(tr);
      }
      table.append(body); content.append(table);
    }
  }
}

controls.windowLevel.addEventListener('click', () => viewer.selectMprTool(WindowLevelTool, controls.windowLevel));
controls.distance.addEventListener('click', () => viewer.selectMprTool(LengthTool, controls.distance));
controls.preset.addEventListener('change', () => viewer.setPreset(controls.preset.value));

viewer.initialize()
  .then(() => { folderButton.disabled = false; setStatus('Ready. Choose one DICOM folder.'); })
  .catch(() => setStatus('Viewer initialization failed. Reload the page and try again.', 'error'));
