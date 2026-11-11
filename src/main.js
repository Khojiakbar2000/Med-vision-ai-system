import './style.css';
import { LengthTool, WindowLevelTool } from '@cornerstonejs/tools';
import { SeriesSelectionError } from './core/dicom-series.js';
import { LocalDicomViewer } from './services/local-viewer.js';

const $ = (id) => document.getElementById(id);
const folder = $('folder-input');
const folderButton = $('folder-button');
const status = $('status');
const controls = { windowLevel: $('window-button'), distance: $('ruler-button'), preset: $('preset-select') };
const parameters = new URLSearchParams(location.search);
const viewer = new LocalDicomViewer(controls, { preview: parameters.get('preview') === 'stack', enhanced: parameters.get('enhanced') === '1' });

folderButton.addEventListener('click', () => folder.disabled ? location.reload() : folder.click());

function setStatus(message, state = 'ready') {
  status.dataset.state = state;
  status.querySelector('.status-dot').hidden = state === 'ready';
  status.lastChild.textContent = ` ${message}`;
}

folder.addEventListener('change', async ({ target }) => {
  const files = [...target.files];
  if (!files.length) return;
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

controls.windowLevel.addEventListener('click', () => viewer.selectMprTool(WindowLevelTool, controls.windowLevel));
controls.distance.addEventListener('click', () => viewer.selectMprTool(LengthTool, controls.distance));
controls.preset.addEventListener('change', () => viewer.setPreset(controls.preset.value));

viewer.initialize()
  .then(() => { folderButton.disabled = false; setStatus('Ready. Choose one DICOM folder.'); })
  .catch(() => setStatus('Viewer initialization failed. Reload the page and try again.', 'error'));
