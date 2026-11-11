import { logging } from '@cornerstonejs/utils';
import dicomImageLoader, { init as dicomImageLoaderInit } from '@cornerstonejs/dicom-image-loader';
import { CONSTANTS, Enums, RenderingEngine, imageLoader, init as coreInit, setVolumesForViewports, volumeLoader } from '@cornerstonejs/core';
import { Enums as ToolEnums, LengthTool, StackScrollTool, PanTool, ZoomTool, ToolGroupManager, TrackballRotateTool, WindowLevelTool, addTool, init as toolsInit } from '@cornerstonejs/tools';
import { chooseLargestCompatibleSeries, parseCandidate, expandFrames, validateVolumeGeometry, SUPPORTED_TRANSFER_SYNTAXES, SeriesSelectionError } from '../core/dicom-series.js';

const ENGINE_ID = 'private-dicom-rendering-engine';
const VOLUME_ID = 'cornerstoneStreamingImageVolume:private-local-volume';
const MPR_IDS = ['AXIAL', 'SAGITTAL', 'CORONAL'];
const VOLUME_ID_VIEW = 'VOLUME_3D';
const VIEW_IDS = [...MPR_IDS, VOLUME_ID_VIEW];
const PRESETS = CONSTANTS.VIEWPORT_PRESETS;

export class LocalDicomViewer {
  #engine;
  #mprTools;
  #volumeReady = false;
  #controls;
  #preview;
  #enhanced;

  constructor(controls, { preview = false, enhanced = false } = {}) { this.#controls = controls; this.#preview = preview; this.#enhanced = enhanced; }

  async initialize() {
    await coreInit();
    await dicomImageLoaderInit({ maxWebWorkers: 2, useLegacyMetadataProvider: true });
    await toolsInit();
    // Disable library diagnostics before any selected medical files are processed.
    logging.log.setLevel('silent');
    Object.values(logging.log.getLoggers()).forEach(logger => logger.setLevel('silent'));
    [WindowLevelTool, LengthTool, StackScrollTool, PanTool, ZoomTool, TrackballRotateTool].forEach(addTool);
    this.#engine = new RenderingEngine(ENGINE_ID);
    this.#engine.setViewports([
      ['AXIAL', Enums.OrientationAxis.AXIAL, 'axial'],
      ['SAGITTAL', Enums.OrientationAxis.SAGITTAL, 'sagittal'],
      ['CORONAL', Enums.OrientationAxis.CORONAL, 'coronal'],
      [VOLUME_ID_VIEW, Enums.OrientationAxis.CORONAL, 'volume-3d'],
    ].filter((_, index) => !this.#preview || index === 0).map(([viewportId, orientation, elementId], index) => ({
      viewportId,
      type: this.#preview ? Enums.ViewportType.STACK : index === 3 ? Enums.ViewportType.VOLUME_3D : Enums.ViewportType.ORTHOGRAPHIC,
      element: document.getElementById(elementId),
      defaultOptions: { orientation, background: [0, 0, 0] },
    })));
    new ResizeObserver(() => this.#engine.resize(true, true)).observe(document.querySelector(".viewport-grid"));
    document.querySelectorAll(".viewport").forEach(element => element.addEventListener("contextmenu", event => event.preventDefault()));
    if (!this.#preview) this.#configureTools();
    this.#populatePresets();
  }

  async loadFiles(files, onProgress = () => {}) {
    if (this.#volumeReady) throw new Error('Reload the page before choosing a different folder.');
    if (files.length > 4096 || files.reduce((size, file) => size + file.size, 0) > 1024 ** 3) throw new SeriesSelectionError('VOLUME_LIMIT');
    const candidates = [];
    const skipped = { MISSING_SERIES_UID: 0, MISSING_DIMENSIONS: 0, MALFORMED_DICOM: 0 };
    for (let i = 0; i < files.length; i += 1) {
      const candidate = await parseCandidate(files[i], (reason) => { skipped[reason] += 1; });
      if (candidate) candidates.push(...(this.#enhanced ? expandFrames(candidate) : [candidate]));
      if ((i + 1) % 25 === 0 || i + 1 === files.length) onProgress('headers', i + 1, files.length);
    }
    if (!candidates.length && skipped.MISSING_SERIES_UID) throw new SeriesSelectionError('MISSING_SERIES_UID');
    const series = this.#preview ? candidates.filter(candidate => candidate.numberOfFrames === 1).slice(0, 1) : chooseLargestCompatibleSeries(candidates);
    if (!series.length) throw new SeriesSelectionError('NO_DICOM_IMAGES');
    if (!this.#preview) validateVolumeGeometry(series);
    this.#selectModalityPreset(series[0].modality);
    if (series.some(s => !SUPPORTED_TRANSFER_SYNTAXES.has(s.transferSyntax))) throw new SeriesSelectionError('UNSUPPORTED_SYNTAX');
    const registeredFiles = new Map();
    const imageIds = series.map(({ file, frameNumber }) => {
      if (!registeredFiles.has(file)) registeredFiles.set(file, dicomImageLoader.wadouri.fileManager.add(file));
      const id = registeredFiles.get(file);
      return frameNumber ? `${id}&frame=${frameNumber}` : id;
    });
    const imageLoads = imageLoader.loadAndCacheImages(imageIds);
    let decodedImages = 0;
    onProgress('decode', decodedImages, imageIds.length);
    let timeout;
    try { await Promise.race([Promise.all(imageLoads.map(async (imageLoad) => {
      await imageLoad;
      decodedImages += 1;
      onProgress('decode', decodedImages, imageIds.length);
    })), new Promise((_, reject) => timeout = setTimeout(() => reject(new SeriesSelectionError('DECODE_TIMEOUT')), 60000))]); } finally { clearTimeout(timeout); }
    if (this.#preview) {
      await this.#engine.getViewport('AXIAL').setStack(imageIds);
      this.#engine.renderViewports(['AXIAL']);
      this.#updateStats(series);
      this.#volumeReady = true;
      return series.length;
    }
    const volume = await volumeLoader.createAndCacheVolume(VOLUME_ID, { imageIds });
    volume.load();
    await setVolumesForViewports(this.#engine, [{ volumeId: VOLUME_ID }], VIEW_IDS);
    this.#engine.getViewport(VOLUME_ID_VIEW).setProperties({ preset: this.#controls.preset.value });
    this.#engine.renderViewports(VIEW_IDS);
    this.#updateStats(series);
    this.#volumeReady = true;
    const { MouseBindings } = ToolEnums;
    this.#mprTools.setToolActive(WindowLevelTool.toolName, { bindings: [{ mouseButton: MouseBindings.Primary }] });
    this.#mprTools.setToolActive(StackScrollTool.toolName, { bindings: [{ mouseButton: MouseBindings.Wheel }] });
    this.#mprTools.setToolActive(PanTool.toolName, { bindings: [{ mouseButton: MouseBindings.Auxiliary }] });
    this.#mprTools.setToolActive(ZoomTool.toolName, { bindings: [{ mouseButton: MouseBindings.Secondary }] });
    this.#controls.windowLevel.disabled = false;
    this.#controls.distance.disabled = false;
    this.#controls.preset.disabled = false;
    return series.length;
  }

  setPreset(name) {
    if (!this.#volumeReady) return;
    const viewport = this.#engine.getViewport(VOLUME_ID_VIEW);
    viewport.setProperties({ preset: name });
    viewport.render();
  }

  selectMprTool(toolClass, button) {
    if (!this.#volumeReady) return;
    const { MouseBindings } = ToolEnums;
    [WindowLevelTool, LengthTool].forEach((tool) => this.#mprTools.setToolPassive(tool.toolName, { removeAllBindings: true }));
    this.#mprTools.setToolActive(toolClass.toolName, { bindings: [{ mouseButton: MouseBindings.Primary }] });
    this.#controls.windowLevel.classList.toggle('active', button === this.#controls.windowLevel);
    this.#controls.distance.classList.toggle('active', button === this.#controls.distance);
  }

  #configureTools() {
    const { MouseBindings } = ToolEnums;
    this.#mprTools = ToolGroupManager.createToolGroup('MPR_TOOL_GROUP');
    this.#mprTools.addTool(WindowLevelTool.toolName);
    this.#mprTools.addTool(LengthTool.toolName, { volumeId: VOLUME_ID });
    this.#mprTools.addTool(StackScrollTool.toolName);
    this.#mprTools.addTool(PanTool.toolName);
    this.#mprTools.addTool(ZoomTool.toolName);
    MPR_IDS.forEach((id) => this.#mprTools.addViewport(id, ENGINE_ID));
    this.#mprTools.setToolPassive(WindowLevelTool.toolName);
    this.#mprTools.setToolPassive(LengthTool.toolName);
    this.#mprTools.setToolPassive(StackScrollTool.toolName);
    const volumeTools = ToolGroupManager.createToolGroup('VOLUME_TOOL_GROUP');
    volumeTools.addTool(TrackballRotateTool.toolName);
    volumeTools.addViewport(VOLUME_ID_VIEW, ENGINE_ID);
    volumeTools.setToolActive(TrackballRotateTool.toolName, { bindings: [{ mouseButton: MouseBindings.Primary }] });
  }

  #populatePresets() {
    for (const preset of PRESETS) {
      const option = document.createElement('option');
      option.value = preset.name;
      option.textContent = preset.name;
      this.#controls.preset.append(option);
    }
    const names = PRESETS.map(({ name }) => name);
    this.#controls.preset.value = names.includes('CT-Bone') ? 'CT-Bone' : names[0] ?? '';
  }

  #selectModalityPreset(modality) {
    // CT transfer functions make ordinary MR anatomy nearly invisible. Keep
    // the chosen preset and selector in sync when the detected modality changes.
    const preferred = modality === 'MR' ? 'MR-Default' : 'CT-Bone';
    if ([...this.#controls.preset.options].some(option => option.value === preferred)) {
      this.#controls.preset.value = preferred;
    }
  }

  #updateStats(series) {
    const first = series[0];
    const spacing = first.pixelSpacing.length === 2 ? `${first.pixelSpacing.map((v) => v.toFixed(3)).join(' × ')} mm` : 'Not reported';
    const thickness = Number.isFinite(first.sliceThickness) ? `${first.sliceThickness.toFixed(3)} mm` : 'Not reported';
    document.getElementById('stat-modality').textContent = first.modality;
    document.getElementById('stat-slices').textContent = String(series.length);
    document.getElementById('stat-matrix').textContent = `${first.rows} × ${first.columns}`;
    document.getElementById('stat-spacing').textContent = spacing;
    document.getElementById('stat-thickness').textContent = thickness;
  }
}
