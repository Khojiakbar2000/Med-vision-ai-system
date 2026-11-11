import dicomParser from 'dicom-parser';

export class SeriesSelectionError extends Error {
  constructor(code) {
    super(code);
    this.name = 'SeriesSelectionError';
    this.code = code;
  }
}

/** Parse only the small set of geometry fields needed to choose a local stack. */
export async function parseCandidate(file, onIssue = () => {}) {
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const dataSet = dicomParser.parseDicom(bytes, { untilTag: 'x7fe00010' });
    const seriesUid = dataSet.string('x0020000e');
    const rows = dataSet.uint16('x00280010');
    const columns = dataSet.uint16('x00280011');
    if (!seriesUid) { onIssue('MISSING_SERIES_UID'); return null; }
    if (!rows || !columns) { onIssue('MISSING_DIMENSIONS'); return null; }

    const orientation = numbers(dataSet.string('x00200037'));
    const imagePosition = numbers(dataSet.string('x00200032'));
    const numberOfFrames = dataSet.intString('x00280008') ?? 1;
    return {
      file,
      transferSyntax: dataSet.string('x00020010'),
      frames: Array.from({ length: Math.min(numberOfFrames, 4096) }, (_, index) => {
        const perFrame = dataSet.elements.x52009230?.items?.[index]?.dataSet;
        const shared = dataSet.elements.x52009229?.items?.[0]?.dataSet;
        const value = (sequence, tag) => [perFrame, shared, dataSet].map(ds => ds?.elements[sequence]?.items?.[0]?.dataSet?.string(tag) ?? ds?.string(tag)).find(Boolean);
        return { frameNumber: index + 1, orientation: numbers(value('x00209116', 'x00200037')), imagePosition: numbers(value('x00209113', 'x00200032')), pixelSpacing: numbers(value('x00289110', 'x00280030')), sliceThickness: Number(value('x00289110', 'x00180050')) || undefined };
      }),
      seriesUid,
      frameOfReferenceUid: dataSet.string('x00200052') ?? '',
      orientation,
      imagePosition,
      instanceNumber: dataSet.intString('x00200013') ?? 0,
      modality: safeModality(dataSet.string('x00080060')),
      rows,
      columns,
      numberOfFrames,
      pixelSpacing: [dataSet.floatString('x00280030', 0), dataSet.floatString('x00280030', 1)].filter(Number.isFinite),
      sliceThickness: dataSet.floatString('x00180050'),
    };
  } catch {
    // Malformed/unrecognized files are ignored without exposing parser details.
    onIssue('MALFORMED_DICOM');
    return null;
  }
}

function numbers(value = '') {
  return value.trim() ? value.split('\\').map(Number).filter(Number.isFinite) : [];
}

function safeModality(value) {
  return value && /^[A-Z]{1,8}$/.test(value) ? value : 'Not reported';
}

function groupKey(candidate) {
  return [candidate.seriesUid, candidate.frameOfReferenceUid, candidate.orientation.join(','), `${candidate.rows}x${candidate.columns}`].join('|');
}

function projectedPosition(candidate) {
  if (candidate.orientation.length !== 6 || candidate.imagePosition.length !== 3) return null;
  const [rx, ry, rz, cx, cy, cz] = candidate.orientation;
  const normal = [ry * cz - rz * cy, rz * cx - rx * cz, rx * cy - ry * cx];
  return normal.reduce((sum, value, index) => sum + value * candidate.imagePosition[index], 0);
}

/** Select the largest compatible, single-frame stack without displaying private identifiers. */
export function chooseLargestCompatibleSeries(candidates) {
  if (!candidates.length) throw new SeriesSelectionError('NO_DICOM_IMAGES');
  const singleFrame = candidates.filter((candidate) => candidate.numberOfFrames === 1);
  if (!singleFrame.length) throw new SeriesSelectionError('MULTIFRAME_UNSUPPORTED');
  const groups = new Map();
  for (const candidate of singleFrame) {
    const key = groupKey(candidate);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(candidate);
  }
  const selected = [...groups.values()].sort((a, b) => b.length - a.length)[0];
  if (!selected || selected.length < 2) {
    throw new SeriesSelectionError('NO_MULTISLICE_SERIES');
  }
  selected.sort((a, b) => {
    const pa = projectedPosition(a);
    const pb = projectedPosition(b);
    if (pa !== null && pb !== null && pa !== pb) return pa - pb;
    return a.instanceNumber - b.instanceNumber;
  });
  return selected;
}

/** Expand enhanced instances only when every frame has explicit spatial metadata. */
export function expandFrames(candidate) {
  if (candidate.numberOfFrames === 1) return [candidate];
  if (candidate.numberOfFrames > 4096 || candidate.frames.some(frame => frame.imagePosition.length !== 3 || frame.orientation.length !== 6)) {
    throw new SeriesSelectionError('MULTIFRAME_GEOMETRY');
  }
  return candidate.frames.map(frame => ({ ...candidate, ...frame, numberOfFrames: 1 }));
}

export function validateVolumeGeometry(series) {
  if (series.some(s => s.orientation.length !== 6 || s.imagePosition.length !== 3 || s.pixelSpacing.length !== 2 || s.pixelSpacing.some(v => !Number.isFinite(v) || v <= 0))) throw new SeriesSelectionError('INVALID_GEOMETRY');
  const positions = series.map(projectedPosition);
  const gaps = positions.slice(1).map((value, i) => value - positions[i]);
  const spacing = gaps.reduce((a,b) => a+b, 0) / gaps.length;
  if (spacing <= 0 || gaps.some(gap => Math.abs(gap - spacing) > Math.max(0.01, spacing * 0.02))) throw new SeriesSelectionError('IRREGULAR_SERIES');
  const first = series[0];
  const r = first.orientation.slice(0,3), c = first.orientation.slice(3);
  if (Math.abs(Math.hypot(...r)-1)>0.001 || Math.abs(Math.hypot(...c)-1)>0.001 || Math.abs(r.reduce((sum,v,i)=>sum+v*c[i],0))>0.001) throw new SeriesSelectionError('INVALID_GEOMETRY');
  const normal = [r[1]*c[2]-r[2]*c[1],r[2]*c[0]-r[0]*c[2],r[0]*c[1]-r[1]*c[0]];
  if (series.some(s => {
    const delta=s.imagePosition.map((value,i)=>value-first.imagePosition[i]);
    const distance=delta.reduce((sum,value,i)=>sum+value*normal[i],0);
    return Math.hypot(...delta.map((value,i)=>value-distance*normal[i]))>0.01;
  })) throw new SeriesSelectionError('INVALID_GEOMETRY');
  if (series.some(s => s.pixelSpacing.some((v,i)=>Math.abs(v-first.pixelSpacing[i])>0.001))) throw new SeriesSelectionError('INVALID_GEOMETRY');
  return spacing;
}

export const SUPPORTED_TRANSFER_SYNTAXES = new Set([
  '1.2.840.10008.1.2', ...['1','2','1.99','1.98','8.1','5','4.50','4.51','4.53','4.55','4.57','4.70','4.80','4.81','4.90','4.91','4.201','4.202','4.203','4.110','4.111','4.112'].map(suffix=>`1.2.840.10008.1.2.${suffix}`),
]);
