import test from 'node:test';
import assert from 'node:assert/strict';
import { parseCandidate, chooseLargestCompatibleSeries, SeriesSelectionError } from '../src/core/dicom-series.js';

function element(group, number, vr, value) {
  const data = typeof value === 'string' ? new TextEncoder().encode(value.padEnd(value.length + (value.length % 2), vr === 'UI' ? '\0' : ' ')) : new Uint8Array([value & 255, value >> 8]);
  const header = new Uint8Array(8);
  const view = new DataView(header.buffer);
  view.setUint16(0, group, true); view.setUint16(2, number, true);
  header[4] = vr.charCodeAt(0); header[5] = vr.charCodeAt(1); view.setUint16(6, data.length, true);
  return [header, data];
}

function syntheticDicom({ series = '1.2.3', frame = '1.2.4', instance = 1, position = 0, orientation = '1\\0\\0\\0\\1\\0', modality = 'CT', frames } = {}) {
  const tags = [
    element(0x0002, 0x0010, 'UI', '1.2.840.10008.1.2.1'),
    element(0x0008, 0x0060, 'CS', modality), element(0x0018, 0x0050, 'DS', '1.5'),
    element(0x0020, 0x000e, 'UI', series), element(0x0020, 0x0052, 'UI', frame),
    element(0x0020, 0x0013, 'IS', String(instance)), element(0x0020, 0x0032, 'DS', `0\\0\\${position}`),
    element(0x0020, 0x0037, 'DS', orientation), element(0x0028, 0x0010, 'US', 32),
    element(0x0028, 0x0011, 'US', 32), element(0x0028, 0x0030, 'DS', '0.7\\0.7'),
  ];
  if (frames !== undefined) tags.push(element(0x0028, 0x0008, 'IS', String(frames)));
  const bytes = new Uint8Array(132 + tags.reduce((sum, pair) => sum + pair.reduce((n, part) => n + part.length, 0), 0));
  bytes.set(new TextEncoder().encode('DICM'), 128);
  let offset = 132;
  for (const pair of tags) for (const part of pair) { bytes.set(part, offset); offset += part.length; }
  return new Blob([bytes]);
}
function namedBlob(blob) { Object.defineProperty(blob, 'name', { value: 'synthetic-test-fixture.dcm' }); return blob; }

test('parses only geometry and modality from a synthetic DICOM object', async () => {
  const candidate = await parseCandidate(namedBlob(syntheticDicom({ instance: 2, position: 4 })));
  assert.equal(candidate.modality, 'CT');
  assert.equal(candidate.rows, 32);
  assert.equal(candidate.columns, 32);
  assert.equal(candidate.instanceNumber, 2);
  assert.deepEqual(candidate.pixelSpacing, [0.7, 0.7]);
  assert.deepEqual(candidate.imagePosition, [0, 0, 4]);
  assert.equal('patientName' in candidate, false);
});

test('classifies missing series identity without exposing header values', async () => {
  const reasons = [];
  const candidate = await parseCandidate(namedBlob(syntheticDicom({ series: '' })), reason => reasons.push(reason));
  assert.equal(candidate, null);
  assert.deepEqual(reasons, ['MISSING_SERIES_UID']);
});

test('accepts a synthetic MR series without exposing identifying tags', async () => {
  const series = await Promise.all([0, 1].map((position) => parseCandidate(
    namedBlob(syntheticDicom({ modality: 'MR', instance: position + 1, position })),
  )));
  const selected = chooseLargestCompatibleSeries(series);
  assert.equal(selected.length, 2);
  assert.equal(selected[0].modality, 'MR');
  assert.deepEqual(selected.map((item) => item.imagePosition[2]), [0, 1]);
  assert.ok(selected.every((item) => !('patientName' in item)));
});

test('ignores malformed bytes and rejects incompatible or multi-frame candidates', async () => {
  const invalid = { arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer };
  assert.equal(await parseCandidate(invalid), null);
  const common = { seriesUid: 'series', frameOfReferenceUid: 'frame', orientation: [1, 0, 0, 0, 1, 0], rows: 32, columns: 32, numberOfFrames: 1, imagePosition: [0, 0, 0], instanceNumber: 1 };
  assert.throws(() => chooseLargestCompatibleSeries([{ ...common, numberOfFrames: 2 }]), (error) => error instanceof SeriesSelectionError && error.code === 'MULTIFRAME_UNSUPPORTED');
  assert.throws(() => chooseLargestCompatibleSeries([{ ...common, rows: 20 }, { ...common, rows: 30 }]), (error) => error instanceof SeriesSelectionError && error.code === 'NO_MULTISLICE_SERIES');
  assert.throws(() => chooseLargestCompatibleSeries([]), (error) => error instanceof SeriesSelectionError && error.code === 'NO_DICOM_IMAGES');
});

test('chooses the largest compatible series and sorts by physical position', () => {
  const candidate = (instanceNumber, imagePosition, seriesUid = 'dominant') => ({ seriesUid, frameOfReferenceUid: 'ref', orientation: [1, 0, 0, 0, 1, 0], rows: 32, columns: 32, numberOfFrames: 1, imagePosition: [0, 0, imagePosition], instanceNumber });
  const selected = chooseLargestCompatibleSeries([
    candidate(3, 8), candidate(1, 2), candidate(2, 5), candidate(1, 0, 'small'), candidate(2, 1, 'small'),
  ]);
  assert.deepEqual(selected.map((item) => item.imagePosition[2]), [2, 5, 8]);
  assert.ok(selected.every((item) => item.seriesUid === 'dominant'));
});

test('requires calibrated geometry and rejects missing or duplicate slices', async () => {
  const { validateVolumeGeometry, expandFrames } = await import('../src/core/dicom-series.js');
  const first = await parseCandidate(syntheticDicom({position:0}));
  const second = await parseCandidate(syntheticDicom({position:1.5}));
  assert.equal(validateVolumeGeometry([first,second]),1.5);
  assert.throws(()=>validateVolumeGeometry([first,{...second,pixelSpacing:[0,1]}]), /INVALID_GEOMETRY/);
  assert.throws(()=>validateVolumeGeometry([first,first]), /IRREGULAR_SERIES/);
  assert.throws(()=>validateVolumeGeometry([first,second,{...second,imagePosition:[0,0,9]}]), /IRREGULAR_SERIES/);
  assert.throws(()=>expandFrames({numberOfFrames:2,frames:[{imagePosition:[],orientation:[]}]}), /MULTIFRAME_GEOMETRY/);
});
