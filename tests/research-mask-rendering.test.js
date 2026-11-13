import test from 'node:test';
import assert from 'node:assert/strict';
import { createResearchMaskVolumeInput } from '../src/core/research-mask-rendering.js';

test('3D prediction volume is explicitly visible and preserves discrete labels', () => {
  const colors = { name: 'color-transfer-function' };
  const opacity = { name: 'scalar-opacity' };
  const input = createResearchMaskVolumeInput('research-mask', colors, opacity);

  assert.equal(input.visibility, true);
  assert.equal(input.volumeId, 'research-mask');

  const calls = [];
  input.callback({
    volumeActor: {
      getProperty: () => ({
        setInterpolationTypeToNearest: () => calls.push(['nearest']),
        setRGBTransferFunction: (...args) => calls.push(['color', ...args]),
        setScalarOpacity: (...args) => calls.push(['opacity', ...args]),
      }),
    },
  });

  assert.deepEqual(calls, [
    ['nearest'],
    ['color', 0, colors],
    ['opacity', 0, opacity],
  ]);
});
