import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { pairedResearchDeltas, validateEvaluationResults } from '../src/core/research-evaluation.js';

const result = (score, caseId = 'case_000001') => ({
  schema: 'medvision-evaluation-results-v1',
  datasetProvenance: { dataset: { name: 'fixture' } },
  summary: { successfulCaseCount: 1, failedCaseCount: 0 },
  cases: [{
    case: caseId, status: 'success', inferenceMs: 10,
    metrics: Object.fromEntries(['whole_tumor', 'tumor_core', 'enhancing_tumor'].map(region => [region, { dice: score, iou: score / 2 }])),
  }],
});

test('validates result schema, region metrics, and rejects malformed results', () => {
  assert.equal(validateEvaluationResults(result(0.8)).cases.length, 1);
  assert.throws(() => validateEvaluationResults({ schema: 'other', cases: [] }), /INVALID_RESEARCH_RESULTS/);
  const bad = result(0.8); delete bad.cases[0].metrics.whole_tumor;
  assert.throws(() => validateEvaluationResults(bad), /INVALID_RESEARCH_RESULTS/);
  const unsafeAlias = result(0.8); unsafeAlias.cases[0].case = 'patient-identifier';
  assert.throws(() => validateEvaluationResults(unsafeAlias), /INVALID_RESEARCH_RESULTS/);
});

test('produces paired candidate-minus-baseline deltas only for matching successful aliases', () => {
  const base = result(0.7);
  const candidate = result(0.8);
  candidate.cases.push({ case: 'case_000002', status: 'failed' });
  const [paired] = pairedResearchDeltas(base, candidate);
  assert.equal(paired.case, 'case_000001');
  assert.ok(Math.abs(paired['whole_tumor.dice'] - 0.1) < 1e-12);
});

test('dashboard loads only a user-selected local evaluation JSON and marks study use as research-only', async () => {
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const app = await readFile(new URL('../src/main.js', import.meta.url), 'utf8');
  assert.match(html, /id="research-evaluation"/);
  assert.match(html, /id="research-baseline-file"/);
  assert.match(html, /no files leave this browser/i);
  assert.match(app, /await file\.text\(\)/);
  assert.match(app, /no file contents were transmitted/i);
  assert.match(html, /Independent validation and fine-tuned model improvement are not established/);
});
