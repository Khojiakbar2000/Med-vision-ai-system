import { chromium } from '@playwright/test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

const root = process.cwd();
const apiPort = 8771;
const webPort = 5191;
const api = spawn(`${root}/.venv/bin/python`, ['-m', 'medvision_ai'], {
  cwd: `${root}/backend`,
  env: { ...process.env, MEDVISION_API_PORT: String(apiPort) },
  stdio: 'ignore',
});
const vite = spawn(`${root}/node_modules/.bin/vite`, ['--host', '127.0.0.1', '--port', String(webPort), '--strictPort'], {
  cwd: root,
  env: { ...process.env, MEDVISION_AI_URL: `http://127.0.0.1:${apiPort}` },
  stdio: 'ignore',
});
let browser;
try {
  let ready = false;
  for (let attempt = 0; attempt < 100; attempt++) {
    if (api.exitCode !== null || vite.exitCode !== null) throw new Error('A local test service stopped unexpectedly.');
    try {
      const response = await fetch(`http://127.0.0.1:${apiPort}/api/health`);
      if (response.ok) { ready = true; break; }
    } catch { /* wait for loopback services */ }
    await delay(250);
  }
  assert.ok(ready, 'FastAPI loopback service should be available');
  browser = await chromium.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  const externalRequests = [];
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (!['127.0.0.1', 'localhost'].includes(url.hostname) && !['blob:', 'data:'].includes(url.protocol)) {
      externalRequests.push(url.protocol);
      return route.abort();
    }
    return route.continue();
  });
  await page.goto(`http://127.0.0.1:${webPort}/`);
  await page.locator('#rag-heading').waitFor();
  await page.waitForFunction(() => document.querySelector('#rag-provider-disclosure').textContent.includes('stay on this computer'));
  await page.locator('[data-rag-question="What does the Dice score measure, and how is it different from IoU?"]').click();
  await page.locator('#rag-answer:not([hidden])').waitFor({ timeout: 10000 });
  assert.match(await page.locator('#rag-answer').textContent(), /intersection/i);
  assert.ok(await page.locator('#rag-answer a[href^="https://"]').count() > 0, 'Retrieved evidence should link to source documents');
  assert.match(await page.locator('#rag-status').textContent(), /retrieved sources/i);
  assert.match(await page.locator('#rag-context-status').textContent(), /No active segmentation context/);
  const makeResult = (score, failed = false) => ({
    schema: 'medvision-evaluation-results-v1',
    researchUseOnly: true,
    datasetProvenance: {
      dataset: { name: 'Synthetic test fixture', version: 'fixture-v1', license: 'synthetic', source: 'https://example.org/dataset', independenceAssessment: 'synthetic only' },
      model: { name: 'Synthetic test model', version: 'fixture', checkpointSha256: 'fixture-hash' },
      runtime: { device: 'cpu', modelLoadMs: 1 },
    },
    summary: { caseCount: failed ? 2 : 1, successfulCaseCount: 1, failedCaseCount: failed ? 1 : 0, metrics: { 'whole_tumor.dice': { n: 1, mean: score, median: score, stddev: 0, min: score, max: score } } },
    cases: [
      { case: 'case_000001', status: 'success', inferenceMs: 10, predictedWholeCm3: 1, predictedCoreCm3: 0.5, predictedEnhancingCm3: 0.2, metrics: Object.fromEntries(['whole_tumor', 'tumor_core', 'enhancing_tumor'].map(region => [region, { dice: score, iou: score / 2 }])) },
      ...(failed ? [{ case: 'case_000002', status: 'failed', errorCode: 'INVALID_GEOMETRY' }] : []),
    ],
  });
  await page.locator('#research-evaluation > summary').click();
  await page.locator('#research-baseline-file').setInputFiles({ name: 'baseline.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(makeResult(0.7, true))) });
  await page.locator('#research-candidate-file').setInputFiles({ name: 'candidate.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(makeResult(0.8))) });
  await page.waitForFunction(() => document.querySelector('#research-evaluation-content').textContent.includes('Synthetic test fixture'));
  assert.match(await page.locator('#research-evaluation-content').textContent(), /Synthetic test fixture/);
  assert.match(await page.locator('#research-evaluation-content').textContent(), /Failed · INVALID_GEOMETRY/);
  assert.match(await page.locator('#research-evaluation-content').textContent(), /Mean paired deltas/);
  assert.equal(externalRequests.length, 0, 'Viewer must make no external requests');
  assert.equal(errors.length, 0, 'Assistant UI should not raise page errors');
  console.log('PASS: browser assistant retrieved local evidence; local research dashboard rendered metrics, failure, and paired comparison without external requests.');
} finally {
  await browser?.close();
  vite.kill('SIGTERM');
  api.kill('SIGTERM');
}
