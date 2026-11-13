import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('research assistant UI exposes local questions, citations, context, and research-use notice', async () => {
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const app = await readFile(new URL('../src/main.js', import.meta.url), 'utf8');
  assert.match(html, /id="rag-form"/);
  assert.match(html, /id="rag-question"/);
  assert.match(html, /id="rag-answer"/);
  assert.match(html, /id="rag-provider-disclosure"/);
  assert.match(html, /LOCAL · SOURCE CITED/);
  assert.match(html, /Not for diagnosis or treatment decisions/);
  assert.match(app, /fetch\('\/api\/rag\/query'/);
  assert.match(app, /fetch\('\/api\/rag\/status'/);
  assert.match(app, /studyContextId: activeStudyContextId/);
  assert.match(app, /citation\.url\?\.startsWith\('https:\/\/'\)/);
});
