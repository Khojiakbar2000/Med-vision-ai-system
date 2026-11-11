import { chromium } from '@playwright/test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
const mode = process.env.DICOM_TEST_MODE ?? 'single';
const origin = process.env.DICOM_TEST_ORIGIN ?? 'http://127.0.0.1:5180';
const fixture = mkdtempSync('/private/tmp/dicom-synthetic-');
execFileSync(process.env.DICOM_TEST_PYTHON ?? 'python3', ['tests/browser/generate_fixture.py',fixture, mode==='reject'?'multi':mode]);
let selectedFixture=fixture;
if (mode === 'scrubbed') {
  selectedFixture=fixture+'-scrubbed';
  execFileSync(process.env.DICOM_TEST_PYTHON ?? 'python3', ['tools/deidentify_audit.py','create-copy',fixture,selectedFixture]);
  execFileSync(process.env.DICOM_TEST_PYTHON ?? 'python3', ['tools/deidentify_audit.py','audit',fixture,selectedFixture]);
}
const browser = await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless:true, args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page = await browser.newPage({viewport:{width:1440,height:1100}});
let externalRequests = 0, leaked=false;
page.on('console',message=>{if(message.text().includes('SYNTHETIC_PRIVACY_SENTINEL')) leaked=true;});
await page.route('**/*', route => {
  const url = new URL(route.request().url());
  if (!['127.0.0.1', 'localhost'].includes(url.hostname) && !['blob:', 'data:'].includes(url.protocol)) { externalRequests++; return route.abort(); }
  return route.continue();
});
let pageErrors=0; page.on('pageerror', () => pageErrors++);
try {
  await page.goto(origin+(mode==='multi'?'?enhanced=1':mode==='stack'?'?preview=stack':''));
  await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('Ready.'),null, {timeout:60000});
  assert.equal(await page.locator('#window-button').isDisabled(),true);
  const choosing=page.waitForEvent('filechooser');
  await page.locator('#folder-button').click();
  await (await choosing).setFiles(selectedFixture);
  await page.waitForFunction(()=> document.querySelector('#status').dataset.state==='error' || document.querySelector('#status').textContent.includes('Loaded'),null, {timeout:60000});
  const status = await page.locator('#status').textContent();
  if(mode==='reject' || mode==='malformed' || mode==='missinguid') {
    assert.ok(status.includes(mode==='reject'?'multi-frame':mode==='missinguid'?'missing their Series Instance UID':'No DICOM images'));
    assert.equal(await page.locator('#folder-button').isDisabled(),false);
    assert.equal(pageErrors,0);
    console.log('PASS: unsupported/malformed/missing-UID input safely rejected with retry available.');
    await browser.close(); rmSync(fixture,{recursive:true,force:true}); process.exit(0);
  }
  assert.match(status,mode==='stack'?/Loaded 1 slices/:/Loaded 16 slices/,'Synthetic series must decode and load');
  await page.waitForTimeout(2000);
  for(const id of (mode==='stack'?['axial']:['axial','sagittal','coronal','volume-3d'])) {
    assert.ok(await page.locator(`#${id} canvas`).count(),`${id} canvas required`);
    const screenshot=await page.locator(`#${id}`).screenshot();
    const visible=await page.evaluate(async base64 => {
      const image=new Image(); image.src='data:image/png;base64,'+base64; await image.decode();
      const canvas=document.createElement('canvas'); canvas.width=image.width; canvas.height=image.height;
      const context=canvas.getContext('2d'); context.drawImage(image,0,0);
      const data=context.getImageData(0,0,canvas.width,canvas.height).data;
      let lit=0; const shades=new Set();
      for(let i=0;i<data.length;i+=4) { if(data[i]+data[i+1]+data[i+2]>30) lit++; shades.add(data[i]); }
      return lit>100 && shades.size>3;
    },screenshot.toString('base64'));
    assert.ok(visible,`${id} must contain visible image pixels`);
  }
  if(mode==='production') {
    assert.equal(pageErrors,0); assert.equal(externalRequests,0); assert.equal(leaked,false);
    await page.screenshot({path:'/private/tmp/dicom-production-proof.png'});
    console.log('PASS: production bundle decodes and visibly renders all four viewports.');
    await browser.close(); rmSync(fixture,{recursive:true,force:true}); process.exit(0);
  }
  if(mode==='stack') {
    assert.equal(await page.locator('#sagittal canvas').count(),0);
    await page.screenshot({path:'/private/tmp/dicom-stack-proof.png'});
    console.log('PASS: isolated synthetic STACK slice visibly rendered through folder button chooser.');
    await browser.close(); rmSync(fixture,{recursive:true,force:true}); process.exit(0);
  }
  const source = await (await page.request.get(origin+'/src/services/local-viewer.js')).text();
  const corePath = source.match(/from "([^"]*cornerstonejs_core[^"]*)"/)[1];
  const info = await page.evaluate(async path => {
    const core = await import(path);
    const engine = core.getRenderingEngine('private-dicom-rendering-engine');
    const volume = core.cache.getVolume('cornerstoneStreamingImageVolume:private-local-volume');
    const viewport=engine.getViewport('AXIAL');
    const center=viewport.getCamera().focalPoint;
    return { dimensions:volume.dimensions, spacing:volume.spacing, endpoints:[viewport.worldToCanvas(center),viewport.worldToCanvas([center[0]+20,center[1],center[2]])], camera:viewport.getCamera() };
  },corePath);
  assert.deepEqual(info.dimensions,[64,64,16]);
  assert.deepEqual(info.spacing,[1,1,2]);
  const oriented=await page.evaluate(async path=>{
    const core=await import(path); const engine=core.getRenderingEngine('private-dicom-rendering-engine');
    return [['AXIAL',2],['SAGITTAL',0],['CORONAL',1]].every(([id,axis])=>Math.abs(engine.getViewport(id).getCamera().viewPlaneNormal[axis])>0.999);
  },corePath);
  assert.ok(oriented,'All MPR cameras must align with the phantom patient axes');
  await page.locator('#ruler-button').click();
  const box=await page.locator('#axial').boundingBox();
  await page.mouse.move(box.x+info.endpoints[0][0],box.y+info.endpoints[0][1]);
  await page.mouse.down();
  await page.mouse.move(box.x+info.endpoints[1][0],box.y+info.endpoints[1][1],{steps:10});
  await page.mouse.up();
  await page.waitForTimeout(600);
  const mainSource = await (await page.request.get(origin+'/src/main.js')).text();
  const toolsPath=mainSource.match(/from "([^"]*cornerstonejs_tools[^"]*)"/)[1];
  const lengths=await page.evaluate(async path => {
    const tools=await import(path);
    return tools.annotation.state.getAnnotations('Length',document.querySelector('#axial')).flatMap(a=>Object.values(a.data.cachedStats).map(s=>s.length));
  },toolsPath);
  assert.ok(lengths.some(length=>Math.abs(length-20)<0.5),'Rendered length must measure 20 mm');
  const state=()=>page.evaluate(async path=>{
    const core=await import(path); const view=core.getRenderingEngine('private-dicom-rendering-engine').getViewport('AXIAL');
    return {camera:view.getCamera(),voi:view.getProperties().voiRange};
  },corePath);
  await page.locator('#window-button').click();
  let before=await state();
  const cx=box.x+box.width/2-60, cy=box.y+box.height/2-60;
  await page.mouse.move(cx,cy); await page.mouse.down(); await page.mouse.move(cx+30,cy+25,{steps:8}); await page.mouse.up();
  assert.notDeepEqual((await state()).voi,before.voi,'Window/level must change');
  before=await state();
  await page.mouse.move(cx,cy); await page.mouse.down({button:'middle'}); await page.mouse.move(cx+20,cy+10,{steps:8}); await page.mouse.up({button:'middle'});
  assert.notDeepEqual((await state()).camera.focalPoint,before.camera.focalPoint,'Pan must change camera');
  before=await state();
  await page.mouse.move(cx,cy); await page.mouse.down({button:'right'}); await page.mouse.move(cx,cy+40,{steps:8}); await page.mouse.up({button:'right'});
  assert.notEqual((await state()).camera.parallelScale,before.camera.parallelScale,'Zoom must change scale');
  before=await state();
  await page.mouse.move(cx,cy); await page.mouse.wheel(0,120); await page.waitForTimeout(250);
  assert.notDeepEqual((await state()).camera.focalPoint,before.camera.focalPoint,'Wheel must change slice position');
  assert.ok(!(await page.locator('body').textContent()).includes('SYNTHETIC_PRIVACY_SENTINEL'));
  assert.equal(leaked,false,'Console must not contain identifying metadata');
  const rotationBefore=await page.evaluate(async path=>(await import(path)).getRenderingEngine('private-dicom-rendering-engine').getViewport('VOLUME_3D').getCamera().position,corePath);
  const volumeBox=await page.locator('#volume-3d').boundingBox();
  await page.mouse.move(volumeBox.x+volumeBox.width/2,volumeBox.y+volumeBox.height/2); await page.mouse.down();
  await page.mouse.move(volumeBox.x+volumeBox.width/2+60,volumeBox.y+volumeBox.height/2+30,{steps:12}); await page.mouse.up();
  const rotationAfter=await page.evaluate(async path=>(await import(path)).getRenderingEngine('private-dicom-rendering-engine').getViewport('VOLUME_3D').getCamera().position,corePath);
  assert.notDeepEqual(rotationAfter,rotationBefore,'3D primary drag must rotate camera');
  const volumeBefore=await page.locator('#volume-3d').screenshot();
  const presetOptions=await page.locator('#preset-select option').evaluateAll(options=>options.map(option=>option.value));
  await page.locator('#preset-select').selectOption(presetOptions.find(value=>value.includes('MR')) ?? presetOptions.at(-1));
  await page.waitForTimeout(500);
  const volumeAfter=await page.locator('#volume-3d').screenshot();
  assert.ok(!volumeBefore.equals(volumeAfter),'Preset must change volume rendering');
  assert.equal(pageErrors,0);
  assert.equal(externalRequests,0);
  await page.screenshot({path:`/private/tmp/dicom-${mode}-proof.png`});
  console.log('PASS: synthetic MR decoded, four viewports contain image pixels, correct dimensions/spacing, drawn distance calibrated to 20 mm, no external requests.');
} catch (error) { console.log('Safe UI status:', await page.locator('#status').textContent()); throw error; } finally { await browser.close(); rmSync(fixture,{recursive:true,force:true}); if(selectedFixture!==fixture) rmSync(selectedFixture,{recursive:true,force:true}); }
