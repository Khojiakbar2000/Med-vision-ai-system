import { defineConfig } from 'vite';

export default defineConfig({
  // Keep the loader's module Worker URL visible to Vite's worker transform.
  optimizeDeps: { exclude: ['@cornerstonejs/dicom-image-loader'], include: ['@cornerstonejs/codec-libjpeg-turbo-8bit/decodewasmjs', '@cornerstonejs/codec-charls/decodewasmjs', '@cornerstonejs/codec-openjpeg/decodewasmjs', '@cornerstonejs/codec-openjph/wasmjs', '@cornerstonejs/codec-libjxl/decodewasmjs'] },
  worker: { format: 'es' },
  resolve: {
    // dcmjs' XML streaming helper extends Node EventEmitter. Force Vite to use
    // the browser polyfill instead of its browser-external builtin stub.
    alias: { events: 'events/' },
  },
});
