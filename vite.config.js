import { defineConfig } from 'vite';
import process from 'node:process';

export default defineConfig({
  // Keep the loader's module Worker URL visible to Vite's worker transform.
  optimizeDeps: { exclude: ['@cornerstonejs/dicom-image-loader'], include: ['@cornerstonejs/codec-libjpeg-turbo-8bit/decodewasmjs', '@cornerstonejs/codec-charls/decodewasmjs', '@cornerstonejs/codec-openjpeg/decodewasmjs', '@cornerstonejs/codec-openjph/wasmjs', '@cornerstonejs/codec-libjxl/decodewasmjs'] },
  worker: { format: 'es' },
  server: {
    watch: { ignored: ['**/.venv/**', '**/backend/.venv/**'] },
    proxy: { '/api': { target: process.env.MEDVISION_AI_URL ?? 'http://127.0.0.1:8765', changeOrigin: false } },
  },
  resolve: {
    // dcmjs' XML streaming helper extends Node EventEmitter. Force Vite to use
    // the browser polyfill instead of its browser-external builtin stub.
    alias: { events: 'events/' },
  },
});
