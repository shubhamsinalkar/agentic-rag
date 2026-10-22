import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  build: {
    // FastAPI serves this folder. The built files are committed, so the
    // Render deploy stays Python-only (no Node needed there).
    outDir: '../app/static',
    emptyOutDir: true,
  },
  server: {
    // `pnpm dev` talks to the Python API running on :8000.
    proxy: { '/ask': 'http://127.0.0.1:8000', '/health': 'http://127.0.0.1:8000' },
    // The benchmark chart is imported from ../docs (single source of truth).
    fs: { allow: ['..'] },
  },
})
