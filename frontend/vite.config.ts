import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'node:path';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    // Production build is followed by scripts/check-bundle-budget.mjs. Keep
    // this manifest as its stable input rather than coupling the check to
    // Vite's terminal formatting.
    manifest: true,
    chunkSizeWarningLimit: 500,
  },
  server: {
    port: 5173,
    // client.ts's resolveBaseUrl() falls back to a relative API_BASE_URL
    // whenever the browser's hostname isn't localhost/127.0.0.1 (e.g. a
    // forwarded LAN URL) — without a proxy, those relative /v1 and /health
    // requests hit Vite's own SPA fallback and get back this app's HTML
    // instead of the control plane's JSON.
    proxy: {
      '/v1': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/health': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
});
