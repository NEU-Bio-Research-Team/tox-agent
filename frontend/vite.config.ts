import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'node:path';
import { execSync } from 'node:child_process';
import { readFileSync } from 'node:fs';

const packageJson = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8')) as { version: string };

// The commit the bundle was built from, shown beside the version in the
// navigation bar. CI may pass it in; a local build asks git; a source tree
// with no git says so rather than failing the build.
function buildCommit(): string {
  const fromEnv = (process.env.VITE_APP_COMMIT ?? process.env.GITHUB_SHA ?? '').trim();
  if (fromEnv) return fromEnv.slice(0, 7);
  try {
    return execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
  } catch {
    return 'unknown';
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: {
    __APP_VERSION__: JSON.stringify(packageJson.version),
    __APP_BUILD_TIME__: JSON.stringify(new Date().toISOString()),
    __APP_COMMIT__: JSON.stringify(buildCommit()),
  },
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
