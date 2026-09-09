import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'node:path';

// Separate from vite.config.ts on purpose: the production build has no
// reason to know about jsdom or the test setup file, and this file has no
// reason to know about the dev-server proxy. Both share the same `@` alias
// and React plugin so a component under test resolves imports identically
// to how the app itself does.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  test: {
    // Playwright owns browser specs under e2e/. Without this boundary Vitest
    // discovers `*.spec.ts` there too and tries to execute Playwright's
    // global test API inside jsdom.
    include: ['./src/**/*.{test,spec}.{ts,tsx}'],
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
});
