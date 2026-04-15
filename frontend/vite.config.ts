import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import packageJson from './package.json';

const appVersion = packageJson.version ?? '0.0.0';
const buildTime = new Date().toISOString();

export default defineConfig({
  define: {
    __APP_VERSION__: JSON.stringify(appVersion),
    __APP_BUILD_TIME__: JSON.stringify(buildTime),
  },
  plugins: [
    // The React and Tailwind plugins are both required for Make, even if
    // Tailwind is not being actively used – do not remove them
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      // Alias @ to the src directory
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    commonjsOptions: {
      // ketcher-core ships mixed ESM+CJS and contains require('raphael') in ESM files.
      // This ensures Rollup rewrites those requires for browser-safe output.
      transformMixedEsModules: true,
    },
    modulePreload: {
      resolveDependencies(_url, deps, context) {
        if (context.hostType === 'html') {
          return deps.filter((dep) => !dep.includes('ketcher-'));
        }
        return deps;
      },
    },
    rollupOptions: {
      output: {
        manualChunks(id): string | undefined {
          if (!id.includes('node_modules')) {
            return undefined;
          }

          if (id.includes('recharts')) {
            return 'charts';
          }

          if (
            id.includes('@radix-ui') ||
            id.includes('embla-carousel-react') ||
            id.includes('react-day-picker') ||
            id.includes('sonner') ||
            id.includes('next-themes')
          ) {
            return 'ui-vendor';
          }

          if (id.includes('react-router')) {
            return 'router';
          }

          if (id.includes('lucide-react')) {
            return 'icons';
          }

          if (
            id.includes('ketcher-react') ||
            id.includes('ketcher-standalone') ||
            id.includes('ketcher-core') ||
            id.includes('indigo-ketcher')
          ) {
            return 'ketcher';
          }

          return 'vendor';
        },
      },
    },
  },

  // File types to support raw imports. Never add .css, .tsx, or .ts files to this.
  assetsInclude: ['**/*.svg', '**/*.csv'],
});
