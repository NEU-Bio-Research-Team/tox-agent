import type { ComponentType } from 'react';
import { createBrowserRouter } from 'react-router';
import { RequireToken } from './components/auth/RequireToken';
import { RouteErrorBoundary } from './components/route-error-boundary';

/**
 * Plan section 5.2: `/s/:sessionId` and its four artifact sub-routes must
 * render the *identical* Component reference so React reconciles navigation
 * between them as a params update on one element, not a remount — that's
 * what keeps the transcript, composer and SSE subscription alive while only
 * the artifacts selection changes. `router.lazy()` is called independently
 * per route object, so returning a fresh inline arrow function from each
 * call (as every other route below does) would give five *different*
 * closures wrapping the same page and defeat this. Caching the wrapped
 * component at module scope — resolved once, by whichever of the five
 * routes loads first — guarantees every route object gets back the exact
 * same function.
 */
let workbenchRouteComponent: ComponentType | null = null;

async function loadWorkbenchRoute() {
  if (!workbenchRouteComponent) {
    const mod = await import('./pages/WorkbenchPage');
    workbenchRouteComponent = () => (
      <RequireToken>
        <mod.WorkbenchPage />
      </RequireToken>
    );
  }
  return { Component: workbenchRouteComponent };
}

export const router = createBrowserRouter([
  {
    path: '/',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const mod = await import('./pages/LandingPage');
      return { Component: mod.LandingPage };
    },
  },
  {
    path: '/about',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const mod = await import('./pages/AboutPage');
      return { Component: mod.AboutPage };
    },
  },
  {
    path: '/sessions',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const mod = await import('./pages/SessionsPage');
      return {
        Component: () => (
          <RequireToken>
            <mod.SessionsPage />
          </RequireToken>
        ),
      };
    },
  },
  {
    path: '/predict',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const mod = await import('./pages/QuickPredictPage');
      return {
        Component: () => (
          <RequireToken>
            <mod.QuickPredictPage />
          </RequireToken>
        ),
      };
    },
  },
  ...['', '/runs/:runId', '/analyses/:analysisId', '/answers/:answerId', '/observations/:observationId', '/evidence/:evidenceId'].map(
    (suffix) => ({
      path: `/s/:sessionId${suffix}`,
      errorElement: <RouteErrorBoundary />,
      lazy: loadWorkbenchRoute,
    }),
  ),
  {
    path: '/settings',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const mod = await import('./pages/SettingsPage');
      return {
        Component: () => (
          <RequireToken>
            <mod.SettingsPage />
          </RequireToken>
        ),
      };
    },
  },
]);
