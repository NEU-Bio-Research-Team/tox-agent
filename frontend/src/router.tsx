import type { ComponentType } from 'react';
import { createBrowserRouter, redirect } from 'react-router';
import { RequireToken } from './features/auth/RequireToken';
import { RouteErrorBoundary } from './app/route-error-boundary';

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
    const mod = await import('./features/workbench/WorkbenchPage');
    workbenchRouteComponent = () => (
      <RequireToken>
        <mod.WorkbenchPage />
      </RequireToken>
    );
  }
  return { Component: workbenchRouteComponent };
}

async function loadAnalysisRoute() {
  const mod = await import('./features/quick-predict/QuickPredictPage');
  return {
    Component: () => (
      <RequireToken>
        <mod.QuickPredictPage />
      </RequireToken>
    ),
  };
}

export const router = createBrowserRouter([
  {
    path: '/',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const mod = await import('./features/landing/LandingPage');
      return { Component: mod.LandingPage };
    },
  },
  // Documentation is public and has its own frame. `/about` and `/documents`
  // were two pages of reading; they are now articles of the docs.
  {
    path: '/docs',
    errorElement: <RouteErrorBoundary />,
    loader: () => redirect('/docs/intro'),
  },
  {
    path: '/docs/:slug',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const mod = await import('./features/docs/DocsPage');
      return { Component: mod.DocsPage };
    },
  },
  { path: '/about', loader: () => redirect('/docs/principles') },
  { path: '/documents', loader: () => redirect('/docs/intro') },
  {
    path: '/sessions',
    errorElement: <RouteErrorBoundary />,
    lazy: async () => {
      const mod = await import('./features/sessions/SessionsPage');
      return {
        Component: () => (
          <RequireToken>
            <mod.SessionsPage />
          </RequireToken>
        ),
      };
    },
  },
  // The analysis page under both of its addresses: `/analyze` is the
  // production site's URL and the one the navigation bar uses; `/predict` is
  // the one this codebase's links and bookmarks already hold. One loader, so
  // there is one page.
  ...['/analyze', '/predict'].map((path) => ({
    path,
    errorElement: <RouteErrorBoundary />,
    lazy: loadAnalysisRoute,
  })),
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
      const mod = await import('./features/settings/SettingsPage');
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
