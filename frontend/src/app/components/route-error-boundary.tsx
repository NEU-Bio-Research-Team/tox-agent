import { isRouteErrorResponse, useRouteError } from 'react-router';
import { Footer } from './footer';
import { Navbar } from './navbar';
import { Button } from './ui/button';

function getErrorDescription(error: unknown): { title: string; detail: string; status?: string } {
  if (isRouteErrorResponse(error)) {
    const status = `${error.status} ${error.statusText}`.trim();
    const detail =
      typeof error.data === 'string'
        ? error.data
        : 'The page encountered an unexpected routing error.';

    return {
      title: 'Page failed to load',
      detail,
      status,
    };
  }

  if (error instanceof Error) {
    return {
      title: 'Unexpected application error',
      detail: error.message || 'An unexpected error occurred while rendering this page.',
    };
  }

  return {
    title: 'Unexpected application error',
    detail: 'An unknown error occurred while rendering this page.',
  };
}

export function RouteErrorBoundary() {
  const routeError = useRouteError();
  const description = getErrorDescription(routeError);

  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg)', fontFamily: 'Inter, sans-serif' }}>
      <Navbar />

      <main className="mx-auto max-w-3xl px-6 py-16">
        <section
          className="rounded-xl p-6"
          style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}
        >
          <p className="mb-2 text-xs uppercase tracking-wider" style={{ color: 'var(--text-faint)' }}>
            Runtime fallback
          </p>

          <h1 className="mb-2 text-2xl font-bold" style={{ color: 'var(--text)' }}>
            {description.title}
          </h1>

          {description.status && (
            <p className="mb-3 text-sm font-semibold" style={{ color: 'var(--accent-red)' }}>
              {description.status}
            </p>
          )}

          <p className="mb-6 text-sm" style={{ color: 'var(--text-muted)' }}>
            {description.detail}
          </p>

          <div className="flex flex-wrap gap-3">
            <Button
              type="button"
              onClick={() => window.location.reload()}
              style={{ backgroundColor: 'var(--accent-blue)', color: '#fff' }}
            >
              Reload page
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                window.location.href = '/analyze';
              }}
              style={{ borderColor: 'var(--border)', color: 'var(--text)' }}
            >
              Go to analysis page
            </Button>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
