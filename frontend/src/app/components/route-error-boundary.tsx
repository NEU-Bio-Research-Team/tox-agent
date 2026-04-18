import { isRouteErrorResponse, useRouteError } from 'react-router';

function formatError(error: unknown): { title: string; detail: string } {
  if (isRouteErrorResponse(error)) {
    return {
      title: `Request failed: ${error.status}`,
      detail: error.statusText || 'The page could not be loaded.',
    };
  }

  if (error instanceof Error) {
    return {
      title: 'Unexpected application error',
      detail: error.message || 'An unknown runtime error occurred.',
    };
  }

  return {
    title: 'Unexpected application error',
    detail: 'An unknown runtime error occurred.',
  };
}

export function RouteErrorBoundary() {
  const error = useRouteError();
  const { title, detail } = formatError(error);

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'grid',
        placeItems: 'center',
        backgroundColor: 'var(--bg)',
        padding: '24px',
      }}
    >
      <section
        style={{
          width: 'min(720px, 100%)',
          backgroundColor: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: '16px',
          padding: '24px',
        }}
      >
        <h1 style={{ color: 'var(--text)', fontSize: '1.5rem', fontWeight: 700, marginBottom: '12px' }}>
          {title}
        </h1>
        <p style={{ color: 'var(--text-muted)', marginBottom: '20px' }}>{detail}</p>
        <a
          href="/"
          style={{
            display: 'inline-block',
            padding: '10px 14px',
            borderRadius: '10px',
            backgroundColor: 'var(--accent-blue)',
            color: '#fff',
            fontWeight: 600,
            textDecoration: 'none',
          }}
        >
          Return to analysis page
        </a>
      </section>
    </div>
  );
}
