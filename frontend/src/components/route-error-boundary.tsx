import { isRouteErrorResponse, useRouteError } from 'react-router';
import { Navbar } from './shell/Navbar';
import { Footer } from './shell/Footer';
import { Button } from './ui/button';

function describeError(error: unknown): { title: string; detail: string; status?: string } {
  if (isRouteErrorResponse(error)) {
    return {
      title: 'Không tải được trang',
      status: `${error.status} ${error.statusText}`.trim(),
      detail: typeof error.data === 'string' ? error.data : 'Đã xảy ra lỗi định tuyến không mong đợi.',
    };
  }
  if (error instanceof Error) {
    return { title: 'Lỗi ứng dụng', detail: error.message || 'Đã xảy ra lỗi không mong đợi.' };
  }
  return { title: 'Lỗi ứng dụng', detail: 'Đã xảy ra lỗi không xác định.' };
}

export function RouteErrorBoundary() {
  const description = describeError(useRouteError());

  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg)' }}>
      <Navbar />
      <main className="mx-auto max-w-3xl px-6 py-16">
        <section className="rounded-xl p-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
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
            <Button type="button" onClick={() => window.location.reload()}>
              Tải lại trang
            </Button>
            <Button type="button" variant="outline" onClick={() => { window.location.href = '/sessions'; }}>
              Về danh sách session
            </Button>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  );
}
