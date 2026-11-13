import { isRouteErrorResponse, useRouteError } from 'react-router';
import { Button } from '../shared/ui/button';
import { Logo } from '../shared/ui/logo';

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
    <div className="flex min-h-screen items-center justify-center bg-canvas-subtle px-6 py-12">
      <main className="w-full max-w-md space-y-6">
        <Logo withWordmark />
        <section className="space-y-4 rounded-xl border border-line bg-card p-6">
          <h1 className="text-2xl font-bold text-ink">{description.title}</h1>
          {description.status ? <p className="text-sm font-semibold text-danger">{description.status}</p> : null}
          <p className="text-sm text-ink-secondary">{description.detail}</p>
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
    </div>
  );
}
