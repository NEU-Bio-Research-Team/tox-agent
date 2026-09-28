import { useEffect, useState } from 'react';
import { ImageOff } from 'lucide-react';
import { getReportFigure } from '../../../../shared/api/endpoints';
import type { ReportFigure as ReportFigureData } from '../../../../shared/api/types';

/**
 * REP-02: one figure, fetched with the session's bearer token.
 *
 * The artifact has carried `figures` and per-section `figure_ids` all along;
 * what it never had was a delivery path. An `<img src>` cannot carry an
 * authorization header, so the bytes are fetched here and handed to the DOM as
 * an object URL, which is revoked on unmount — a leaked one pins the whole SVG
 * in memory for as long as the tab lives.
 *
 * A failure shows the alt text rather than a broken-image glyph. The alt text is
 * written to say what the picture says (both contribution directions, and the
 * class they point at), so a reader who cannot see the figure still gets its
 * content instead of an apology.
 */
export function ReportFigure({
  sessionId,
  reportId,
  figure,
}: {
  sessionId: string;
  reportId: string;
  figure: ReportFigureData;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let objectUrl: string | null = null;
    let cancelled = false;
    setUrl(null);
    setFailed(false);
    void (async () => {
      try {
        const blob = await getReportFigure(sessionId, reportId, figure.figure_id);
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      } catch {
        if (!cancelled) setFailed(true);
      }
    })();
    return () => {
      cancelled = true;
      // Revoked on unmount and on every figure change, including the case where
      // the fetch resolved after the component went away.
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [sessionId, reportId, figure.figure_id]);

  return (
    <figure className="my-3 space-y-1.5">
      {failed ? (
        <div
          className="flex items-start gap-2 rounded-md border border-dashed p-3 text-sm"
          style={{ color: 'var(--text-muted)' }}
          role="img"
          aria-label={figure.alt_text}
        >
          <ImageOff size={16} className="mt-0.5 shrink-0" aria-hidden />
          <span>
            Không tải được hình. Nội dung hình: {figure.alt_text}
          </span>
        </div>
      ) : url ? (
        <img
          src={url}
          alt={figure.alt_text}
          className="max-w-full rounded-md border"
          style={{ borderColor: 'var(--border-subtle)' }}
        />
      ) : (
        <div
          className="h-48 w-full animate-pulse rounded-md border"
          style={{ borderColor: 'var(--border-subtle)' }}
          aria-label="Đang tải hình"
        />
      )}
      <figcaption className="text-xs" style={{ color: 'var(--text-muted)' }}>
        {figure.caption}
      </figcaption>
    </figure>
  );
}
