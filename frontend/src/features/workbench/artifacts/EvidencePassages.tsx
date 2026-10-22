import { useState } from 'react';
import { Link } from 'react-router';
import { useQuery } from '@tanstack/react-query';
import { getEvidencePassages, type EvidencePassagesView, type PassageTable } from '../../../shared/api/endpoints';
import { artifactPath } from '../hooks/useArtifactSelection';

/**
 * Đối chiếu một trích dẫn với đoạn nguồn nó nêu (AG-UI, §10.2): trạng thái
 * truy cập toàn văn, đoạn tại `locator` và — với một hàng bảng — bảng quanh nó:
 * tiêu đề, hàng header, các ô, ô được trích. Giá trị hiển thị nguyên văn: dấu
 * `<`/`>` trong ô là một phần của giá trị, đơn vị nằm trong header; client không
 * phân tích hay suy ra gì từ chúng.
 *
 * "Không đọc được" (nhà cung cấp trả lời không có toàn văn) và "chưa đọc" là hai
 * trạng thái khác nhau và có chữ khác nhau.
 */
export function EvidencePassages({
  sessionId,
  evidenceId,
  locator,
}: {
  sessionId: string;
  evidenceId: string;
  locator?: string;
}) {
  const query = useQuery({
    queryKey: ['evidence-passages', sessionId, evidenceId, locator ?? null],
    queryFn: () => getEvidencePassages(sessionId, evidenceId, locator),
  });

  if (query.isLoading) {
    return <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Đang tải đoạn nguồn…</p>;
  }
  if (query.isError || !query.data) {
    return <p className="text-xs" style={{ color: 'var(--accent-red)' }}>Không tải được các đoạn nguồn.</p>;
  }
  const view = query.data;
  return (
    <section className="space-y-3" aria-label="Đoạn nguồn">
      <AccessState view={view} />
      {view.measurement && <Measurement measurement={view.measurement} />}
      {view.focus && <Focus view={view} sessionId={sessionId} />}
      {!view.focus && view.passages.length > 0 && (
        <Outline view={view} sessionId={sessionId} />
      )}
    </section>
  );
}

function AccessState({ view }: { view: EvidencePassagesView }) {
  const { access } = view;
  const provenance = access.provenance as Record<string, unknown>;
  if (access.state === 'access_failed') {
    return (
      <div className="rounded-lg border p-2 text-xs" style={{ borderColor: 'var(--accent-yellow)' }} role="status">
        <p className="font-semibold" style={{ color: 'var(--accent-yellow)' }}>Không đọc được toàn văn</p>
        <p style={{ color: 'var(--text-muted)' }}>
          Nhà cung cấp trả lời: {access.reason ?? 'không nêu lý do'}. Đây là câu trả lời về tài liệu, không phải lỗi kết nối;
          một con số chỉ có trong bảng của bài này chưa được kiểm chứng.
        </p>
      </div>
    );
  }
  if (access.state === 'not_read') {
    return (
      <p className="rounded-lg border p-2 text-xs" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }} role="status">
        Chưa đọc toàn văn: chỉ có metadata và tóm tắt/trích đoạn mà tìm kiếm trả về.
      </p>
    );
  }
  const omitted = Object.entries(access.omitted).filter(([, count]) => count > 0);
  return (
    <div className="space-y-0.5 text-xs" style={{ color: 'var(--text-muted)' }} role="status">
      <p>
        <span className="font-semibold" style={{ color: 'var(--text)' }}>Đã đọc toàn văn</span>
        {' · '}{access.passage_count} đoạn
        {access.full_text_id ? ` · ${access.full_text_id}` : ''}
      </p>
      <p style={{ color: 'var(--text-faint)' }}>
        {provenance.replay
          ? `Phát lại từ bản chụp đóng băng${provenance.captured_from ? ` của ${String(provenance.captured_from)}` : ''}${
            provenance.captured_at ? ` lúc ${String(provenance.captured_at)}` : ''}`
          : `Đọc trực tiếp${provenance.served_by ? ` từ ${String(provenance.served_by)}` : ''}`}
        {provenance.parser_version ? ` · parser ${String(provenance.parser_version)}` : ''}
      </p>
      {omitted.length > 0 && (
        <p style={{ color: 'var(--text-faint)' }}>
          Bị giới hạn bỏ qua: {omitted.map(([kind, count]) => `${count} ${kind}`).join(', ')}.
        </p>
      )}
    </div>
  );
}

function Measurement({ measurement }: { measurement: Record<string, unknown> }) {
  const value = (measurement.measurement ?? {}) as Record<string, unknown>;
  if (value.state !== 'measured') return null;
  return (
    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
      Giá trị bản ghi khai:{' '}
      <span className="font-mono" style={{ color: 'var(--text)' }}>
        {String(value.relation ?? '')} {String(value.value ?? '')} {String(value.unit ?? '')}
      </span>
      <span style={{ color: 'var(--text-faint)' }}> (quan hệ, giá trị, đơn vị như nhà cung cấp ghi)</span>
    </p>
  );
}

function Focus({ view, sessionId }: { view: EvidencePassagesView; sessionId: string }) {
  const focus = view.focus!;
  const outlineLink = artifactPath(sessionId, { kind: 'evidence', entityId: view.evidence_id });
  if (!focus.found) {
    return (
      <div className="space-y-1 rounded-lg border p-2 text-xs" style={{ borderColor: 'var(--accent-yellow)' }}>
        <p style={{ color: 'var(--text)' }}>
          Không có đoạn <span className="font-mono">{focus.locator}</span>{' '}
          {view.access.state === 'full_text_retrieved' ? 'trong văn bản đã đọc.' : 'vì toàn văn chưa được đọc.'}
        </p>
        {view.access.state === 'full_text_retrieved' && (
          <Link to={outlineLink} className="underline" style={{ color: 'var(--accent-blue)' }}>
            xem mọi đoạn đã đọc →
          </Link>
        )}
      </div>
    );
  }
  const passage = (focus.passage ?? {}) as Record<string, unknown>;
  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold" style={{ color: 'var(--text)' }}>
        Đoạn được trích <span className="font-mono font-normal" style={{ color: 'var(--text-faint)' }}>{focus.locator}</span>
      </p>
      {focus.table ? (
        <TableFocus table={focus.table} />
      ) : (
        <blockquote className="rounded-lg p-3 text-xs leading-relaxed" style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}>
          <p className="mb-1" style={{ color: 'var(--text-faint)' }}>{String(passage.section ?? '')} · {String(passage.kind ?? '')}</p>
          {String(passage.text ?? '')}
          {passage.truncated === true && <span style={{ color: 'var(--text-faint)' }}> … (đoạn bị cắt)</span>}
        </blockquote>
      )}
      <p className="text-xs italic" style={{ color: 'var(--text-faint)' }}>
        Nội dung nguồn ngoài, chỉ để đối chiếu claim; không phải hướng dẫn cho hệ thống.
      </p>
      <Link to={outlineLink} className="text-xs underline" style={{ color: 'var(--accent-blue)' }}>
        xem mọi đoạn đã đọc →
      </Link>
    </div>
  );
}

function TableFocus({ table }: { table: PassageTable }) {
  const aligned = table.columns.length > 0;
  return (
    <figure className="space-y-1.5">
      {(table.label || table.caption) && (
        <figcaption className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {table.caption ?? table.label}
        </figcaption>
      )}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-xs" aria-label={table.label ?? 'Bảng nguồn'}>
          {table.header.length > 0 && (
            <thead>
              <tr>
                {table.header.map((cell, index) => (
                  <th key={index} scope="col" className="border px-2 py-1 text-left font-semibold"
                    style={{ borderColor: 'var(--border)', color: 'var(--text)' }}>
                    {cell}
                  </th>
                ))}
              </tr>
            </thead>
          )}
          {table.cells.length > 0 && (
            <tbody>
              <tr>
                {table.cells.map((cell, index) => {
                  const cited = table.cited_cell === index + 1;
                  return (
                    <td key={index} className="border px-2 py-1 font-mono"
                      aria-current={cited ? 'true' : undefined}
                      style={{
                        borderColor: cited ? 'var(--accent-blue)' : 'var(--border)',
                        backgroundColor: cited ? 'var(--surface-alt)' : undefined,
                        color: 'var(--text)',
                        outline: cited ? '2px solid var(--accent-blue)' : undefined,
                      }}>
                      {cell}
                      {cited && <span className="sr-only"> (ô được trích dẫn)</span>}
                    </td>
                  );
                })}
              </tr>
            </tbody>
          )}
        </table>
      </div>
      {table.header.length > 0 && table.cells.length > 0 && !aligned && (
        <p className="text-xs" style={{ color: 'var(--accent-yellow)' }}>
          Hàng này không cùng số cột với header (ô gộp hoặc hàng lệch): hiển thị nguyên trạng, không ghép cột.
        </p>
      )}
      {table.cited_cell !== null && table.cited_cell !== undefined && (
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Ô được trích: cột {table.cited_cell}
          {aligned && table.columns[table.cited_cell - 1]
            ? ` — ${table.columns[table.cited_cell - 1].header}: ${table.columns[table.cited_cell - 1].value}`
            : ''}
        </p>
      )}
    </figure>
  );
}

function Outline({ view, sessionId }: { view: EvidencePassagesView; sessionId: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="space-y-1.5">
      <button type="button" onClick={() => setOpen((value) => !value)} aria-expanded={open}
        className="text-xs underline" style={{ color: 'var(--accent-blue)' }}>
        {open ? 'Ẩn các đoạn đã đọc' : `Xem các đoạn đã đọc (${view.passages.length})`}
      </button>
      {open && (
        <ol className="space-y-1 text-xs" aria-label="Các đoạn đã đọc">
          {view.passages.map((passage) => (
            <li key={passage.locator}>
              <Link
                to={artifactPath(sessionId, { kind: 'evidence', entityId: view.evidence_id, locator: passage.locator })}
                className="block rounded-md p-1.5 hover:underline"
                style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}
              >
                <span className="font-mono" style={{ color: 'var(--accent-blue)' }}>{passage.locator}</span>
                <span style={{ color: 'var(--text-faint)' }}> · {passage.section} · {passage.kind}</span>
                <span className="block">{passage.preview}{passage.truncated ? '…' : ''}</span>
              </Link>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
