import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getCaseRunChanges, type CaseRunChanges, type CaseSectionChanges } from '../../../shared/api/endpoints';
import type { ScientificCase } from '../../../shared/api/types';

/**
 * "Lượt này đã đổi gì, giữ gì" (AG-UI, US4). Hai nửa quan trọng như nhau: một
 * lượt lặng lẽ viết lại con số của screening, nếu chỉ nhìn danh sách thay đổi,
 * trông y hệt một lượt không đụng tới nó. Vì vậy phần "giữ nguyên" nêu rõ các
 * artifact đóng băng có cùng hash với lượt trước, và subject là của hồ sơ.
 *
 * Trước/sau là log của hồ sơ phát lại tới hai revision (server làm); client chỉ
 * hiển thị, không tự so sánh.
 */

const STATUS_LABEL: Record<string, string> = {
  open: 'đang mở', supported: 'được ủng hộ', weakened: 'bị yếu đi', refuted: 'bị bác bỏ',
  unresolvable: 'chưa thể giải quyết', resolved: 'đã giải quyết',
};

const ACTOR_LABEL: Record<string, string> = { user: 'nhà nghiên cứu', model: 'mô hình', server: 'máy chủ' };

type Entry = Record<string, unknown>;

function text(value: unknown): string {
  return value === null || value === undefined ? '' : String(value);
}

export function CaseChanges({ sessionId, data }: { sessionId: string; data: ScientificCase }) {
  const runs = data.runs;
  const [runId, setRunId] = useState<string | null>(null);
  const selected = runId ?? runs[runs.length - 1]?.run_id ?? null;
  const query = useQuery({
    queryKey: ['case-changes', sessionId, data.case_id, selected, data.revision],
    queryFn: () => getCaseRunChanges(sessionId, data.case_id, selected as string),
    enabled: selected !== null,
  });
  if (selected === null) return null;
  return (
    <section className="space-y-2" aria-label="Lượt này đã đổi gì, giữ gì">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-faint)' }}>
          Lượt này đã đổi gì, giữ gì
        </h3>
        {runs.length > 1 && (
          <select aria-label="Chọn lượt" className="rounded-md border px-1.5 py-0.5 text-xs"
            style={{ borderColor: 'var(--border)', backgroundColor: 'var(--surface)' }}
            value={selected} onChange={(event) => setRunId(event.target.value)}>
            {runs.map((run, index) => (
              <option key={run.run_id} value={run.run_id}>Lượt {index + 1}</option>
            ))}
          </select>
        )}
      </div>
      {query.isLoading && <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Đang so sánh…</p>}
      {query.isError && <p className="text-xs" style={{ color: 'var(--accent-red)' }}>Không tải được thay đổi của lượt này.</p>}
      {query.data && <ChangesBody changes={query.data} />}
    </section>
  );
}

export function ChangesBody({ changes }: { changes: CaseRunChanges }) {
  const first = changes.previous_run_id === null;
  const changed = changedLines(changes);
  const kept = keptLines(changes);
  return (
    <div className="space-y-2 text-xs">
      {first && (
        <p style={{ color: 'var(--text-muted)' }}>Lượt đầu tiên của hồ sơ: mọi mục đều mới, chưa có gì để so.</p>
      )}
      <div className="rounded-md p-2" style={{ backgroundColor: 'var(--surface-alt)' }}>
        <p className="mb-1 font-semibold" style={{ color: 'var(--text)' }}>Đã đổi ({changed.length})</p>
        {changed.length === 0
          ? <p style={{ color: 'var(--text-faint)' }}>Lượt này không thay đổi mục nào của hồ sơ.</p>
          : <ul className="list-disc space-y-0.5 pl-4" aria-label="Đã đổi" style={{ color: 'var(--text)' }}>
            {changed.map((line, index) => <li key={index}>{line}</li>)}
          </ul>}
      </div>
      <div className="rounded-md border p-2" style={{ borderColor: 'var(--border)' }}>
        <p className="mb-1 font-semibold" style={{ color: 'var(--text)' }}>Giữ nguyên</p>
        <ul className="list-disc space-y-0.5 pl-4" aria-label="Giữ nguyên" style={{ color: 'var(--text-muted)' }}>
          {kept.map((line, index) => <li key={index}>{line}</li>)}
        </ul>
      </div>
      <p className="text-[11px]" style={{ color: 'var(--text-faint)' }}>
        Revision {changes.from_revision} → {changes.to_revision}
        {Object.keys(changes.updates_by_actor).length > 0 && ' · '}
        {Object.entries(changes.updates_by_actor).map(([actor, n]) => `${n} của ${ACTOR_LABEL[actor] ?? actor}`).join(', ')}
      </p>
    </div>
  );
}

function changedLines(changes: CaseRunChanges): string[] {
  const lines: string[] = [];
  for (const item of changes.context.added as Entry[]) {
    lines.push(`${ACTOR_LABEL[text(item.actor)] ?? text(item.actor)} thêm bối cảnh: ${text(item.key)} = ${text(item.value)}${item.unit ? ` ${text(item.unit)}` : ''}${item.source ? ` (nguồn: ${text(item.source)})` : ''}`);
  }
  for (const h of changes.hypotheses.added as Entry[]) {
    lines.push(`Giả thuyết mới ${text(h.id)}: ${text(h.statement)}`);
  }
  for (const h of changes.hypotheses.changed as Entry[]) {
    const status = fieldChange(h, 'status');
    lines.push(status
      ? `${text(h.id)}: ${STATUS_LABEL[text(status.before)] ?? text(status.before)} → ${STATUS_LABEL[text(status.after)] ?? text(status.after)}${h.status_reason ? ` — ${text(h.status_reason)}` : ''}`
      : `${text(h.id)} được viết lại: ${text(h.statement)}`);
  }
  for (const e of changes.evidence.added as Entry[]) {
    lines.push(`Bằng chứng mới ${text(e.id)} (${text(e.stance)}, ${text(e.directness)}): ${text(e.claim)}`);
  }
  for (const u of changes.uncertainties.added as Entry[]) {
    lines.push(`Điều chưa biết mới ${text(u.id)}: ${text(u.description)}`);
  }
  for (const u of changes.uncertainties.changed as Entry[]) {
    lines.push(`${text(u.id)} ${STATUS_LABEL[text(u.status)] ?? text(u.status)}${u.resolution ? `: ${text(u.resolution)}` : ''}`);
  }
  for (const t of changes.next_tests.added as Entry[]) {
    lines.push(`Phép thử đề xuất mới ${text(t.id)}: ${text(t.test)}`);
  }
  const conclusion = changes.conclusion as Entry;
  if (conclusion.changed) {
    const after = (conclusion.after ?? {}) as Entry;
    const proposed = after.status === 'proposed';
    lines.push(`Kết luận ${proposed ? 'được đề xuất (chưa phát hành)' : 'đã thay đổi'}: ${summarise(after)}`);
    if (proposed && conclusion.released_kept) {
      lines.push(`Kết luận đã phát hành trước đó vẫn được giữ: ${summarise(conclusion.released_kept as Entry)}`);
    }
  }
  if ((changes.data_scope as Entry).changed) {
    const after = ((changes.data_scope as Entry).after ?? {}) as Entry;
    lines.push(after.external_search ? 'Phạm vi dữ liệu: cho phép tìm tài liệu bên ngoài' : `Phạm vi dữ liệu: chỉ nội bộ — ${text(after.reason)}`);
  }
  for (const a of changes.artifacts.changed as Entry[]) {
    lines.push(`Đầu vào khác lượt trước: ${text(a.kind)} ${text(a.source_id)} đổi hash ${text(a.before_sha256).slice(0, 12)}… → ${text(a.after_sha256).slice(0, 12)}…`);
  }
  for (const a of changes.artifacts.added as Entry[]) {
    if (changes.artifacts.compared) lines.push(`Artifact mới: ${text(a.kind)} ${text(a.source_id)}`);
  }
  return lines;
}

function keptLines(changes: CaseRunChanges): string[] {
  const lines: string[] = [
    `Subject của hồ sơ (${text((changes.subject as Entry).subject_key)}). Đổi target hoặc ligand là một hồ sơ khác, không phải lượt này.`,
  ];
  for (const a of changes.artifacts.kept as Entry[]) {
    lines.push(`${text(a.kind)} ${text(a.source_id)}: cùng hash ${text(a.content_sha256).slice(0, 12)}… với lượt trước — ước lượng, khoảng tin cậy, quyết định policy và trạng thái stage không đổi.`);
  }
  if (!changes.artifacts.compared && changes.previous_run_id !== null) {
    lines.push(`Không so được artifact: ${text(changes.artifacts.reason)}.`);
  }
  const counts = ([
    ['bối cảnh', changes.context], ['giả thuyết', changes.hypotheses], ['bằng chứng', changes.evidence],
    ['điều chưa biết', changes.uncertainties], ['phép thử', changes.next_tests],
  ] as [string, CaseSectionChanges][])
    .filter(([, section]) => section.kept.length > 0)
    .map(([label, section]) => `${section.kept.length} ${label} (${section.kept.join(', ')})`);
  if (counts.length > 0) lines.push(`Không đổi: ${counts.join('; ')}.`);
  if (!(changes.conclusion as Entry).changed && changes.previous_run_id !== null) lines.push('Kết luận của hồ sơ.');
  return lines;
}

function fieldChange(entry: Entry, field: string): { before: unknown; after: unknown } | null {
  const list = Array.isArray(entry.changes) ? (entry.changes as Entry[]) : [];
  const found = list.find((change) => change.field === field);
  return found ? { before: found.before, after: found.after } : null;
}

function summarise(conclusion: Entry): string {
  const canSay = Array.isArray(conclusion.can_say) ? (conclusion.can_say as Entry[]).map((line) => text(line.text)) : [];
  const cannotSay = Array.isArray(conclusion.cannot_say) ? (conclusion.cannot_say as unknown[]).map(text) : [];
  const parts = [
    canSay.length > 0 && `có thể nói: ${canSay.join('; ')}`,
    cannotSay.length > 0 && `chưa thể nói: ${cannotSay.join('; ')}`,
  ].filter(Boolean);
  return parts.length > 0 ? parts.join(' · ') : 'trống';
}
