import { useState, type FormEvent, type ReactNode } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { addScientificCaseContext, getScientificCase, listScientificCases } from '../../lib/api/endpoints';
import type { CaseEvidenceEntry, HypothesisStatus, ScientificCase } from '../../lib/api/types';

/**
 * The investigation board (ADR 0012, RETHINK §4.4 step 4): the session's
 * scientific case as a researcher reads it — the decision question, competing
 * hypotheses with the evidence for and against each, what is still unknown,
 * what would change the conclusion, and the proposed next test. The researcher
 * can add context (an in-house result, an exposure); the next turn sees it.
 *
 * Coverage numbers are shown side by side and never combined: "has a source"
 * is not "has direct independent evidence".
 */

const STATUS_LABEL: Record<HypothesisStatus, string> = {
  open: 'đang mở',
  supported: 'được ủng hộ',
  weakened: 'bị yếu đi',
  refuted: 'bị bác bỏ',
  unresolvable: 'chưa thể giải quyết',
};

const STATUS_COLOR: Record<HypothesisStatus, string> = {
  open: 'var(--text-muted)',
  supported: 'var(--accent-blue)',
  weakened: 'var(--accent-yellow)',
  refuted: 'var(--accent-red)',
  unresolvable: 'var(--text-faint)',
};

const SOURCE_LABEL: Record<string, string> = {
  predictor_fact: 'điểm mô hình',
  explanation_fact: 'attribution của mô hình',
  external_experimental: 'dữ liệu thực nghiệm',
  external_regulatory: 'nguồn quy định',
  report_fact: 'báo cáo',
  user_supplied: 'dữ liệu nhà nghiên cứu',
};

const MODEL_SIGNALS = new Set(['predictor_fact', 'explanation_fact']);

const SEVERITY_LABEL: Record<string, string> = {
  low: 'thấp', medium: 'trung bình', high: 'cao', blocking: 'chặn kết luận',
};

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-2">
      <h3 className="text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-faint)' }}>{title}</h3>
      {children}
    </section>
  );
}

function EvidenceLine({ entry }: { entry: CaseEvidenceEntry }) {
  const scope = Object.entries(entry.scope).map(([k, v]) => `${k}: ${v}`).join(' · ');
  return (
    <li className="rounded-md p-2 text-xs" style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text)' }}>
      <p>{entry.claim}</p>
      <p className="mt-1" style={{ color: 'var(--text-faint)' }}>
        {entry.id} · {SOURCE_LABEL[entry.source_class] ?? entry.source_class} · {entry.directness}
        {scope ? ` · ${scope}` : ''}
      </p>
      {MODEL_SIGNALS.has(entry.source_class) && (
        <p className="mt-1" style={{ color: 'var(--accent-yellow)' }}>
          Tín hiệu về mô hình, không phải bằng chứng độc lập.
        </p>
      )}
    </li>
  );
}

function CaseBody({ sessionId, data }: { sessionId: string; data: ScientificCase }) {
  const queryClient = useQueryClient();
  const [key, setKey] = useState('');
  const [value, setValue] = useState('');
  const [note, setNote] = useState('');
  const addContext = useMutation({
    mutationFn: () => addScientificCaseContext(sessionId, data.case_id, {
      key: key.trim(), value: value.trim(), ...(note.trim() ? { note: note.trim() } : {}),
    }),
    onSuccess: () => {
      setKey(''); setValue(''); setNote('');
      void queryClient.invalidateQueries({ queryKey: ['case', sessionId, data.case_id] });
      void queryClient.invalidateQueries({ queryKey: ['cases', sessionId] });
    },
  });
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (key.trim() && value.trim()) addContext.mutate();
  };
  const cov = data.coverage;
  const openUncertainties = data.uncertainties.filter((u) => u.status === 'open');

  return (
    <div className="space-y-4">
      <Section title="Câu hỏi quyết định">
        <p className="text-sm" style={{ color: 'var(--text)' }}>{data.question || 'Chưa ghi câu hỏi.'}</p>
        {data.decision_context && <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{data.decision_context}</p>}
      </Section>

      <Section title="Độ phủ bằng chứng">
        <ul className="grid grid-cols-2 gap-2 text-xs" style={{ color: 'var(--text-muted)' }} aria-label="Độ phủ bằng chứng">
          <li>Có nguồn: {cov.with_any_source}/{cov.hypotheses}</li>
          <li>Có bằng chứng độc lập trực tiếp: {cov.with_independent_direct_evidence}/{cov.hypotheses}</li>
          <li>Đã xét phản chứng: {cov.with_counterevidence_considered}/{cov.hypotheses}</li>
          <li>Bất định còn mở: {cov.open_uncertainties} (chặn: {cov.blocking_uncertainties})</li>
        </ul>
        <p className="text-[11px]" style={{ color: 'var(--text-faint)' }}>Các chỉ số tách riêng, không gộp thành một điểm.</p>
      </Section>

      <Section title="Giả thuyết cạnh tranh">
        {data.hypotheses.length === 0 && <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Chưa có giả thuyết.</p>}
        {data.hypotheses.map((h) => {
          const forIt = data.evidence.filter((e) => e.hypothesis_ids.includes(h.id) && e.stance === 'supports');
          const against = data.evidence.filter((e) => e.hypothesis_ids.includes(h.id) && e.stance === 'contradicts');
          return (
            <article key={h.id} className="space-y-2 rounded-lg border p-3" style={{ borderColor: 'var(--border)' }}>
              <div className="flex items-baseline justify-between gap-2">
                <p className="text-sm font-medium" style={{ color: 'var(--text)' }}>{h.id}. {h.statement}</p>
                <span className="shrink-0 text-xs font-medium" style={{ color: STATUS_COLOR[h.status] }}>{STATUS_LABEL[h.status]}</span>
              </div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Bị bác bỏ nếu: {h.refutation_condition}</p>
              {h.status_reason && <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Lý do trạng thái: {h.status_reason}</p>}
              <p className="text-xs font-medium" style={{ color: 'var(--text)' }}>Ủng hộ ({forIt.length})</p>
              <ul className="space-y-1">{forIt.map((e) => <EvidenceLine key={e.id} entry={e} />)}</ul>
              <p className="text-xs font-medium" style={{ color: 'var(--text)' }}>Phản bác ({against.length})</p>
              <ul className="space-y-1">{against.map((e) => <EvidenceLine key={e.id} entry={e} />)}</ul>
            </article>
          );
        })}
      </Section>

      <Section title="Điều chưa biết">
        {openUncertainties.length === 0
          ? <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Không có bất định nào đang mở được ghi nhận.</p>
          : (
            <ul className="space-y-1 text-xs" style={{ color: 'var(--text)' }}>
              {openUncertainties.map((u) => (
                <li key={u.id}>{u.id} · {u.kind} · {SEVERITY_LABEL[u.severity] ?? u.severity}: {u.description}</li>
              ))}
            </ul>
          )}
      </Section>

      <Section title="Kết luận có điều kiện">
        {data.conclusion.can_say.length > 0 && (
          <div className="text-xs" style={{ color: 'var(--text)' }}>
            <p className="font-medium">Có thể nói</p>
            <ul className="list-disc pl-4">
              {data.conclusion.can_say.map((line, index) => <li key={index}>{line.text} <span style={{ color: 'var(--text-faint)' }}>({line.evidence_ids.join(', ')})</span></li>)}
            </ul>
          </div>
        )}
        {data.conclusion.cannot_say.length > 0 && (
          <div className="text-xs" style={{ color: 'var(--text)' }}>
            <p className="font-medium">Chưa thể nói</p>
            <ul className="list-disc pl-4">{data.conclusion.cannot_say.map((line, index) => <li key={index}>{line}</li>)}</ul>
          </div>
        )}
        <div className="text-xs" style={{ color: 'var(--text)' }}>
          <p className="font-medium">Điều gì sẽ thay đổi nhận định</p>
          {data.conclusion.what_would_change.length === 0
            ? <p style={{ color: 'var(--text-faint)' }}>Chưa được nêu.</p>
            : <ul className="list-disc pl-4">{data.conclusion.what_would_change.map((line, index) => <li key={index}>{line}</li>)}</ul>}
        </div>
      </Section>

      <Section title="Phép thử đề xuất">
        {data.next_tests.length === 0
          ? <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Chưa có phép thử nào được đề xuất.</p>
          : data.next_tests.map((t) => (
            <div key={t.id} className="rounded-md p-2 text-xs" style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text)' }}>
              <p className="font-medium">{t.test}</p>
              <p style={{ color: 'var(--text-muted)' }}>{t.rationale}</p>
              <p style={{ color: 'var(--text-faint)' }}>Phân biệt: {t.discriminates.join(', ')}</p>
              {t.expected_readouts.length > 0 && <ul className="list-disc pl-4">{t.expected_readouts.map((r, index) => <li key={index}>{r}</li>)}</ul>}
            </div>
          ))}
      </Section>

      <Section title="Bối cảnh từ nhà nghiên cứu">
        {data.context.length > 0 && (
          <ul className="space-y-1 text-xs" style={{ color: 'var(--text)' }}>
            {data.context.map((c) => <li key={c.id}>{c.id} · {c.key}: {c.value}{c.note ? ` — ${c.note}` : ''}</li>)}
          </ul>
        )}
        {data.status === 'open' && (
          <form onSubmit={submit} className="space-y-2" aria-label="Thêm bối cảnh">
            <input className="w-full rounded-md border px-2 py-1 text-xs" style={{ borderColor: 'var(--border)', backgroundColor: 'var(--surface)' }}
              placeholder="Loại dữ liệu (vd. patch_clamp_ic50)" aria-label="Loại dữ liệu" value={key} onChange={(e) => setKey(e.target.value)} />
            <input className="w-full rounded-md border px-2 py-1 text-xs" style={{ borderColor: 'var(--border)', backgroundColor: 'var(--surface)' }}
              placeholder="Giá trị (vd. 30 µM, in-house, HEK293)" aria-label="Giá trị" value={value} onChange={(e) => setValue(e.target.value)} />
            <input className="w-full rounded-md border px-2 py-1 text-xs" style={{ borderColor: 'var(--border)', backgroundColor: 'var(--surface)' }}
              placeholder="Ghi chú (không bắt buộc)" aria-label="Ghi chú" value={note} onChange={(e) => setNote(e.target.value)} />
            <button type="submit" disabled={!key.trim() || !value.trim() || addContext.isPending}
              className="rounded-md px-3 py-1 text-xs font-medium disabled:opacity-50" style={{ backgroundColor: 'var(--accent-blue)', color: 'white' }}>
              Thêm vào hồ sơ
            </button>
            {addContext.isError && <p className="text-xs" style={{ color: 'var(--accent-red)' }}>Không thêm được bối cảnh.</p>}
            <p className="text-[11px]" style={{ color: 'var(--text-faint)' }}>Lượt hỏi tiếp theo sẽ thấy bối cảnh này trong hồ sơ.</p>
          </form>
        )}
      </Section>

      <p className="text-[11px]" style={{ color: 'var(--text-faint)' }}>
        Hồ sơ {data.case_id} · phiên bản {data.revision} · {data.runs.length} lượt điều tra
      </p>
    </div>
  );
}

export function InvestigationBoard({ sessionId }: { sessionId: string }) {
  const [selected, setSelected] = useState<string | null>(null);
  const list = useQuery({ queryKey: ['cases', sessionId], queryFn: () => listScientificCases(sessionId) });
  const cases = list.data?.cases ?? [];
  const caseId = selected ?? cases.find((c) => c.status === 'open')?.case_id ?? cases[0]?.case_id ?? null;
  const detail = useQuery({
    queryKey: ['case', sessionId, caseId],
    queryFn: () => getScientificCase(sessionId, caseId as string),
    enabled: caseId !== null,
  });

  if (list.isLoading) return <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Đang tải hồ sơ điều tra…</p>;
  if (list.isError) return <p className="text-xs" style={{ color: 'var(--accent-red)' }}>Không tải được hồ sơ điều tra.</p>;
  if (cases.length === 0) {
    return (
      <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
        Phiên này chưa có hồ sơ điều tra. Hồ sơ được tạo khi một lượt hỗ trợ quyết định chạy trên bản triển khai bật hồ sơ điều tra.
      </p>
    );
  }
  return (
    <div className="space-y-3">
      {cases.length > 1 && (
        <select aria-label="Chọn hồ sơ" className="w-full rounded-md border px-2 py-1 text-xs"
          style={{ borderColor: 'var(--border)', backgroundColor: 'var(--surface)' }}
          value={caseId ?? ''} onChange={(event) => setSelected(event.target.value)}>
          {cases.map((c) => <option key={c.case_id} value={c.case_id}>{c.question.slice(0, 80) || c.case_id} ({c.status})</option>)}
        </select>
      )}
      {detail.isLoading && <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Đang tải…</p>}
      {detail.isError && <p className="text-xs" style={{ color: 'var(--accent-red)' }}>Không tải được hồ sơ.</p>}
      {detail.data && <CaseBody sessionId={sessionId} data={detail.data} />}
    </div>
  );
}
