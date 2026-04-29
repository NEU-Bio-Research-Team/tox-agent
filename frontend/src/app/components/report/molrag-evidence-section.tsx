import type {
  FusionResultSection,
  MolragFirestoreState,
  MolragRetrievedExample,
  MolragSection,
} from '../../../lib/api';

interface MolragEvidenceSectionProps {
  data?: MolragSection;
  fusionResult?: FusionResultSection;
  language: 'vi' | 'en';
}

function toFixedNumber(value: number | null | undefined, digits: number): string {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A';
  }
  return value.toFixed(digits);
}

function similarityColor(similarity: number | null | undefined): string {
  if (typeof similarity !== 'number' || Number.isNaN(similarity)) {
    return 'var(--text-muted)';
  }
  if (similarity >= 0.75) return 'var(--accent-red)';
  if (similarity >= 0.5) return 'var(--accent-yellow)';
  return 'var(--accent-green)';
}

function ensureArray<T>(value: unknown): T[] {
  if (Array.isArray(value)) {
    return value as T[];
  }

  if (value && typeof value === 'object') {
    return Object.values(value as Record<string, T>);
  }

  return [];
}

function ensureStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => (typeof item === 'string' ? item.trim() : ''))
    .filter(Boolean);
}

function formatSourceLabel(source?: string | null): string {
  switch ((source || '').toLowerCase()) {
    case 'firestore':
      return 'Firestore';
    case 'csv_fallback':
      return 'CSV fallback';
    default:
      return source || 'Unknown';
  }
}

function llmStatusLabel(status?: string | null): string {
  if (!status) {
    return 'N/A';
  }
  if (status === 'llm_ok') {
    return 'LLM OK';
  }
  if (status === 'llm_unavailable') {
    return 'LLM unavailable';
  }
  return status;
}

function firestoreStatusColor(firestore?: MolragFirestoreState): string {
  if (firestore?.ready) {
    return 'var(--accent-green)';
  }
  if (firestore?.enabled === false) {
    return 'var(--text-muted)';
  }
  return 'var(--accent-yellow)';
}

function renderBadgeList(items: string[], color: string) {
  if (items.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-wrap gap-2 mt-3">
      {items.map((item) => (
        <span
          key={item}
          className="px-2 py-1 rounded-full text-xs"
          style={{ backgroundColor: 'var(--surface-alt)', color }}
        >
          {item}
        </span>
      ))}
    </div>
  );
}

function renderTextList(items: string[]) {
  if (items.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2 mt-3">
      {items.map((item) => (
        <p key={item} className="text-sm" style={{ color: 'var(--text-muted)' }}>
          • {item}
        </p>
      ))}
    </div>
  );
}

function renderFirestoreAttempts(firestore?: MolragFirestoreState) {
  const attempts = ensureArray<{ database_id?: string; ready?: boolean; reason?: string | null }>(firestore?.attempts);
  if (attempts.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2 mt-3">
      {attempts.map((attempt, index) => (
        <p key={`${attempt.database_id || 'db'}-${index}`} className="text-xs" style={{ color: 'var(--text-faint)' }}>
          {attempt.database_id || '(default)'} · {attempt.ready ? 'ready' : 'not ready'}
          {attempt.reason ? ` · ${attempt.reason}` : ''}
        </p>
      ))}
    </div>
  );
}

export function MolragEvidenceSection({ data, fusionResult, language }: MolragEvidenceSectionProps) {
  const enabled = Boolean(data?.enabled);
  const examples = ensureArray<MolragRetrievedExample>(data?.retrieved_examples);
  const strategy = data?.strategy || 'sim_cot';
  const suggestedLabel = data?.suggested_label || 'N/A';
  const confidenceText = toFixedNumber(data?.confidence, 3);
  const fusionLabel = fusionResult?.final_label || 'N/A';
  const agreement = fusionResult?.agreement;
  const mechanismChain = ensureStringArray(data?.mechanism_chain);
  const keySubstructures = ensureStringArray(data?.key_substructures);
  const riskModifiers = ensureStringArray(data?.risk_modifiers);
  const knowledgeHighlights = ensureStringArray(data?.knowledge_highlights);
  const literatureHighlights = ensureStringArray(data?.literature_highlights);
  const retrievalSource = data?.retrieval_overview?.db_source || data?.retrieval_db_source || 'N/A';
  const retrievalDbSize = data?.retrieval_overview?.db_size ?? data?.retrieval_db_size ?? 0;
  const firestore = data?.firestore;

  return (
    <section id="molrag" className="scroll-mt-24 lg:scroll-mt-20">
      <h2 className="text-2xl font-bold mb-6" style={{ color: 'var(--text)' }}>
        {language === 'vi' ? '§4 MolRAG Evidence & Reasoning' : '§4 MolRAG Evidence & Reasoning'}
      </h2>

      {!enabled && (
        <div
          className="rounded-xl p-5 mb-5"
          style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}
        >
          <p style={{ color: 'var(--text-muted)' }}>
            {language === 'vi'
              ? 'MolRAG hiện chưa được bật cho phiên phân tích này.'
              : 'MolRAG is not enabled for this analysis session.'}
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-6 gap-4 mb-6">
        <div className="rounded-xl p-4" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>Strategy</p>
          <p className="font-mono text-lg font-semibold" style={{ color: 'var(--text)' }}>{strategy}</p>
        </div>

        <div className="rounded-xl p-4" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'Retrieved analogs' : 'Retrieved analogs'}
          </p>
          <p className="font-mono text-lg font-semibold" style={{ color: 'var(--text)' }}>{examples.length}</p>
        </div>

        <div className="rounded-xl p-4" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'MolRAG suggested label' : 'MolRAG suggested label'}
          </p>
          <p className="text-lg font-semibold" style={{ color: 'var(--text)' }}>{suggestedLabel}</p>
        </div>

        <div className="rounded-xl p-4" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'MolRAG confidence' : 'MolRAG confidence'}
          </p>
          <p className="font-mono text-lg font-semibold" style={{ color: 'var(--text)' }}>{confidenceText}</p>
        </div>

        <div className="rounded-xl p-4" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'Retrieval source' : 'Retrieval source'}
          </p>
          <p className="font-mono text-sm font-semibold" style={{ color: 'var(--text)' }}>{formatSourceLabel(retrievalSource)}</p>
        </div>

        <div className="rounded-xl p-4" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'Retrieval DB size' : 'Retrieval DB size'}
          </p>
          <p className="font-mono text-sm font-semibold" style={{ color: 'var(--text)' }}>{retrievalDbSize}</p>
        </div>

        <div className="rounded-xl p-4" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'LLM status' : 'LLM status'}
          </p>
          <p className="font-mono text-sm font-semibold" style={{ color: 'var(--text)' }}>{llmStatusLabel(data?.llm_status)}</p>
        </div>
      </div>

      <div className="rounded-xl p-5 mb-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
        <h3 className="font-semibold mb-3" style={{ color: 'var(--text)' }}>
          {language === 'vi' ? 'Reasoning summary' : 'Reasoning summary'}
        </h3>
        <p className="text-sm mb-3" style={{ color: 'var(--text-muted)', lineHeight: '1.7' }}>
          {data?.reasoning_summary || (language === 'vi' ? 'Không có reasoning summary.' : 'No reasoning summary available.')}
        </p>

        <h3 className="font-semibold mb-3" style={{ color: 'var(--text)' }}>
          {language === 'vi' ? 'Evidence summary' : 'Evidence summary'}
        </h3>
        <p className="text-sm" style={{ color: 'var(--text-muted)', lineHeight: '1.7' }}>
          {data?.evidence_summary || (language === 'vi' ? 'Không có evidence summary.' : 'No evidence summary available.')}
        </p>

        {data?.evidence_overview && (
          <>
            <h3 className="font-semibold mt-5 mb-3" style={{ color: 'var(--text)' }}>
              {language === 'vi' ? 'Evidence overview' : 'Evidence overview'}
            </h3>
            <p className="text-sm" style={{ color: 'var(--text-muted)', lineHeight: '1.7' }}>
              {data.evidence_overview}
            </p>
          </>
        )}

        {data?.longform_summary && (
          <>
            <h3 className="font-semibold mt-5 mb-3" style={{ color: 'var(--text)' }}>
              {language === 'vi' ? 'Detailed MolRAG narrative' : 'Detailed MolRAG narrative'}
            </h3>
            <p className="text-sm" style={{ color: 'var(--text-muted)', lineHeight: '1.8' }}>
              {data.longform_summary}
            </p>
          </>
        )}

        {data?.analogy_reasoning && (
          <>
            <h3 className="font-semibold mt-5 mb-3" style={{ color: 'var(--text)' }}>
              {language === 'vi' ? 'Analogy reasoning' : 'Analogy reasoning'}
            </h3>
            <p className="text-sm" style={{ color: 'var(--text-muted)', lineHeight: '1.7' }}>
              {data.analogy_reasoning}
            </p>
          </>
        )}

        {data?.confidence_rationale && (
          <>
            <h3 className="font-semibold mt-5 mb-3" style={{ color: 'var(--text)' }}>
              {language === 'vi' ? 'Confidence rationale' : 'Confidence rationale'}
            </h3>
            <p className="text-sm" style={{ color: 'var(--text-muted)', lineHeight: '1.7' }}>
              {data.confidence_rationale}
            </p>
          </>
        )}
      </div>

      <div className="rounded-xl p-5 mb-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <h3 className="font-semibold" style={{ color: 'var(--text)' }}>
            {language === 'vi' ? 'Firestore / retrieval diagnostics' : 'Firestore / retrieval diagnostics'}
          </h3>
          <span className="px-2 py-1 rounded-full text-xs" style={{ backgroundColor: 'var(--surface-alt)', color: firestoreStatusColor(firestore) }}>
            {firestore?.ready ? 'READY' : firestore?.enabled === false ? 'DISABLED' : 'NOT READY'}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
          <p style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'Credential source' : 'Credential source'}: <span style={{ color: 'var(--text)' }}>{firestore?.credential_source || 'N/A'}</span>
          </p>
          <p style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'Project ID' : 'Project ID'}: <span style={{ color: 'var(--text)' }}>{firestore?.project_id || 'N/A'}</span>
          </p>
          <p style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'Configured DB' : 'Configured DB'}: <span style={{ color: 'var(--text)' }}>{firestore?.configured_database_id || firestore?.database_id || 'N/A'}</span>
          </p>
          <p style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'Active DB' : 'Active DB'}: <span style={{ color: 'var(--text)' }}>{firestore?.database_id || 'N/A'}</span>
          </p>
        </div>

        {firestore?.reason && (
          <p className="text-sm mt-3" style={{ color: firestore?.ready ? 'var(--accent-yellow)' : 'var(--accent-red)' }}>
            {language === 'vi' ? 'Diagnostic' : 'Diagnostic'}: {firestore.reason}
          </p>
        )}

        {firestore?.fallback_reason && (
          <p className="text-sm mt-2" style={{ color: 'var(--accent-yellow)' }}>
            {language === 'vi' ? 'Fallback reason' : 'Fallback reason'}: {firestore.fallback_reason}
          </p>
        )}

        {renderFirestoreAttempts(firestore)}
      </div>

      <div className="rounded-xl p-5 mb-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
        <h3 className="font-semibold mb-3" style={{ color: 'var(--text)' }}>
          {language === 'vi' ? 'Mechanism chain' : 'Mechanism chain'}
        </h3>
        {renderTextList(mechanismChain) || (
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'Chưa có mechanism chain chi tiết.' : 'No detailed mechanism chain is available.'}
          </p>
        )}

        {keySubstructures.length > 0 && (
          <div className="mt-5">
            <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
              {language === 'vi' ? 'Key substructures / motifs' : 'Key substructures / motifs'}
            </p>
            {renderBadgeList(keySubstructures, 'var(--accent-red)')}
          </div>
        )}

        {riskModifiers.length > 0 && (
          <div className="mt-5">
            <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
              {language === 'vi' ? 'Risk modifiers' : 'Risk modifiers'}
            </p>
            {renderBadgeList(riskModifiers, 'var(--accent-yellow)')}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <h3 className="font-semibold mb-3" style={{ color: 'var(--text)' }}>
            {language === 'vi' ? 'Knowledge highlights' : 'Knowledge highlights'}
          </h3>
          {renderTextList(knowledgeHighlights) || (
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              {language === 'vi' ? 'Chưa có curated knowledge highlight.' : 'No curated knowledge highlights are available.'}
            </p>
          )}
        </div>

        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <h3 className="font-semibold mb-3" style={{ color: 'var(--text)' }}>
            {language === 'vi' ? 'Literature highlights' : 'Literature highlights'}
          </h3>
          {renderTextList(literatureHighlights) || (
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              {language === 'vi' ? 'Chưa có literature highlight.' : 'No literature highlights are available.'}
            </p>
          )}
        </div>
      </div>

      <div className="rounded-xl p-5 mb-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
        <h3 className="font-semibold mb-3" style={{ color: 'var(--text)' }}>
          {language === 'vi' ? 'Fusion result (MVP evidence-only)' : 'Fusion result (MVP evidence-only)'}
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>Final label</p>
            <p className="font-semibold" style={{ color: 'var(--text)' }}>{fusionLabel}</p>
          </div>
          <div>
            <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>Agreement</p>
            <p
              className="font-semibold"
              style={{
                color:
                  agreement === true
                    ? 'var(--accent-green)'
                    : agreement === false
                      ? 'var(--accent-yellow)'
                      : 'var(--text-muted)',
              }}
            >
              {agreement === true ? 'YES' : agreement === false ? 'NO' : 'N/A'}
            </p>
          </div>
          <div>
            <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>Mode</p>
            <p className="font-mono" style={{ color: 'var(--text)' }}>{fusionResult?.mode || 'N/A'}</p>
          </div>
        </div>

        {fusionResult?.decision_note && (
          <p className="text-sm mt-4" style={{ color: 'var(--text-muted)', lineHeight: '1.7' }}>
            {fusionResult.decision_note}
          </p>
        )}
      </div>

      <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
        <h3 className="font-semibold mb-3" style={{ color: 'var(--text)' }}>
          {language === 'vi' ? 'Top retrieved analog molecules' : 'Top retrieved analog molecules'}
        </h3>

        {examples.length === 0 ? (
          <p style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'Không có analog molecules phù hợp ngưỡng similarity.' : 'No analog molecules met the similarity threshold.'}
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm" style={{ borderCollapse: 'separate', borderSpacing: 0 }}>
              <thead>
                <tr style={{ backgroundColor: 'var(--surface-alt)' }}>
                  <th className="text-left p-3 rounded-tl-lg" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>#</th>
                  <th className="text-left p-3" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Name</th>
                  <th className="text-left p-3" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Similarity</th>
                  <th className="text-left p-3" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Label</th>
                  <th className="text-left p-3" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Source</th>
                  <th className="text-left p-3 rounded-tr-lg" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Exact</th>
                </tr>
              </thead>
              <tbody>
                {examples.map((example, index) => (
                  <tr
                    key={`${example.entry_id || example.canonical_smiles || index}`}
                    style={{ borderTop: '1px solid var(--border)' }}
                  >
                    <td className="p-3 font-mono" style={{ color: 'var(--text)' }}>{index + 1}</td>
                    <td className="p-3" style={{ color: 'var(--text)' }}>{example.name || 'N/A'}</td>
                    <td className="p-3 font-mono" style={{ color: similarityColor(example.similarity) }}>
                      {toFixedNumber(example.similarity, 3)}
                    </td>
                    <td className="p-3" style={{ color: 'var(--text)' }}>{example.label || 'Unknown'}</td>
                    <td className="p-3" style={{ color: 'var(--text-muted)' }}>{example.source || 'N/A'}</td>
                    <td className="p-3" style={{ color: example.is_exact_match ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                      {example.is_exact_match ? 'YES' : 'NO'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {data?.error && (
          <p className="text-sm mt-3" style={{ color: 'var(--accent-red)' }}>
            {language === 'vi' ? 'Lỗi MolRAG:' : 'MolRAG error:'} {data.error}
          </p>
        )}
      </div>
    </section>
  );
}
