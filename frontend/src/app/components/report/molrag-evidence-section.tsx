import type { FusionResultSection, MolragSection } from '../../../lib/api';

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

export function MolragEvidenceSection({ data, fusionResult, language }: MolragEvidenceSectionProps) {
  const enabled = Boolean(data?.enabled);
  const examples = data?.retrieved_examples ?? [];
  const strategy = data?.strategy || 'sim_cot';
  const suggestedLabel = data?.suggested_label || 'N/A';
  const confidenceText = toFixedNumber(data?.confidence, 3);
  const fusionLabel = fusionResult?.final_label || 'N/A';
  const agreement = fusionResult?.agreement;

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

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 mb-6">
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
