import type {
  BioassayItem,
  LiteraturePaper,
  LiteratureSection,
  LiteratureSynthesis,
} from '../../../lib/api';

interface LiteratureContextSectionProps {
  data: LiteratureSection;
  language: 'vi' | 'en';
}

function formatAuthors(authors?: string | string[]) {
  if (Array.isArray(authors)) {
    return authors.filter(Boolean).join(', ') || 'N/A';
  }
  if (typeof authors === 'string') {
    return authors || 'N/A';
  }
  return 'N/A';
}

function getPaperSnippet(paper: LiteraturePaper) {
  return paper.snippet || paper.abstract_snippet || '';
}

function formatSourceLabel(source?: string) {
  switch ((source || '').toLowerCase()) {
    case 'pubmed':
      return 'PubMed';
    case 'europepmc':
      return 'Europe PMC';
    case 'semanticscholar':
      return 'Semantic Scholar';
    default:
      return source || 'Unknown';
  }
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

function renderSourceCoverage(synthesis?: LiteratureSynthesis) {
  const coverage = synthesis?.source_coverage;
  if (!coverage || typeof coverage !== 'object') {
    return null;
  }

  const entries = Object.entries(coverage).filter(([, count]) => typeof count === 'number' && count > 0);
  if (entries.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-wrap gap-2 mt-3">
      {entries.map(([source, count]) => (
        <span
          key={source}
          className="px-2 py-1 rounded-full text-xs"
          style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}
        >
          {formatSourceLabel(source)}: {count}
        </span>
      ))}
    </div>
  );
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

export function LiteratureContextSection({ data, language }: LiteratureContextSectionProps) {
  const cid = data?.compound_id?.cid;
  const pubchemUrl = data?.compound_id?.pubchem_url;
  const papers = ensureArray<LiteraturePaper>(data?.relevant_papers);
  const synthesis = data?.literature_synthesis;
  const bioassay = data?.bioassay_evidence;
  const activeAssays = ensureArray<BioassayItem>(bioassay?.active_assays);
  const mechanisms = ensureStringArray(synthesis?.consensus_mechanisms);
  const targets = ensureStringArray(synthesis?.key_targets);
  const doseSignals = ensureStringArray(synthesis?.dose_response_signals);
  const conflicts = ensureStringArray(synthesis?.conflicting_findings);
  const confidence = (synthesis?.confidence_level || 'unknown').toUpperCase();
  const evidenceBasis = synthesis?.evidence_basis === 'title_only'
    ? 'Title/snippet fallback'
    : 'Abstract-backed';

  return (
    <section id="literature" className="scroll-mt-24 lg:scroll-mt-20">
      <h2 className="text-2xl font-bold mb-6" style={{ color: 'var(--text)' }}>
        {language === 'vi' ? '§5 Bối cảnh tài liệu' : '§5 Literature Context'}
      </h2>

      <div className="rounded-xl p-6 mb-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
        <h3 className="font-semibold mb-4" style={{ color: 'var(--text)' }}>PubChem Compound Info</h3>
        <div className="space-y-2 text-sm">
          <p style={{ color: 'var(--text)' }}>
            <span style={{ color: 'var(--text-muted)' }}>CID:</span>{' '}
            <span className="font-mono font-semibold">{cid ?? 'N/A'}</span>
          </p>
          <p style={{ color: 'var(--text)' }}>
            <span style={{ color: 'var(--text-muted)' }}>Query name:</span>{' '}
            <span>{data?.query_name_used || 'N/A'}</span>
          </p>
          <p style={{ color: 'var(--text)' }}>
            <span style={{ color: 'var(--text-muted)' }}>
              {language === 'vi' ? 'Tổng kết quả tìm thấy' : 'Total search results'}:
            </span>{' '}
            <span>{data?.total_found ?? 0}</span>
          </p>
          <p style={{ color: 'var(--text)' }}>
            <span style={{ color: 'var(--text-muted)' }}>
              {language === 'vi' ? 'Nguồn truy vấn' : 'Search source'}:
            </span>{' '}
            <span>{formatSourceLabel(data?.search_source || 'pubmed')}</span>
          </p>
          {data?.fallback_used && (
            <p className="text-sm" style={{ color: 'var(--accent-yellow)' }}>
              {language === 'vi'
                ? 'Có dùng fallback source cho một phần dữ liệu literature.'
                : 'Fallback sources were used for part of the literature evidence.'}
            </p>
          )}
          {data?.search_error && data.search_error !== 'none' && (
            <p className="text-sm" style={{ color: 'var(--text-faint)' }}>
              {language === 'vi' ? 'Lỗi nguồn chính' : 'Primary source error'}: {data.search_error}
            </p>
          )}
          {pubchemUrl && (
            <a
              href={pubchemUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 mt-2 text-sm"
              style={{ color: 'var(--accent-blue)' }}
            >
              {language === 'vi' ? 'Xem trên PubChem' : 'View on PubChem'}
              <svg className="w-3 h-3" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M10 2L2 10M10 2H4M10 2V8" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </a>
          )}
        </div>
      </div>

      <div className="rounded-xl p-6 mb-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <h3 className="font-semibold" style={{ color: 'var(--text)' }}>
            {language === 'vi' ? 'Tổng hợp từ literature' : 'Literature synthesis'}
          </h3>
          <div className="flex flex-wrap gap-2">
            <span className="px-2 py-1 rounded-full text-xs" style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}>
              Confidence: {confidence}
            </span>
            <span className="px-2 py-1 rounded-full text-xs" style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}>
              {language === 'vi' ? 'Evidence basis' : 'Evidence basis'}: {evidenceBasis}
            </span>
            <span className="px-2 py-1 rounded-full text-xs" style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}>
              {language === 'vi' ? 'Paper có nội dung' : 'Papers with content'}: {synthesis?.papers_with_content ?? 0}
            </span>
          </div>
        </div>

        <p className="text-sm" style={{ color: 'var(--text-muted)', lineHeight: '1.75' }}>
          {synthesis?.synthesis_text || (language === 'vi'
            ? 'Chưa có tổng hợp literature.'
            : 'No literature synthesis is available yet.')}
        </p>

        {renderSourceCoverage(synthesis)}

        {mechanisms.length > 0 && (
          <div className="mt-4">
            <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
              Consensus mechanisms
            </p>
            {renderBadgeList(mechanisms, 'var(--accent-red)')}
          </div>
        )}

        {targets.length > 0 && (
          <div className="mt-4">
            <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
              Key targets
            </p>
            {renderBadgeList(targets, 'var(--accent-blue)')}
          </div>
        )}

        {doseSignals.length > 0 && (
          <div className="mt-4">
            <p className="text-sm font-semibold mb-2" style={{ color: 'var(--text)' }}>
              {language === 'vi' ? 'Dose / response signals' : 'Dose / response signals'}
            </p>
            <div className="space-y-2">
              {doseSignals.map((signal) => (
                <p key={signal} className="text-sm" style={{ color: 'var(--text-muted)' }}>
                  • {signal}
                </p>
              ))}
            </div>
          </div>
        )}

        {conflicts.length > 0 && (
          <div className="mt-4">
            <p className="text-sm font-semibold mb-2" style={{ color: 'var(--text)' }}>
              {language === 'vi' ? 'Conflicting findings' : 'Conflicting findings'}
            </p>
            <div className="space-y-2">
              {conflicts.map((item) => (
                <p key={item} className="text-sm" style={{ color: 'var(--text-muted)' }}>
                  • {item}
                </p>
              ))}
            </div>
          </div>
        )}

        {synthesis?.error && synthesis.error !== 'none' && (
          <p className="text-sm mt-4" style={{ color: 'var(--accent-yellow)' }}>
            {language === 'vi' ? 'Synthesis note' : 'Synthesis note'}: {synthesis.error}
          </p>
        )}
      </div>

      <div className="mb-6">
        <h3 className="font-semibold mb-4" style={{ color: 'var(--text)' }}>
          {language === 'vi'
            ? `Nghiên cứu liên quan (${papers.length} bài báo)`
            : `Related studies (${papers.length} papers)`}
        </h3>
        <div className="space-y-3">
          {papers.length === 0 && (
            <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
              <p style={{ color: 'var(--text-muted)' }}>
                {language === 'vi' ? 'Không tìm thấy bài báo phù hợp.' : 'No relevant papers found.'}
              </p>
            </div>
          )}

          {papers.map((paper, idx) => {
            const authors = formatAuthors(paper.authors);
            const snippet = getPaperSnippet(paper);
            const link = paper.pubmed_url || (paper.pmid ? `https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}` : null);
            const abstractSource = formatSourceLabel(paper.abstract_source || paper.search_source || 'unknown');

            return (
              <div
                key={`${paper.pmid || paper.title || idx}`}
                className="rounded-xl p-5 transition-colors"
                style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}
              >
                <div className="flex items-start justify-between gap-3 mb-2">
                  <p className="font-semibold flex-1" style={{ color: 'var(--text)' }}>
                    [{idx + 1}] {paper.title || 'Untitled'}
                  </p>
                </div>
                <p className="text-sm mb-1" style={{ color: 'var(--text-muted)' }}>
                  {authors} · <span className="italic">{paper.journal || 'N/A'}</span> ({paper.year || 'N/A'})
                </p>
                <p className="text-xs mb-2" style={{ color: 'var(--text-faint)' }}>
                  {language === 'vi' ? 'Evidence source' : 'Evidence source'}: {abstractSource}
                </p>
                {snippet && (
                  <p className="text-sm mb-2 italic" style={{ color: 'var(--text-faint)' }}>
                    "{snippet}"
                  </p>
                )}
                <div className="flex items-center gap-3">
                  {paper.pmid && (
                    <span className="font-mono text-xs px-2 py-1 rounded" style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}>
                      PMID: {paper.pmid}
                    </span>
                  )}
                  {link && (
                    <a
                      href={link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs flex items-center gap-1"
                      style={{ color: 'var(--accent-blue)' }}
                    >
                      {language === 'vi' ? 'Đọc bài báo' : 'Open paper'}
                      <svg className="w-3 h-3" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M10 2L2 10M10 2H4M10 2V8" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </a>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div>
        <h3 className="font-semibold mb-2" style={{ color: 'var(--text)' }}>Bioassay Data</h3>
        <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
          {language === 'vi' ? 'Tổng assay đã test' : 'Total tested'}: {bioassay?.total_assays_tested ?? 0} · {language === 'vi' ? 'Active' : 'Active'}: {activeAssays.length}
        </p>

        {data?.bioassay_explanation && (
          <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
            {data.bioassay_explanation}
          </p>
        )}

        <div className="overflow-x-auto">
          <table className="w-full text-sm" style={{ borderCollapse: 'separate', borderSpacing: 0 }}>
            <thead>
              <tr style={{ backgroundColor: 'var(--surface-alt)' }}>
                <th className="text-left p-3 rounded-tl-lg" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>AID</th>
                <th className="text-left p-3" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>
                  {language === 'vi' ? 'Tên assay' : 'Assay name'}
                </th>
                <th className="text-left p-3 rounded-tr-lg" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>
                  {language === 'vi' ? 'Kết quả' : 'Outcome'}
                </th>
              </tr>
            </thead>
            <tbody>
              {activeAssays.length === 0 && (
                <tr>
                  <td colSpan={3} className="p-3" style={{ color: 'var(--text-muted)' }}>
                    {language === 'vi' ? 'Không có bioassay active.' : 'No active bioassay records.'}
                  </td>
                </tr>
              )}

              {activeAssays.map((assay, idx) => {
                const outcome = (assay.activity_outcome || 'N/A').toUpperCase();
                const isActive = outcome === 'ACTIVE';
                return (
                  <tr
                    key={`${assay.aid || idx}`}
                    style={{
                      backgroundColor: isActive ? 'rgba(239,68,68,0.05)' : 'transparent',
                      borderTop: '1px solid var(--border)',
                    }}
                  >
                    <td className="p-3 font-mono" style={{ color: 'var(--text)' }}>{assay.aid ?? 'N/A'}</td>
                    <td className="p-3" style={{ color: 'var(--text)' }}>{assay.assay_name || 'N/A'}</td>
                    <td className="p-3">
                      <span className="flex items-center gap-2">
                        <span style={{ color: isActive ? 'var(--accent-red)' : 'var(--text-faint)', fontSize: '12px' }}>
                          {isActive ? '●' : '○'}
                        </span>
                        <span style={{ color: isActive ? 'var(--accent-red)' : 'var(--text-muted)' }}>{outcome}</span>
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
