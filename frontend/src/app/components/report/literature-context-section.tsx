import type {
  BioassayItem,
  LiteraturePaper,
  LiteratureSection,
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

function ensureArray<T>(value: unknown): T[] {
  if (Array.isArray(value)) {
    return value as T[];
  }

  if (value && typeof value === 'object') {
    return Object.values(value as Record<string, T>);
  }

  return [];
}

export function LiteratureContextSection({ data, language }: LiteratureContextSectionProps) {
  const cid = data?.compound_id?.cid;
  const pubchemUrl = data?.compound_id?.pubchem_url;
  const papers = ensureArray<LiteraturePaper>(data?.relevant_papers);
  const bioassay = data?.bioassay_evidence;
  const activeAssays = ensureArray<BioassayItem>(bioassay?.active_assays);

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
