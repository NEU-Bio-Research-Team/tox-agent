const papers = [
  {
    title: 'Aspirin toxicity in hepatocytes: a mechanistic study',
    authors: 'Smith J., et al.',
    journal: 'Journal of Toxicology',
    year: 2023,
    pmid: '37291847',
    snippet: 'Aspirin at high concentrations showed minimal hepatotoxic effects...'
  },
  {
    title: 'NSAID safety and mechanism of action',
    authors: 'Johnson M., Davis L.',
    journal: 'Clinical Pharmacology Review',
    year: 2024,
    pmid: '38456123',
    snippet: 'Non-steroidal anti-inflammatory drugs demonstrate varied safety profiles...'
  },
];

const bioassays = [
  { aid: '588342', name: 'Tox21 HSE pathway', outcome: 'INACTIVE', target: 'SR-HSE' },
  { aid: '743219', name: 'Tox21 MMP pathway', outcome: 'INACTIVE', target: 'SR-MMP' },
  { aid: '602387', name: 'NR-AR nuclear receptor', outcome: 'INACTIVE', target: 'NR-AR' },
];

export function LiteratureContextSection() {
  return (
    <section id="literature">
      <h2 className="text-2xl font-bold mb-6" style={{ color: 'var(--text)' }}>
        §4 Literature Context
      </h2>

      {/* PubChem Info */}
      <div className="rounded-xl p-6 mb-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
        <h3 className="font-semibold mb-4" style={{ color: 'var(--text)' }}>PubChem Compound Info</h3>
        <div className="space-y-2 text-sm">
          <p style={{ color: 'var(--text)' }}>
            <span style={{ color: 'var(--text-muted)' }}>CID:</span> <span className="font-mono font-semibold">2244</span> · Aspirin
          </p>
          <p style={{ color: 'var(--text)' }}>
            <span style={{ color: 'var(--text-muted)' }}>Formula:</span> <span className="font-mono">C9H8O4</span> · <span style={{ color: 'var(--text-muted)' }}>MW:</span> 180.16 g/mol
          </p>
          <p style={{ color: 'var(--text)' }}>
            <span style={{ color: 'var(--text-muted)' }}>IUPAC:</span> <span className="italic">2-(acetyloxy)benzoic acid</span>
          </p>
          <p style={{ color: 'var(--text-muted)' }}>
            <span>Synonyms:</span> Acetylsalicylic acid, Aspirin, ASA
          </p>
          <a 
            href="https://pubchem.ncbi.nlm.nih.gov/compound/2244" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 mt-2 text-sm"
            style={{ color: 'var(--accent-blue)' }}
          >
            Xem trên PubChem
            <svg className="w-3 h-3" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M10 2L2 10M10 2H4M10 2V8" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </a>
        </div>
      </div>

      {/* Literature Papers */}
      <div className="mb-6">
        <h3 className="font-semibold mb-4" style={{ color: 'var(--text)' }}>
          Nghiên cứu liên quan ({papers.length} bài báo)
        </h3>
        <div className="space-y-3">
          {papers.map((paper, idx) => (
            <div 
              key={idx} 
              className="rounded-xl p-5 transition-colors"
              style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}
            >
              <div className="flex items-start justify-between gap-3 mb-2">
                <p className="font-semibold flex-1" style={{ color: 'var(--text)' }}>
                  [{idx + 1}] {paper.title}
                </p>
              </div>
              <p className="text-sm mb-1" style={{ color: 'var(--text-muted)' }}>
                {paper.authors} · <span className="italic">{paper.journal}</span> ({paper.year})
              </p>
              <p className="text-sm mb-2 italic" style={{ color: 'var(--text-faint)' }}>
                "{paper.snippet}"
              </p>
              <div className="flex items-center gap-3">
                <span className="font-mono text-xs px-2 py-1 rounded" style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}>
                  PMID: {paper.pmid}
                </span>
                <a 
                  href={`https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs flex items-center gap-1"
                  style={{ color: 'var(--accent-blue)' }}
                >
                  Đọc bài báo
                  <svg className="w-3 h-3" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M10 2L2 10M10 2H4M10 2V8" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Bioassay Data */}
      <div>
        <h3 className="font-semibold mb-4" style={{ color: 'var(--text)' }}>Bioassay Data</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm" style={{ borderCollapse: 'separate', borderSpacing: 0 }}>
            <thead>
              <tr style={{ backgroundColor: 'var(--surface-alt)' }}>
                <th className="text-left p-3 rounded-tl-lg" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>AID</th>
                <th className="text-left p-3" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Tên assay</th>
                <th className="text-left p-3" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Kết quả</th>
                <th className="text-left p-3 rounded-tr-lg" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Target</th>
              </tr>
            </thead>
            <tbody>
              {bioassays.map((assay, idx) => (
                <tr 
                  key={idx}
                  style={{ 
                    backgroundColor: assay.outcome === 'ACTIVE' ? 'rgba(239,68,68,0.05)' : 'transparent',
                    borderTop: '1px solid var(--border)'
                  }}
                >
                  <td className="p-3 font-mono" style={{ color: 'var(--text)' }}>{assay.aid}</td>
                  <td className="p-3" style={{ color: 'var(--text)' }}>{assay.name}</td>
                  <td className="p-3">
                    <span className="flex items-center gap-2">
                      <span style={{ 
                        color: assay.outcome === 'ACTIVE' ? 'var(--accent-red)' : 'var(--text-faint)',
                        fontSize: '12px'
                      }}>
                        {assay.outcome === 'ACTIVE' ? '●' : '○'}
                      </span>
                      <span style={{ color: assay.outcome === 'ACTIVE' ? 'var(--accent-red)' : 'var(--text-muted)' }}>
                        {assay.outcome}
                      </span>
                    </span>
                  </td>
                  <td className="p-3 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>{assay.target}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
