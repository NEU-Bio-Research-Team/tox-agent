const topAtoms = [
  { idx: 3, symbol: 'O', contribution: 0.82 },
  { idx: 7, symbol: 'N', contribution: 0.71 },
  { idx: 2, symbol: 'C', contribution: 0.54 },
  { idx: 9, symbol: 'C', contribution: 0.41 },
];

const topBonds = [
  { bond: 'O-C', atoms: '3-7', contribution: 0.79 },
  { bond: 'C=O', atoms: '2-3', contribution: 0.65 },
  { bond: 'C-N', atoms: '7-11', contribution: 0.48 },
];

export function StructuralExplanationSection() {
  return (
    <section id="structural">
      <h2 className="text-2xl font-bold mb-6" style={{ color: 'var(--text)' }}>
        §3 Structural Explanation
      </h2>

      {/* Heatmap */}
      <div className="rounded-xl p-6 mb-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2 mb-4">
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2L2 7L12 12L22 7L12 2Z" />
            <path d="M2 17L12 22L22 17" />
            <path d="M2 12L12 17L22 12" />
          </svg>
          <span className="font-semibold" style={{ color: 'var(--text)' }}>Molecular Heatmap</span>
        </div>
        
        <div className="rounded-lg mb-4 flex items-center justify-center h-64" style={{ backgroundColor: 'var(--surface-alt)', border: '1px solid var(--border)' }}>
          <div className="text-center" style={{ color: 'var(--text-faint)' }}>
            <svg className="w-16 h-16 mx-auto mb-2 opacity-30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="12" cy="12" r="3" />
              <path d="M12 1v6m0 6v6m0-6h6m-6 0H6" />
            </svg>
            <p className="text-sm">GNN Attribution Heatmap</p>
          </div>
        </div>

        <div className="flex items-center justify-between text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: 'var(--accent-green)' }} />
            <span>Ít nguy hiểm</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: 'var(--accent-red)' }} />
            <span>Đóng góp độc tính cao</span>
          </div>
        </div>
      </div>

      {/* Top Atoms and Bonds */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Top Atoms */}
        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <h3 className="font-semibold mb-4" style={{ color: 'var(--text)' }}>Top Atoms</h3>
          <div className="space-y-3">
            <div className="grid grid-cols-[40px_40px_1fr_40px] gap-2 text-xs font-semibold pb-2" style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}>
              <div>Idx</div>
              <div>Sym</div>
              <div>Score</div>
              <div></div>
            </div>
            {topAtoms.map((atom, idx) => {
              const dotColor = atom.contribution >= 0.7 ? 'var(--accent-red)' : atom.contribution >= 0.4 ? 'var(--accent-yellow)' : 'var(--accent-green)';
              return (
                <div key={idx} className="grid grid-cols-[40px_40px_1fr_40px] gap-2 text-sm items-center">
                  <div style={{ color: 'var(--text)' }}>{atom.idx}</div>
                  <div className="font-mono font-semibold" style={{ color: 'var(--text)' }}>{atom.symbol}</div>
                  <div className="font-mono" style={{ color: 'var(--text)' }}>{atom.contribution.toFixed(2)}</div>
                  <div>
                    <span style={{ color: dotColor, fontSize: '16px' }}>●</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Top Bonds */}
        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <h3 className="font-semibold mb-4" style={{ color: 'var(--text)' }}>Top Bonds</h3>
          <div className="space-y-3">
            <div className="grid grid-cols-[50px_50px_1fr_40px] gap-2 text-xs font-semibold pb-2" style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}>
              <div>Bond</div>
              <div>Atoms</div>
              <div>Score</div>
              <div></div>
            </div>
            {topBonds.map((bond, idx) => {
              const dotColor = bond.contribution >= 0.7 ? 'var(--accent-red)' : bond.contribution >= 0.4 ? 'var(--accent-yellow)' : 'var(--accent-green)';
              return (
                <div key={idx} className="grid grid-cols-[50px_50px_1fr_40px] gap-2 text-sm items-center">
                  <div className="font-mono" style={{ color: 'var(--text)' }}>{bond.bond}</div>
                  <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{bond.atoms}</div>
                  <div className="font-mono" style={{ color: 'var(--text)' }}>{bond.contribution.toFixed(2)}</div>
                  <div>
                    <span style={{ color: dotColor, fontSize: '16px' }}>●</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
