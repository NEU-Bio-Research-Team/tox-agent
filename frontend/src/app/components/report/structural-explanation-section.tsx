import type { StructuralSection } from '../../../lib/api';

interface StructuralExplanationSectionProps {
  data: StructuralSection;
}

export function StructuralExplanationSection({ data }: StructuralExplanationSectionProps) {
  const topAtoms = data?.top_atoms ?? [];
  const topBonds = data?.top_bonds ?? [];
  const heatmap = data?.heatmap_base64 || null;

  return (
    <section id="structural">
      <h2 className="text-2xl font-bold mb-6" style={{ color: 'var(--text)' }}>
        §3 Structural Explanation
      </h2>

      <div className="rounded-xl p-6 mb-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2 mb-4">
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2L2 7L12 12L22 7L12 2Z" />
            <path d="M2 17L12 22L22 17" />
            <path d="M2 12L12 17L22 12" />
          </svg>
          <span className="font-semibold" style={{ color: 'var(--text)' }}>Molecular Heatmap</span>
        </div>

        <div className="rounded-lg mb-4 flex items-center justify-center min-h-64" style={{ backgroundColor: 'var(--surface-alt)', border: '1px solid var(--border)' }}>
          {heatmap ? (
            <img
              src={`data:image/png;base64,${heatmap}`}
              alt="GNN Attribution Heatmap"
              className="w-full rounded-lg"
            />
          ) : (
            <div className="text-center" style={{ color: 'var(--text-faint)' }}>
              <svg className="w-16 h-16 mx-auto mb-2 opacity-30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 1v6m0 6v6m0-6h6m-6 0H6" />
              </svg>
              <p className="text-sm">Chua co du lieu heatmap_base64</p>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: 'var(--accent-green)' }} />
            <span>It nguy hiem</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: 'var(--accent-red)' }} />
            <span>Dong gop doc tinh cao</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <h3 className="font-semibold mb-4" style={{ color: 'var(--text)' }}>Top Atoms</h3>
          <div className="space-y-3">
            <div className="grid grid-cols-[40px_40px_1fr_40px] gap-2 text-xs font-semibold pb-2" style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}>
              <div>Idx</div>
              <div>Sym</div>
              <div>Score</div>
              <div></div>
            </div>
            {topAtoms.length === 0 && (
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Khong co du lieu atom attribution.</p>
            )}
            {topAtoms.map((atom) => {
              const contribution = Number(atom.importance ?? 0);
              const dotColor = contribution >= 0.7 ? 'var(--accent-red)' : contribution >= 0.4 ? 'var(--accent-yellow)' : 'var(--accent-green)';
              return (
                <div key={`${atom.atom_idx}-${atom.element}`} className="grid grid-cols-[40px_40px_1fr_40px] gap-2 text-sm items-center">
                  <div style={{ color: 'var(--text)' }}>{atom.atom_idx}</div>
                  <div className="font-mono font-semibold" style={{ color: 'var(--text)' }}>{atom.element}</div>
                  <div className="font-mono" style={{ color: 'var(--text)' }}>{contribution.toFixed(2)}</div>
                  <div>
                    <span style={{ color: dotColor, fontSize: '16px' }}>●</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <h3 className="font-semibold mb-4" style={{ color: 'var(--text)' }}>Top Bonds</h3>
          <div className="space-y-3">
            <div className="grid grid-cols-[70px_70px_1fr_40px] gap-2 text-xs font-semibold pb-2" style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}>
              <div>Bond</div>
              <div>Atoms</div>
              <div>Score</div>
              <div></div>
            </div>
            {topBonds.length === 0 && (
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Khong co du lieu bond attribution.</p>
            )}
            {topBonds.map((bond, idx) => {
              const contribution = Number(bond.importance ?? 0);
              const dotColor = contribution >= 0.7 ? 'var(--accent-red)' : contribution >= 0.4 ? 'var(--accent-yellow)' : 'var(--accent-green)';
              return (
                <div key={`${bond.atom_pair || idx}-${bond.bond_type || 'bond'}`} className="grid grid-cols-[70px_70px_1fr_40px] gap-2 text-sm items-center">
                  <div className="font-mono" style={{ color: 'var(--text)' }}>{bond.bond_type || 'N/A'}</div>
                  <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{bond.atom_pair || 'N/A'}</div>
                  <div className="font-mono" style={{ color: 'var(--text)' }}>{contribution.toFixed(2)}</div>
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
