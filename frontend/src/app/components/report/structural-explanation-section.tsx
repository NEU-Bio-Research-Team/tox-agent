import type { StructuralSection } from '../../../lib/api';

interface StructuralExplanationSectionProps {
  data: StructuralSection;
  language: 'vi' | 'en';
}

export function StructuralExplanationSection({ data, language }: StructuralExplanationSectionProps) {
  const topAtoms = data?.top_atoms ?? [];
  const topBonds = data?.top_bonds ?? [];
  const heatmap = data?.heatmap_base64 || null;
  const moleculePng = data?.molecule_png_base64 || null;
  const heatmapSrc = heatmap ? `data:image/png;base64,${heatmap}` : null;
  const moleculeSrc = moleculePng ? `data:image/png;base64,${moleculePng}` : null;

  return (
    <section id="structural" className="scroll-mt-24 lg:scroll-mt-20">
      <h2 className="text-2xl font-bold mb-6" style={{ color: 'var(--text)' }}>
        {language === 'vi' ? '§3 Giải thích cấu trúc' : '§3 Structural Explanation'}
      </h2>

      <div className="rounded-xl p-6 mb-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2 mb-4">
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2L2 7L12 12L22 7L12 2Z" />
            <path d="M2 17L12 22L22 17" />
            <path d="M2 12L12 17L22 12" />
          </svg>
          <span className="font-semibold" style={{ color: 'var(--text)' }}>
            {language === 'vi' ? 'Molecular Structure + Heatmap' : 'Molecular Structure + Heatmap'}
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
          <div className="rounded-lg p-3" style={{ backgroundColor: 'var(--surface-alt)', border: '1px solid var(--border)' }}>
            <p className="text-xs uppercase mb-2" style={{ color: 'var(--text-muted)' }}>
              {language === 'vi' ? 'Cấu trúc hóa học (SMILES)' : 'Chemical Structure (SMILES)'}
            </p>
            <div className="flex items-center justify-center min-h-56">
              {moleculeSrc ? (
                <img
                  src={moleculeSrc}
                  alt="Molecule structure"
                  className="w-full rounded-lg"
                />
              ) : (
                <p className="text-sm" style={{ color: 'var(--text-faint)' }}>
                  {language === 'vi' ? 'Chưa có ảnh cấu trúc phân tử.' : 'Molecule image is not available.'}
                </p>
              )}
            </div>
          </div>

          <div className="rounded-lg p-3" style={{ backgroundColor: 'var(--surface-alt)', border: '1px solid var(--border)' }}>
            <p className="text-xs uppercase mb-2" style={{ color: 'var(--text-muted)' }}>
              {language === 'vi' ? 'Heatmap đóng góp độc tính' : 'Toxicity Attribution Heatmap'}
            </p>
            <div className="flex items-center justify-center min-h-56">
              {heatmapSrc ? (
                <img
                  src={heatmapSrc}
                  alt="GNN attribution heatmap"
                  className="w-full rounded-lg"
                />
              ) : (
                <p className="text-sm" style={{ color: 'var(--text-faint)' }}>
                  {language === 'vi' ? 'Chưa có dữ liệu heatmap.' : 'Heatmap is not available.'}
                </p>
              )}
            </div>
          </div>
        </div>

        {!moleculeSrc && !heatmapSrc && (
          <div className="text-center" style={{ color: 'var(--text-faint)' }}>
            <svg className="w-16 h-16 mx-auto mb-2 opacity-30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="12" cy="12" r="3" />
              <path d="M12 1v6m0 6v6m0-6h6m-6 0H6" />
            </svg>
            <p className="text-sm">
              {language === 'vi' ? 'Chưa có dữ liệu heatmap/molecule image' : 'No heatmap or molecule image available'}
            </p>
          </div>
        )}

        {data?.explainer_note && (
          <p className="text-xs mb-4" style={{ color: 'var(--text-faint)' }}>
            {data.explainer_note}
          </p>
        )}

        <div className="mb-2 flex flex-col gap-2 text-xs sm:flex-row sm:items-center sm:justify-between" style={{ color: 'var(--text-muted)' }}>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: 'var(--accent-green)' }} />
            <span>{language === 'vi' ? 'Ít nguy hiểm' : 'Lower risk contribution'}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: 'var(--accent-red)' }} />
            <span>{language === 'vi' ? 'Đóng góp độc tính cao' : 'Higher toxicity contribution'}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <h3 className="font-semibold mb-4" style={{ color: 'var(--text)' }}>{language === 'vi' ? 'Top nguyên tử' : 'Top Atoms'}</h3>
          <div className="space-y-3">
            <div className="grid grid-cols-[40px_40px_1fr_40px] gap-2 text-xs font-semibold pb-2" style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}>
              <div>Idx</div>
              <div>Sym</div>
              <div>Score</div>
              <div></div>
            </div>
            {topAtoms.length === 0 && (
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                {language === 'vi' ? 'Không có dữ liệu atom attribution.' : 'No atom attribution data.'}
              </p>
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
          <h3 className="font-semibold mb-4" style={{ color: 'var(--text)' }}>{language === 'vi' ? 'Top liên kết' : 'Top Bonds'}</h3>
          <div className="space-y-3">
            <div className="grid grid-cols-[70px_70px_1fr_40px] gap-2 text-xs font-semibold pb-2" style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}>
              <div>Bond</div>
              <div>Atoms</div>
              <div>Score</div>
              <div></div>
            </div>
            {topBonds.length === 0 && (
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                {language === 'vi' ? 'Không có dữ liệu bond attribution.' : 'No bond attribution data.'}
              </p>
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
