import { MoleculeDepiction } from './MoleculeDepiction';

const ATOM_ORDER_VERSION = 'rdkit-output-order-v1';

/** A single-hue sequential ramp keyed to magnitude only (D-XAI-7). No
 * red/green, no toxic/safe colour coding. Alpha ∝ relative_importance / max. */
function rampAlpha(value: number, max: number): number {
  if (max <= 0) return 0;
  return 0.12 + 0.78 * Math.min(1, value / max);
}

export interface AtomHighlightAtom {
  atom_index: number;
  symbol: string;
  relative_importance: number;
}

/**
 * PR-C3 spike outcome: `smiles-drawer`'s `highlight_atoms` keys off SMILES
 * atom-map classes (`[C:1]`), not positional atom index, so it cannot align to
 * the `atom_index` contract (D-XAI-4) without rewriting the string the panel is
 * required to depict. So the depiction stays the plain 2D structure and the
 * per-atom attribution is shown as a ranked magnitude bar list beside it —
 * which is index-exact by construction. `atom_order_version` is still checked:
 * a mismatch means even the bar list's indices are untrustworthy.
 */
export function AtomHighlightDepiction({
  smiles,
  atomOrderVersion,
  atoms,
  size = 220,
  topK = 8,
}: {
  /** MUST be `canonical_smiles` from the explain response. */
  smiles: string;
  atomOrderVersion: string | null;
  atoms: AtomHighlightAtom[];
  size?: number;
  topK?: number;
}) {
  const aligned = atomOrderVersion === ATOM_ORDER_VERSION;
  const ranked = [...atoms]
    .sort((a, b) => b.relative_importance - a.relative_importance)
    .slice(0, topK);
  const max = ranked.length > 0 ? ranked[0].relative_importance : 0;

  return (
    <div className="space-y-3">
      <div className="flex justify-center">
        <MoleculeDepiction smiles={smiles} size={size} />
      </div>

      {!aligned ? (
        <p className="text-xs" style={{ color: 'var(--accent-yellow)' }}>
          Không căn được attribution theo nguyên tử cho phiên bản thứ tự nguyên tử
          này ({atomOrderVersion ?? 'không rõ'}); chỉ hiển thị cấu trúc và danh sách token.
        </p>
      ) : (
        <ul className="space-y-1">
          {ranked.map((atom) => (
            <li key={atom.atom_index} className="flex items-center gap-2 text-xs">
              <span
                className="w-14 shrink-0 font-mono"
                style={{ color: 'var(--text-muted)' }}
              >
                {atom.symbol}
                <span style={{ color: 'var(--text-faint)' }}>#{atom.atom_index}</span>
              </span>
              <span className="relative h-3 flex-1 overflow-hidden rounded" style={{ backgroundColor: 'var(--surface-alt)' }}>
                <span
                  className="absolute inset-y-0 left-0 rounded"
                  style={{
                    width: `${Math.max(2, (atom.relative_importance / (max || 1)) * 100)}%`,
                    backgroundColor: 'var(--accent-blue)',
                    opacity: rampAlpha(atom.relative_importance, max),
                  }}
                />
              </span>
              <span className="w-12 shrink-0 text-right font-mono" style={{ color: 'var(--text-faint)' }}>
                {(atom.relative_importance * 100).toFixed(1)}%
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
