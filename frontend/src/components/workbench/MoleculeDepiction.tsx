import { useEffect, useRef, useState } from 'react';
import SmilesDrawer from 'smiles-drawer';

/**
 * Client-side 2D depiction only — a pragmatic substitute for the RDKit.js
 * choice in the redesign plan's D-5: `smiles-drawer` is pure JS (no WASM
 * fetch/instantiate step), which keeps the workbench's first paint fast. The
 * architectural point of D-5 is unchanged either way: no new backend
 * endpoint, `toxpred` stays a stateless predictor that never renders
 * anything for the UI.
 */
export function MoleculeDepiction({ smiles, size = 220 }: { smiles: string; size?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
    const canvas = canvasRef.current;
    if (!canvas || !smiles) return;

    SmilesDrawer.parse(
      smiles,
      (tree) => {
        const drawer = new SmilesDrawer.Drawer({ width: size, height: size, bondThickness: 1.2 });
        drawer.draw(tree, canvas, 'light');
      },
      () => setFailed(true),
    );
  }, [smiles, size]);

  if (failed) {
    return (
      <div
        className="flex items-center justify-center rounded-lg text-xs"
        style={{ width: size, height: size, backgroundColor: 'var(--surface-alt)', color: 'var(--text-faint)' }}
      >
        Không vẽ được cấu trúc
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      className="rounded-lg"
      style={{ backgroundColor: '#ffffff' }}
    />
  );
}
