import { useState } from 'react';
import { WandSparkles } from 'lucide-react';
import { CanvasMoleculeEditor, type CanvasEditorOnChangeMolecule } from 'react-ocl';
import { Button } from '../../shared/ui/button';

/**
 * The "Draw Molecule" tab: the same client-side openchemlib editor the
 * composer's dialog uses, laid out inline the way the production page had its
 * editor. Nothing is sent anywhere while drawing; the SMILES reaches the
 * analysis input only when the user asks for it, so a half-drawn structure
 * never replaces what they typed.
 */
export function StructureDrawPanel({
  onSmilesExtracted,
  disabled = false,
}: {
  onSmilesExtracted: (smiles: string) => void;
  disabled?: boolean;
}) {
  const [pendingSmiles, setPendingSmiles] = useState('');

  const handleChange = (event: CanvasEditorOnChangeMolecule) => {
    setPendingSmiles(event.getSmiles());
  };

  return (
    <div className="space-y-3">
      <div
        className="flex flex-col gap-2 rounded-lg border p-3 md:flex-row md:items-center md:justify-between"
        style={{ borderColor: 'var(--border)', backgroundColor: 'var(--surface-alt)' }}
      >
        <div className="text-sm" style={{ color: 'var(--text-muted)' }}>
          Draw a structure, then use its SMILES for analysis.
        </div>
        <Button
          type="button"
          onClick={() => onSmilesExtracted(pendingSmiles)}
          disabled={disabled || !pendingSmiles}
          className="bg-gradient-to-r from-emerald-600 to-teal-600 text-white hover:from-emerald-700 hover:to-teal-700"
        >
          <WandSparkles className="mr-2 h-4 w-4" />
          Use Drawn SMILES
        </Button>
      </div>

      <div className="overflow-hidden rounded-xl border" style={{ borderColor: 'var(--border)', backgroundColor: '#ffffff' }}>
        <div className="w-full" style={{ minHeight: 420, height: 'clamp(420px, 58vh, 520px)' }}>
          <CanvasMoleculeEditor width="100%" height="100%" onChange={handleChange} />
        </div>
      </div>

      <p className="truncate font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
        {pendingSmiles || 'No structure drawn yet.'}
      </p>
    </div>
  );
}
