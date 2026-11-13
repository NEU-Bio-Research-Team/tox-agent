import { useState } from 'react';
import { WandSparkles } from 'lucide-react';
import { CanvasMoleculeEditor, type CanvasEditorOnChangeMolecule } from 'react-ocl';
import { Button } from '../../shared/ui/button';

/**
 * The "Vẽ" tab: the same client-side openchemlib editor the
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
      <p className="text-13 text-ink-secondary">Vẽ cấu trúc, rồi dùng SMILES của nó để phân tích.</p>

      <div className="overflow-hidden rounded-xl border border-line bg-depiction">
        <div className="w-full" style={{ minHeight: 320, height: 'clamp(320px, 44vh, 420px)' }}>
          <CanvasMoleculeEditor width="100%" height="100%" onChange={handleChange} />
        </div>
      </div>

      <p className="truncate font-mono text-xs text-ink-secondary">{pendingSmiles || 'Chưa vẽ cấu trúc nào.'}</p>

      <Button type="button" variant="outline" className="w-full gap-2" onClick={() => onSmilesExtracted(pendingSmiles)} disabled={disabled || !pendingSmiles}>
        <WandSparkles />
        Dùng SMILES đã vẽ
      </Button>
    </div>
  );
}
