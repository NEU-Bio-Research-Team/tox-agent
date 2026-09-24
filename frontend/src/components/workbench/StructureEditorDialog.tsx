import { useState } from 'react';
import { CanvasMoleculeEditor, type CanvasEditorOnChangeMolecule } from 'react-ocl';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '../ui/dialog';
import { Button } from '../ui/button';

/**
 * A full 2D structure editor (atoms, bonds, templates, undo) rendered
 * entirely client-side by openchemlib's canvas editor — no backend call, no
 * new predictor endpoint. The SMILES it produces feeds the composer's
 * existing `smiles` field, so it rides the same submit path a pasted SMILES
 * string already does.
 */
export function StructureEditorDialog({
  open,
  onOpenChange,
  onConfirm,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: (smiles: string) => void;
}) {
  const [pendingSmiles, setPendingSmiles] = useState('');

  const handleChange = (event: CanvasEditorOnChangeMolecule) => {
    setPendingSmiles(event.getSmiles());
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) setPendingSmiles('');
        onOpenChange(next);
      }}
    >
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Vẽ cấu trúc phân tử</DialogTitle>
          <DialogDescription>
            Vẽ cấu trúc 2D bằng công cụ bên dưới rồi dùng SMILES sinh ra để phân tích.
          </DialogDescription>
        </DialogHeader>

        <div
          className="h-[420px] w-full overflow-hidden rounded-lg border"
          style={{ borderColor: 'var(--border)', backgroundColor: '#ffffff' }}
        >
          <CanvasMoleculeEditor width="100%" height="100%" onChange={handleChange} />
        </div>

        <p className="truncate font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
          {pendingSmiles || 'Chưa có cấu trúc nào được vẽ.'}
        </p>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Huỷ
          </Button>
          <Button
            disabled={!pendingSmiles}
            onClick={() => {
              onConfirm(pendingSmiles);
              onOpenChange(false);
              setPendingSmiles('');
            }}
          >
            Dùng cấu trúc này
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
