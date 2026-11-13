import { FlaskConical, Sparkles } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '../../shared/ui/dialog';
import { Button } from '../../shared/ui/button';
import { StatusPill } from '../../shared/ui/status-pill';
import { APP_COMMIT, APP_VERSION, BUILD_TIME_LABEL } from '../build-info';
import { RELEASE_NOTES_ITEMS } from '../release-notes';

export function ReleaseNotesModal({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <div className="mb-1 flex flex-wrap items-center gap-3">
            <StatusPill tone="brand" className="uppercase tracking-wide">
              <FlaskConical className="size-3" />
              Beta
            </StatusPill>
            <span className="font-mono text-xs text-ink-tertiary">
              v{APP_VERSION} · {APP_COMMIT} · {BUILD_TIME_LABEL}
            </span>
          </div>
          <DialogTitle className="text-2xl text-ink">
            <span className="inline-flex items-center gap-2">
              <Sparkles className="size-5 text-link" />
              Ghi chú phát hành
            </span>
          </DialogTitle>
          <DialogDescription className="text-ink-secondary">Điểm nổi bật của bản này.</DialogDescription>
        </DialogHeader>

        <ul className="list-disc space-y-3 pl-5 text-sm text-ink">
          {RELEASE_NOTES_ITEMS.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>

        <div className="flex justify-end pt-2">
          <Button type="button" onClick={() => onOpenChange(false)}>
            Đóng
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
