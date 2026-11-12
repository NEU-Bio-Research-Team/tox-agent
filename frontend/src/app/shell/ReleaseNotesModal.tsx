import { FlaskConical, Sparkles } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '../../shared/ui/dialog';
import { Button } from '../../shared/ui/button';
import { APP_COMMIT, APP_VERSION, BUILD_TIME_LABEL } from '../build-info';
import { RELEASE_NOTES_ITEMS } from '../release-notes';

export function ReleaseNotesModal({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="legacy-ui sm:max-w-2xl">
        <DialogHeader>
          <div className="mb-1 flex items-center gap-3">
            <div
              className="inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[11px] font-semibold uppercase tracking-wide"
              style={{
                borderColor: 'var(--accent-blue)',
                backgroundColor: 'var(--accent-blue-muted)',
                color: 'var(--accent-blue)',
              }}
            >
              <FlaskConical className="h-3 w-3" />
              Beta
            </div>
            <span className="text-xs" style={{ color: 'var(--text-faint)' }}>
              v{APP_VERSION} • {APP_COMMIT} • {BUILD_TIME_LABEL}
            </span>
          </div>
          <DialogTitle className="text-2xl" style={{ color: 'var(--text)' }}>
            <span className="inline-flex items-center gap-2">
              <Sparkles className="h-5 w-5" style={{ color: 'var(--accent-blue)' }} />
              Release Notes
            </span>
          </DialogTitle>
          <DialogDescription style={{ color: 'var(--text-muted)' }}>
            Production update highlights for this build.
          </DialogDescription>
        </DialogHeader>

        <ul className="list-disc space-y-3 pl-5 text-sm" style={{ color: 'var(--text)' }}>
          {RELEASE_NOTES_ITEMS.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>

        <div className="flex justify-end pt-2">
          <Button
            type="button"
            onClick={() => onOpenChange(false)}
            style={{ backgroundColor: 'var(--accent-blue)', color: '#ffffff' }}
          >
            Continue
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
