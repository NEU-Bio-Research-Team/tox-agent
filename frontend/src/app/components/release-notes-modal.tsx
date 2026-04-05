import { FlaskConical, Sparkles } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from './ui/dialog';
import { Button } from './ui/button';
import { RELEASE_NOTES_ITEMS } from '../release-notes';

interface ReleaseNotesModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const BUILD_TIME_LABEL = new Date(__APP_BUILD_TIME__).toLocaleString('en-GB', {
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
  timeZone: 'UTC',
});

export function ReleaseNotesModal({ open, onOpenChange }: ReleaseNotesModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <div className="flex items-center gap-3 mb-1">
            <div
              className="inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[11px] font-semibold uppercase tracking-wide"
              style={{
                borderColor: 'var(--accent-blue)',
                backgroundColor: 'var(--accent-blue-muted)',
                color: 'var(--accent-blue)',
              }}
            >
              <FlaskConical className="w-3 h-3" />
              Beta
            </div>
            <span className="text-xs" style={{ color: 'var(--text-faint)' }}>
              v{__APP_VERSION__} • {BUILD_TIME_LABEL} UTC
            </span>
          </div>
          <DialogTitle className="text-2xl" style={{ color: 'var(--text)' }}>
            <span className="inline-flex items-center gap-2">
              <Sparkles className="w-5 h-5" style={{ color: 'var(--accent-blue)' }} />
              Release Notes
            </span>
          </DialogTitle>
          <DialogDescription style={{ color: 'var(--text-muted)' }}>
            Production update highlights for this build.
          </DialogDescription>
        </DialogHeader>

        <ul className="space-y-3 text-sm pl-5 list-disc" style={{ color: 'var(--text)' }}>
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
