import { ChevronRight, Loader2 } from 'lucide-react';
import { Button } from '../../shared/ui/button';

/** Quick analysis keeps nothing. This is the way out of that: it opens a session
 * with the same molecule so the numbers get an audit trail and the agent can be
 * asked about them. */
export function SaveBar({
  saving,
  onSave,
}: {
  saving: 'save' | 'ask' | null;
  onSave: (askAgent: boolean) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-3 rounded-2xl border border-brand-soft-strong bg-brand-soft px-5 py-4">
      <div className="min-w-[240px] flex-1">
        <h3 className="text-sm font-semibold text-ink">Kết quả này chưa được lưu</h3>
        <p className="mt-0.5 text-13 leading-snug text-ink-secondary">
          Mở session để giữ lại, hỏi agent về từng con số và có dấu vết kiểm chứng.
        </p>
      </div>
      <Button variant="outline" className="border-brand-border bg-surface-solid text-brand-ink" disabled={saving !== null} onClick={() => onSave(true)}>
        {saving === 'ask' ? <Loader2 className="animate-spin" /> : null}
        Hỏi agent về kết quả
      </Button>
      <Button className="gap-1.5" disabled={saving !== null} onClick={() => onSave(false)}>
        {saving === 'save' ? <Loader2 className="animate-spin" /> : null}
        Lưu vào session mới
        <ChevronRight />
      </Button>
    </div>
  );
}
