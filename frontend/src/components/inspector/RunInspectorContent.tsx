import { useSearchParams } from 'react-router';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Button } from '../ui/button';
import { cancelRun, getRun } from '../../lib/api/endpoints';
import { RUN_STATUS_LABEL_VI } from '../../lib/labels';
import { scrollToRun } from '../../lib/scrollToRun';
import { RunTimelineTab } from './RunTimelineTab';
import { RuntimeManifestTab } from './RuntimeManifestTab';
import { ValidationTab } from './ValidationTab';
import { RawJsonTab } from './RawJsonTab';

const TAB_VALUES = ['timeline', 'runtime', 'validation', 'raw'] as const;
type TabValue = (typeof TAB_VALUES)[number];

function isTabValue(value: string | null): value is TabValue {
  return value !== null && (TAB_VALUES as readonly string[]).includes(value);
}

/**
 * Run viewer content, extracted from the former `RunInspectorDrawer` Sheet.
 * Plan section 8.5: on desktop this now lives inline in `ArtifactsPanel`
 * (no drawer covering chat); on tablet/mobile the panel itself becomes a
 * Sheet, so this component never wraps its own overlay.
 */
export function RunInspectorContent({ sessionId, runId }: { sessionId: string; runId: string }) {
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedTab = searchParams.get('tab');
  const activeTab: TabValue = isTabValue(requestedTab) ? requestedTab : 'timeline';

  const query = useQuery({
    queryKey: ['run', sessionId, runId],
    queryFn: () => getRun(sessionId, runId),
  });

  const cancelMutation = useMutation({
    mutationFn: () => cancelRun(sessionId, runId),
    onSuccess: (result) => {
      void queryClient.invalidateQueries({ queryKey: ['run', sessionId, runId] });
      void queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
      toast.info(
        result.runtime_cancel_supported
          ? 'Đã yêu cầu huỷ run.'
          : 'Đã yêu cầu dừng; runtime không hỗ trợ huỷ giữa lượt nên tiến trình sẽ bị chấm dứt.',
      );
    },
  });

  const run = query.data;
  const cancelRequested = cancelMutation.data?.requested === true;
  const canCancel = !cancelRequested && run && (run.status === 'queued' || run.status === 'running' || run.status === 'validating');

  return (
    <div className="space-y-4">
      <div>
        <p className="font-mono text-sm" style={{ color: 'var(--text)' }}>
          {runId}
        </p>
        <p className="text-xs" style={{ color: 'var(--text-faint)' }} role="status" aria-live="polite" aria-atomic="true">
          {run ? `${RUN_STATUS_LABEL_VI[run.status]} · ${run.lane} · ${run.intent}` : 'Đang tải…'}
        </p>
        <button
          type="button"
          onClick={() => scrollToRun(runId)}
          className="mt-1 text-xs font-medium underline"
          style={{ color: 'var(--accent-blue)' }}
        >
          về lượt chat tạo kết quả
        </button>
      </div>

      {canCancel && (
        <Button variant="outline" size="sm" disabled={cancelMutation.isPending} onClick={() => cancelMutation.mutate()}>
          Huỷ run
        </Button>
      )}
      {cancelRequested && run && run.status !== 'cancelled' && (
        <p className="rounded-lg border p-2 text-xs" style={{ borderColor: 'var(--accent-yellow)', color: 'var(--text-muted)' }}>
          Đã gửi yêu cầu huỷ; trạng thái sẽ đổi thành “đã huỷ” sau khi control plane xác nhận. Chưa coi run là đã huỷ.
        </p>
      )}

      {query.isLoading && <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Đang tải…</p>}
      {query.isError && <p className="text-sm" style={{ color: 'var(--accent-red)' }}>Không tải được run này.</p>}

      {run && (
        <Tabs
          value={activeTab}
          onValueChange={(value) => {
            const next = new URLSearchParams(searchParams);
            next.set('tab', value);
            setSearchParams(next, { replace: true });
          }}
        >
          {/* The artifacts panel is often narrower than the four tab labels
              combined (it used to live in a >=576px Sheet before this
              viewer moved inline) — scroll horizontally instead of clipping
              "JSON thô" off the edge. */}
          <div className="overflow-x-auto">
            <TabsList>
              <TabsTrigger value="timeline">Dòng thời gian</TabsTrigger>
              <TabsTrigger value="runtime">Runtime</TabsTrigger>
              <TabsTrigger value="validation">Kiểm định</TabsTrigger>
              <TabsTrigger value="raw">JSON thô</TabsTrigger>
            </TabsList>
          </div>
          <TabsContent value="timeline" className="pt-4">
            <RunTimelineTab run={run} />
          </TabsContent>
          <TabsContent value="runtime" className="pt-4">
            <RuntimeManifestTab run={run} />
          </TabsContent>
          <TabsContent value="validation" className="pt-4">
            <ValidationTab sessionId={sessionId} runId={run.run_id} />
          </TabsContent>
          <TabsContent value="raw" className="pt-4">
            <RawJsonTab run={run} />
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}
