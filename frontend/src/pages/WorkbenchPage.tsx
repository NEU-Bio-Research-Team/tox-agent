import { lazy, Suspense, useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { ArrowDown, PanelRight } from 'lucide-react';
import { WorkspaceLayout } from '../components/shell/WorkspaceLayout';
import { WorkspaceHeader } from '../components/shell/WorkspaceHeader';
import { Button } from '../components/ui/button';
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from '../components/ui/sheet';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '../components/ui/resizable';
import { MessageComposer, type AnalysisContext, type SmilesPrefill } from '../components/workbench/MessageComposer';
import { EmptyStateHero } from '../components/workbench/EmptyStateHero';
import { Transcript } from '../components/transcript/Transcript';
import {
  getHealthReady,
  getSession,
  listAllMessages,
  listEventsThroughSnapshot,
  sendMessage,
} from '../lib/api/endpoints';
import { errorMessageVi } from '../lib/labels';
import { ApiError } from '../lib/api/types';
import { useSessionEvents } from '../hooks/useSessionEvents';
import { useArtifactSelectionFromUrl, artifactPath } from '../hooks/useArtifactSelection';
import { useBreakpoint } from '../hooks/useBreakpoint';
import { useStickToBottom } from '../hooks/useStickToBottom';
import { getLayoutPreferences, setLayoutPreferences, clampArtifactsWidth } from '../lib/preferences';
import {
  addPendingSend,
  confirmPendingSends,
  pendingSendFromInput,
  type PendingUserSend,
} from '../lib/store/pendingSends';
import type { SessionProjection } from '../lib/api/types';
import type { SendMessageInput } from '../lib/api/endpoints';

// Artifact viewers pull charts, inspectors, and analysis renderers. Keeping
// them behind the user-opened right panel preserves the transcript's first
// paint and keeps their route state in the same Workbench component.
const ArtifactsPanel = lazy(async () => {
  const module = await import('../components/artifacts/ArtifactsPanel');
  return { default: module.ArtifactsPanel };
});

export function WorkbenchPage() {
  // The route is always "/s/:sessionId" or one of its artifact sub-routes
  // (see router.tsx) — all five map to this same component reference, so
  // switching between them never remounts anything below WorkspaceLayout.
  const { sessionId = '' } = useParams<{ sessionId: string }>();

  const bootstrap = useQuery({
    queryKey: ['session', sessionId],
    queryFn: () => getSession(sessionId),
  });

  return (
    <WorkspaceLayout>
      {bootstrap.isLoading && (
        <div className="flex h-full items-center justify-center text-sm" style={{ color: 'var(--text-muted)' }}>
          Đang tải session…
        </div>
      )}
      {bootstrap.isError && (
        <div className="flex h-full items-center justify-center text-sm" style={{ color: 'var(--accent-red)' }}>
          {bootstrap.error instanceof ApiError
            ? errorMessageVi(bootstrap.error.code, bootstrap.error.message)
            : 'Không tải được session.'}
        </div>
      )}
      {bootstrap.data && <WorkbenchView key={sessionId} sessionId={sessionId} initial={bootstrap.data} />}
    </WorkspaceLayout>
  );
}

function WorkbenchView({ sessionId, initial }: { sessionId: string; initial: SessionProjection }) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const breakpoint = useBreakpoint();
  const selection = useArtifactSelectionFromUrl();

  // Undefined until first bumped — MessageComposer's effects fire on
  // "!== undefined", so starting these at a real number (e.g. 0) would fire
  // every one of them, including the two dialogs, on the very first mount.
  const [focusSmilesSignal, setFocusSmilesSignal] = useState<number>();
  const [openDrawSignal, setOpenDrawSignal] = useState<number>();
  const [openImageSignal, setOpenImageSignal] = useState<number>();
  const [smilesPrefill, setSmilesPrefill] = useState<SmilesPrefill>();
  const [analysisContext, setAnalysisContext] = useState<AnalysisContext | null>(null);
  // Section 8.2.1's UI state: none of this is a durable preference (7.6) —
  // it resets whenever the session itself changes (WorkbenchView remounts
  // via `key={sessionId}` above), which is exactly right: an artifact from
  // session A must never bleed into session B's panel.
  const [manuallyClosed, setManuallyClosed] = useState(false);
  const [emptyPanelOpen, setEmptyPanelOpen] = useState(false);
  const [seenArtifactSequence, setSeenArtifactSequence] = useState(0);
  const [artifactsWidthPct, setArtifactsWidthPct] = useState(() => getLayoutPreferences().artifactsWidthPct);

  const sessionQuery = useQuery({
    queryKey: ['session', sessionId],
    queryFn: () => getSession(sessionId),
    initialData: initial,
  });
  const session = sessionQuery.data ?? initial;

  const messagesQuery = useQuery({
    queryKey: ['messages', sessionId],
    queryFn: () => listAllMessages(sessionId),
  });
  const historyEventsQuery = useQuery({
    // `initial` is the GET session snapshot used to seed SSE. Keeping this
    // key fixed for the mount gives the history a finite boundary even when a
    // busy session receives more events while its older pages are loading.
    queryKey: ['session-event-history', sessionId, initial.latest_event_sequence],
    queryFn: () => listEventsThroughSnapshot(sessionId, initial.latest_event_sequence),
    staleTime: Infinity,
  });

  // A deployment fact (is TOXAGENT_OCR_URL configured?), not a per-session
  // one — long staleTime so this doesn't re-poll on every session switch.
  const capabilitiesQuery = useQuery({
    queryKey: ['health-ready'],
    queryFn: () => getHealthReady(),
    staleTime: 5 * 60_000,
  });
  const structureRecognitionAvailable = capabilitiesQuery.data?.capabilities?.structure_recognition ?? false;

  const { status, liveToolCalls, recoveryBanners, analysisIdByRun, latestArtifact } = useSessionEvents(
    sessionId,
    initial.latest_event_sequence,
    historyEventsQuery.data,
  );
  const [pendingSends, setPendingSends] = useState<PendingUserSend[]>([]);

  // A successful POST only means the client received a response. The pending
  // bubble disappears when the durable transcript itself contains the same
  // idempotency key, including the response-lost/retry case.
  useEffect(() => {
    if (!messagesQuery.data) return;
    setPendingSends((current) => confirmPendingSends(current, messagesQuery.data!.messages));
  }, [messagesQuery.data]);

  const hasUnseenArtifact = latestArtifact !== null && latestArtifact.sequence > seenArtifactSequence;

  // Section 8.2.1: the first artifact created in the current interaction may
  // auto-open once on desktop — but only while the user hasn't already
  // looked at something (selection === null) and hasn't explicitly closed
  // the panel. Anything after that just raises the badge above instead of
  // yanking the view away from whatever they're reading.
  useEffect(() => {
    if (!latestArtifact || latestArtifact.sequence <= seenArtifactSequence) return;
    if (breakpoint === 'desktop' && !manuallyClosed && selection === null) {
      navigate(artifactPath(sessionId, latestArtifact), { replace: true });
      setSeenArtifactSequence(latestArtifact.sequence);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- fires only on latestArtifact identity change
  }, [latestArtifact]);

  // Navigating to any artifact (via a chat link, the picker, or the effect
  // above) counts as "đã xem" — clears the badge and un-sticks manualClose
  // so a fresh badge can auto-open again next time.
  useEffect(() => {
    if (selection) {
      setManuallyClosed(false);
      setEmptyPanelOpen(false);
      if (latestArtifact) setSeenArtifactSequence((prev) => Math.max(prev, latestArtifact.sequence));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection?.kind, selection?.entityId]);

  const sendMutation = useMutation({
    mutationFn: (input: SendMessageInput) => sendMessage(sessionId, input),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
      void queryClient.invalidateQueries({ queryKey: ['messages', sessionId] });
    },
    onError: (error) => {
      const message = error instanceof ApiError ? errorMessageVi(error.code, error.message) : 'Gửi thất bại.';
      toast.error(message);
    },
  });

  const runs = [...session.recent_runs];
  if (session.active_run && !runs.some((r) => r.run_id === session.active_run!.run_id)) {
    runs.push(session.active_run);
  }
  const activeRunBusy = session.active_run !== null;

  const handleClarificationAction = (action: string) => {
    if (action === 'submit_smiles') setFocusSmilesSignal((n) => (n ?? 0) + 1);
  };

  const handleUseRecognizedSmiles = (smiles: string) => {
    setSmilesPrefill((current) => ({ smiles, signal: (current?.signal ?? 0) + 1 }));
  };

  const panelVisible = selection !== null || emptyPanelOpen;

  const closeArtifacts = () => {
    setManuallyClosed(true);
    setEmptyPanelOpen(false);
    if (selection) navigate(`/s/${sessionId}`);
  };

  const toggleArtifacts = () => {
    if (panelVisible) {
      closeArtifacts();
    } else {
      setManuallyClosed(false);
      setEmptyPanelOpen(true);
      if (latestArtifact) setSeenArtifactSequence(latestArtifact.sequence);
    }
  };

  const handleAskAboutAnalysis = (analysisId: string) => {
    const label =
      session.active_analysis?.analysis_id === analysisId
        ? session.active_analysis.canonical_smiles
        : `${analysisId.slice(0, 12)}…`;
    setAnalysisContext({ analysisId, label });
  };

  const { containerRef: transcriptRef, showJump, jumpToBottom } = useStickToBottom(
    (messagesQuery.data?.messages.length ?? 0) + pendingSends.length,
  );

  const artifactsButton = (
    <Button variant="outline" size="sm" className="relative gap-1.5" onClick={toggleArtifacts} aria-expanded={panelVisible} aria-controls="artifacts-panel">
      <PanelRight className="h-4 w-4" />
      Artifacts
      {hasUnseenArtifact && (
        <span
          className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full"
          style={{ backgroundColor: 'var(--accent-blue)' }}
          aria-label="Có kết quả mới"
        />
      )}
    </Button>
  );

  const chatColumn = (
    <div className="relative flex h-full min-w-0 flex-col">
      <WorkspaceHeader
        title={session.title ?? session.session_id}
        subtitle={session.session_id}
        status={status}
        actions={artifactsButton}
      />
      <div ref={transcriptRef} className="flex-1 overflow-y-auto px-4 py-4 md:px-6">
        <div className="mx-auto max-w-[800px]">
          {messagesQuery.isLoading && (
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Đang tải hội thoại…
            </p>
          )}
          {messagesQuery.isError && (
            <p className="text-sm" style={{ color: 'var(--accent-red)' }}>
              Không tải được lịch sử hội thoại.
            </p>
          )}
          {messagesQuery.data && messagesQuery.data.messages.length === 0 && pendingSends.length === 0 && (
            <EmptyStateHero
              onPickSmiles={() => setFocusSmilesSignal((n) => (n ?? 0) + 1)}
              onPickImage={() => setOpenImageSignal((n) => (n ?? 0) + 1)}
              onPickDraw={() => setOpenDrawSignal((n) => (n ?? 0) + 1)}
            />
          )}
          {messagesQuery.data && (messagesQuery.data.messages.length > 0 || pendingSends.length > 0) && (
            <Transcript
              sessionId={sessionId}
              messages={messagesQuery.data.messages}
              pendingSends={pendingSends}
              runs={runs}
              liveToolCalls={liveToolCalls}
              recoveryBanners={recoveryBanners}
              analysisIdByRun={analysisIdByRun}
              activeAnalysisId={session.active_analysis?.analysis_id ?? null}
              onClarificationAction={handleClarificationAction}
              onUseRecognizedSmiles={handleUseRecognizedSmiles}
            />
          )}
        </div>
      </div>
      {showJump && (
        <div className="pointer-events-none absolute inset-x-0 bottom-24 flex justify-center">
          <Button size="sm" className="pointer-events-auto gap-1.5 shadow-md" onClick={jumpToBottom}>
            <ArrowDown className="h-3.5 w-3.5" />
            Tin nhắn mới
          </Button>
        </div>
      )}
      <div className="border-t px-4 py-3 md:px-6" style={{ borderColor: 'var(--border)' }}>
        <div className="mx-auto max-w-[800px]">
          <MessageComposer
            sessionId={sessionId}
            hasActiveAnalysis={Boolean(session.active_analysis)}
            disabled={sendMutation.isPending || activeRunBusy}
            focusSmilesSignal={focusSmilesSignal}
            openDrawSignal={openDrawSignal}
            openImageSignal={openImageSignal}
            smilesPrefill={smilesPrefill}
            structureRecognitionAvailable={structureRecognitionAvailable}
            analysisContext={analysisContext}
            onClearAnalysisContext={() => setAnalysisContext(null)}
            onSend={async (input) => {
              setPendingSends((current) => addPendingSend(current, pendingSendFromInput(input)));
              try {
                await sendMutation.mutateAsync(input);
                setAnalysisContext(null);
                return true;
              } catch {
                // The mutation's own onError already surfaced a toast; the
                // composer just needs to know not to clear the draft.
                return false;
              }
            }}
          />
        </div>
      </div>
    </div>
  );

  const artifactsContent = panelVisible ? (
    <Suspense
      fallback={
        <div className="flex h-full items-center justify-center text-sm" style={{ color: 'var(--text-muted)' }}>
          Đang tải artifacts…
        </div>
      }
    >
      <ArtifactsPanel
        sessionId={sessionId}
        session={session}
        selection={selection}
        onClose={closeArtifacts}
        onAskAboutAnalysis={handleAskAboutAnalysis}
      />
    </Suspense>
  ) : null;

  if (breakpoint === 'desktop') {
    return (
      <ResizablePanelGroup direction="horizontal" className="h-full">
        <ResizablePanel id="chat-panel" order={1} minSize={35} defaultSize={panelVisible ? 100 - artifactsWidthPct : 100}>
          {chatColumn}
        </ResizablePanel>
        {panelVisible && (
          <>
            <ResizableHandle withHandle />
            <ResizablePanel
              id="artifacts-panel"
              order={2}
              minSize={18}
              maxSize={40}
              defaultSize={artifactsWidthPct}
              onResize={(size) => {
                const clamped = clampArtifactsWidth(size);
                setArtifactsWidthPct(clamped);
                setLayoutPreferences({ artifactsWidthPct: clamped });
              }}
            >
              {artifactsContent}
            </ResizablePanel>
          </>
        )}
      </ResizablePanelGroup>
    );
  }

  return (
    <>
      {chatColumn}
      <Sheet open={panelVisible} onOpenChange={(open) => !open && closeArtifacts()}>
        <SheetContent
          side="right"
          className={
            breakpoint === 'tablet'
              ? 'w-full gap-0 p-0 [&>button]:hidden sm:max-w-[480px]'
              : 'w-full gap-0 p-0 [&>button]:hidden'
          }
        >
          <SheetHeader className="sr-only">
            <SheetTitle>Artifacts</SheetTitle>
            <SheetDescription>Kết quả predictor, tiến trình run và kiểm chứng đáp án.</SheetDescription>
          </SheetHeader>
          {artifactsContent}
        </SheetContent>
      </Sheet>
    </>
  );
}
