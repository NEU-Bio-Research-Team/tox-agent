import { useMemo } from 'react';
import type { Message, RunProjection } from '../../lib/api/types';
import { isClarificationContent, isStructureRecognitionContent } from '../../lib/api/types';
import type { RecoveryBanner as RecoveryBannerData, ToolCallLive } from '../../hooks/useSessionEvents';
import { MessageBubble } from './MessageBubble';
import { ClarificationCard } from './ClarificationCard';
import { StructureRecognitionCard } from './StructureRecognitionCard';
import { AnswerBlock } from './AnswerBlock';
import { RunBlock } from './RunBlock';
import { AnalysisSystemCard } from './AnalysisSystemCard';
import { SystemEventCard } from './SystemEventCard';
import { RecoveryBanner } from './RecoveryBanner';
import type { PendingUserSend } from '../../lib/store/pendingSends';

// 'structure_recognition' lands here too: a successful run hands its SMILES
// straight to CreateAnalysis under the *same* run_id (recognize_structure.py),
// so it renders exactly like a typed-SMILES analysis — molecule card, hERG/
// Tox21 sections, the lot. A run that never reached recognition (capability
// unavailable, no structure found) falls through to the generic RunBlock
// below since no Analysis exists for it to link to.
const ANALYSIS_INTENTS = new Set(['analysis', 'analysis_batch', 'structure_recognition']);
const SILENT_INTENTS = new Set(['clarification_required', 'out_of_scope']);

export function Transcript({
  sessionId,
  messages,
  pendingSends,
  runs,
  liveToolCalls,
  recoveryBanners,
  analysisIdByRun,
  activeAnalysisId,
  onClarificationAction,
  onUseRecognizedSmiles,
}: {
  sessionId: string;
  messages: Message[];
  pendingSends: PendingUserSend[];
  runs: RunProjection[];
  liveToolCalls: Record<string, ToolCallLive[]>;
  recoveryBanners: RecoveryBannerData[];
  /** run_id -> analysis_id seen live this session (useSessionEvents). */
  analysisIdByRun: Record<string, string>;
  /** The session's current `active_analysis.analysis_id`, if any — used as a
   * fallback link target for the single most recently completed analysis
   * run when it wasn't observed live (e.g. right after a reload). */
  activeAnalysisId?: string | null;
  onClarificationAction: (action: string) => void;
  /** A recognition result is a prefill only; it never auto-submits analysis. */
  onUseRecognizedSmiles: (smiles: string) => void;
}) {
  // A trigger message can own more than one run — a failed run and its
  // recovery share the same trigger_message_id — so this keeps every run in
  // the group instead of the last one written overwriting the others.
  const runsByTrigger = useMemo(() => {
    const map = new Map<string, RunProjection[]>();
    for (const run of runs) {
      if (SILENT_INTENTS.has(run.intent)) continue;
      map.set(run.trigger_message_id, [...(map.get(run.trigger_message_id) ?? []), run]);
    }
    for (const group of map.values()) {
      group.sort((a, b) => a.created_at.localeCompare(b.created_at));
    }
    return map;
  }, [runs]);

  const latestCompletedAnalysisRunId = useMemo(() => {
    const completed = runs.filter(
      (run) =>
        ANALYSIS_INTENTS.has(run.intent) &&
        run.status === 'completed' &&
        // Unlike analysis/analysis_batch, a completed structure_recognition
        // run does not guarantee an analysis exists (capability unavailable,
        // no structure found) — this reload-time fallback must not attribute
        // whatever the *current* active_analysis happens to be to a run that
        // never actually produced one, so it only trusts a run whose
        // analysis.created event this client actually observed live.
        (run.intent !== 'structure_recognition' || run.run_id in analysisIdByRun),
    );
    if (completed.length === 0) return null;
    return completed.reduce((latest, run) => (run.created_at > latest.created_at ? run : latest)).run_id;
  }, [runs, analysisIdByRun]);

  const bannersByOriginalRun = useMemo(() => {
    const map = new Map<string, RecoveryBannerData[]>();
    for (const banner of recoveryBanners) {
      map.set(banner.originalRunId, [...(map.get(banner.originalRunId) ?? []), banner]);
    }
    return map;
  }, [recoveryBanners]);

  if (messages.length === 0 && pendingSends.length === 0) {
    return (
      <div className="flex h-full min-h-[240px] items-center justify-center text-sm" style={{ color: 'var(--text-faint)' }}>
        Chưa có tin nhắn nào. Nhập SMILES hoặc đặt câu hỏi để bắt đầu.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {messages.map((message) => {
        const runGroup = message.role === 'user' ? runsByTrigger.get(message.message_id) : undefined;
        return (
          <div key={message.message_id}>
            {message.role === 'user' && <UserMessageContent message={message} />}

            {message.role === 'assistant' && renderAssistant(message, sessionId, onClarificationAction, onUseRecognizedSmiles)}

            {message.role === 'system_event' && <SystemEventCard message={message} />}

            {runGroup?.map((run) => {
              const resolvedAnalysisId =
                analysisIdByRun[run.run_id] ??
                (run.run_id === latestCompletedAnalysisRunId ? (activeAnalysisId ?? undefined) : undefined);
              // A structure_recognition run can complete with no analysis to
              // show for it — the assistant's ClarificationCard already
              // explains why, so claiming "Đã tạo phân tích" here would be
              // false. analysis/analysis_batch have no such case: completed
              // always means an analysis exists.
              const showAsAnalysis =
                ANALYSIS_INTENTS.has(run.intent) &&
                (run.intent !== 'structure_recognition' ||
                  run.status !== 'completed' ||
                  resolvedAnalysisId !== undefined);
              return (
                <div key={run.run_id}>
                  {showAsAnalysis ? (
                    <AnalysisSystemCard sessionId={sessionId} run={run} analysisId={resolvedAnalysisId} />
                  ) : (
                    <RunBlock sessionId={sessionId} run={run} liveToolCalls={liveToolCalls[run.run_id] ?? []} />
                  )}
                  {(bannersByOriginalRun.get(run.run_id) ?? []).map((banner) => (
                    <RecoveryBanner key={banner.recoveryRunId} banner={banner} />
                  ))}
                </div>
              );
            })}
          </div>
        );
      })}
      {pendingSends.map((pending) => (
        <PendingUserMessage key={pending.clientMessageId} pending={pending} />
      ))}
    </div>
  );
}

/** Deliberately a user-only placeholder. A client never invents a run,
 * analysis, answer, or event sequence while waiting for a POST/retry to be
 * reflected in the durable transcript. */
function PendingUserMessage({ pending }: { pending: PendingUserSend }) {
  return (
    <MessageBubble role="user">
      <div className="space-y-1.5" aria-live="polite">
        {pending.text && <p>{pending.text}</p>}
        {pending.smiles && (
          <code className="block break-all rounded bg-black/10 px-2 py-1 font-mono text-xs">{pending.smiles}</code>
        )}
        {pending.hasImage && <p className="text-xs opacity-90">Đã đính kèm 1 ảnh cấu trúc</p>}
        <p className="text-xs opacity-80">Đang gửi…</p>
      </div>
    </MessageBubble>
  );
}

/** A user message can carry a `text` part, an `analysis_ref` part (smiles or
 * batch_smiles), or both together — see submit_message.py `_user_parts`. A
 * SMILES-only "analyze" submission has no text part at all, so rendering
 * only the text part (as an earlier version of this component did) leaves
 * that message bubble empty. */
function UserMessageContent({ message }: { message: Message }) {
  const textPart = message.parts.find((p) => p.type === 'text');
  const analysisRefPart = message.parts.find((p) => p.type === 'analysis_ref');
  const imageRefPart = message.parts.find((p) => p.type === 'image_ref');
  const text = textPart?.content.text as string | undefined;
  const smiles = analysisRefPart?.content.smiles as string | undefined;
  const batchSmiles = analysisRefPart?.content.batch_smiles as string[] | undefined;
  const imageSizeBytes = imageRefPart?.content.size_bytes as number | undefined;

  return (
    <MessageBubble role="user">
      <div className="space-y-1.5">
        {text && <p>{text}</p>}
        {smiles && (
          <code className="block break-all rounded bg-black/10 px-2 py-1 font-mono text-xs">{smiles}</code>
        )}
        {batchSmiles && (
          <p className="text-xs opacity-90">{batchSmiles.length} SMILES gửi hàng loạt</p>
        )}
        {imageRefPart && (
          <p className="text-xs opacity-90">
            Đã gửi 1 ảnh cấu trúc
            {imageSizeBytes ? ` (${Math.max(1, Math.round(imageSizeBytes / 1024))} KB)` : ''}
          </p>
        )}
      </div>
    </MessageBubble>
  );
}

function renderAssistant(
  message: Message,
  sessionId: string,
  onClarificationAction: (action: string) => void,
  onUseRecognizedSmiles: (smiles: string) => void,
) {
  const textPart = message.parts.find((p) => p.type === 'text');
  const answerRefPart = message.parts.find((p) => p.type === 'answer_ref');

  if (textPart && isStructureRecognitionContent(textPart.content)) {
    return <StructureRecognitionCard content={textPart.content} onUseSmiles={onUseRecognizedSmiles} />;
  }

  if (textPart && isClarificationContent(textPart.content)) {
    return <ClarificationCard content={textPart.content} onAction={onClarificationAction} />;
  }

  if (answerRefPart) {
    return (
      <MessageBubble role="assistant">
        <AnswerBlock sessionId={sessionId} answerId={answerRefPart.content.answer_id as string} />
      </MessageBubble>
    );
  }

  if (textPart) {
    return <MessageBubble role="assistant">{String(textPart.content.text ?? '')}</MessageBubble>;
  }

  return null;
}
