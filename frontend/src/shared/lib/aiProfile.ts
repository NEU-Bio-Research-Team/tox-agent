export interface AgentProfileChoice {
  /** What the null (no profile saved) option is called. */
  defaultOptionLabel: string;
  /** Shown under the control when something needs configuring. */
  notice: string | null;
  /** True when picking a profile cannot help. */
  disabled: boolean;
}

/**
 * What to call "no profile saved", given what the deployment actually has.
 *
 * I06: this was unconditionally "Runtime mặc định", which reads as "a usable
 * default is in force". It said that in a predictor-only deployment with no
 * runtime at all, and in an agent deployment with no profiles configured —
 * two states where nothing would run. Separated out from the component
 * because the wording is the fix; the control around it is not.
 *
 * `agentEnabled` is `null` while /health/ready is still in flight: assume
 * nothing, and say nothing, rather than flashing a claim either way.
 */
export function describeAgentProfileChoice({
  agentEnabled,
  readyProfileCount,
  userAiRequired = false,
}: {
  agentEnabled: boolean | null;
  readyProfileCount: number;
  /** The server has no AI of its own (`runtime.ai_source === 'user_connections'`). */
  userAiRequired?: boolean;
}): AgentProfileChoice {
  if (agentEnabled === false) {
    return {
      defaultOptionLabel: 'Không có agent runtime',
      notice:
        'Bản triển khai này chỉ chạy dự đoán. Câu hỏi về báo cáo, attribution và tra cứu ' +
        'tài liệu không khả dụng ở đây.',
      disabled: true,
    };
  }
  if (userAiRequired) {
    // There is no server default to fall back to: offering one by name would
    // be I06 again, a label promising a runtime that is not there.
    return readyProfileCount === 0
      ? {
          defaultOptionLabel: 'Chưa có AI — hãy thêm kết nối',
          notice:
            'Server này không có AI riêng: mỗi người tự kết nối AI của mình (API key hoặc ' +
            'ChatGPT Plus/Pro) trong Cài đặt, rồi chọn model ngay trong khung chat.',
          disabled: false,
        }
      : {
          defaultOptionLabel: 'Kết nối đầu tiên đã sẵn sàng',
          notice: null,
          disabled: false,
        };
  }
  if (readyProfileCount === 0) {
    return {
      defaultOptionLabel: 'Chưa cấu hình AI',
      notice: 'Chưa có profile AI nào sẵn sàng.',
      disabled: false,
    };
  }
  return {
    defaultOptionLabel: 'Runtime mặc định của server',
    notice: null,
    disabled: false,
  };
}

/** A connection (the credential) and which of its account's models to run. */
export interface AiChoice {
  connectionId: string;
  /** `null`: the connection's own model, the one its test ran on. */
  modelId: string | null;
}

/**
 * What a session should run on when the workbench opens it.
 *
 * The model is picked in the chat, per session. A session keeps its own pick
 * while that connection is still ready; a session with none (or a deleted /
 * failed one) starts on the user's last pick, and — on a deployment with no
 * AI of its own, where `null` would fail every run — on the first ready
 * connection. `null` is the deployment's own runtime.
 *
 * A failed or deleted connection must never stay pinned: every run would fail
 * with "profile unavailable" until the user noticed.
 */
export function choiceForSession({
  saved,
  lastChoice,
  readyConnectionIds,
  userAiRequired = false,
}: {
  saved: AiChoice | null;
  lastChoice: AiChoice | null;
  readyConnectionIds: readonly string[];
  userAiRequired?: boolean;
}): AiChoice | null {
  if (saved && readyConnectionIds.includes(saved.connectionId)) return saved;
  if (lastChoice && readyConnectionIds.includes(lastChoice.connectionId)) return lastChoice;
  if (userAiRequired && readyConnectionIds[0]) return { connectionId: readyConnectionIds[0], modelId: null };
  return null;
}

export function sameChoice(a: AiChoice | null, b: AiChoice | null): boolean {
  if (a === null || b === null) return a === b;
  return a.connectionId === b.connectionId && (a.modelId ?? null) === (b.modelId ?? null);
}
