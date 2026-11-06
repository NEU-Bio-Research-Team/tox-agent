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
            'ChatGPT Plus/Pro) ở bên dưới rồi bấm Test.',
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

/**
 * The profile a session's next run should be pinned to.
 *
 * The Settings default when the user chose one; otherwise whatever the
 * session was already pinned to, so a profile picked in the old per-session
 * popover survives until a default is chosen. Either way only while that
 * profile is still ready: a deleted or failed profile left pinned makes every
 * run fail with "profile unavailable", and with no per-session picker the user
 * would have no way to unpin it. `null` is the deployment's own runtime.
 */
export function profileForSession({
  defaultProfileId,
  currentProfileId,
  readyProfileIds,
  userAiRequired = false,
}: {
  /** `undefined`: the user never chose a default. `null`: they chose the server runtime. */
  defaultProfileId: string | null | undefined;
  currentProfileId: string | null;
  readyProfileIds: readonly string[];
  /** The server has no runtime of its own, so `null` would fail every run. */
  userAiRequired?: boolean;
}): string | null {
  const wanted = defaultProfileId === undefined ? currentProfileId : defaultProfileId;
  if (wanted && readyProfileIds.includes(wanted)) return wanted;
  // With no server runtime, any ready connection beats one the run cannot use.
  return userAiRequired ? (readyProfileIds[0] ?? null) : null;
}
