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
}: {
  agentEnabled: boolean | null;
  readyProfileCount: number;
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
}: {
  /** `undefined`: the user never chose a default. `null`: they chose the server runtime. */
  defaultProfileId: string | null | undefined;
  currentProfileId: string | null;
  readyProfileIds: readonly string[];
}): string | null {
  const wanted = defaultProfileId === undefined ? currentProfileId : defaultProfileId;
  return wanted && readyProfileIds.includes(wanted) ? wanted : null;
}
