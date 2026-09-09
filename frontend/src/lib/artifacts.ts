import type { SessionProjection } from './api/types';
import { RUN_STATUS_LABEL_VI, INTENT_LABEL_VI } from './labels';

export interface ArtifactPickerOption {
  value: string;
  label: string;
}

/**
 * Plan section 8.2.1/10.2: MVP has no dedicated artifact-catalog API, so the
 * picker is built from what the session projection already carries —
 * `active_analysis` and up to 10 `recent_runs` — rather than replaying the
 * full event history. Answers/observations stay reachable through in-chat
 * links instead of appearing here; §10.2 tracks the backend catalog gap
 * this would need to close properly.
 */
export function buildArtifactPickerOptions(session: SessionProjection): ArtifactPickerOption[] {
  const options: ArtifactPickerOption[] = [];

  if (session.active_analysis) {
    options.push({
      value: `analysis:${session.active_analysis.analysis_id}`,
      label: `Predictor · ${session.active_analysis.analysis_id}`,
    });
  }

  const runs = [...session.recent_runs];
  if (session.active_run && !runs.some((r) => r.run_id === session.active_run!.run_id)) {
    runs.push(session.active_run);
  }
  runs.sort((a, b) => b.created_at.localeCompare(a.created_at));

  for (const run of runs) {
    options.push({
      value: `run:${run.run_id}`,
      label: `Run ${INTENT_LABEL_VI[run.intent] ?? run.intent} · ${RUN_STATUS_LABEL_VI[run.status]} · ${run.run_id.slice(0, 12)}…`,
    });
  }

  return options;
}
