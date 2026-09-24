import { useParams, useSearchParams } from 'react-router';

export type ArtifactKind = 'analysis' | 'run' | 'answer' | 'observation' | 'evidence';

export interface ArtifactSelection {
  kind: ArtifactKind;
  entityId: string;
  /** An exact claim field path when the artifact is opened as provenance. */
  fieldPath?: string;
}

export function artifactPath(sessionId: string, selection: ArtifactSelection | null): string {
  if (!selection) return `/s/${sessionId}`;
  const segment: Record<ArtifactKind, string> = {
    analysis: 'analyses',
    run: 'runs',
    answer: 'answers',
    observation: 'observations',
    evidence: 'evidence',
  };
  const path = `/s/${sessionId}/${segment[selection.kind]}/${selection.entityId}`;
  if (selection.kind !== 'observation' || !selection.fieldPath) return path;
  return `${path}?${new URLSearchParams({ field_path: selection.fieldPath }).toString()}`;
}

/**
 * Plan section 5.2/7.6: the URL is the primary source for artifact
 * selection. `/s/:sessionId/{runs,analyses,answers,observations}/:id` are
 * five distinct router entries that all render the same lazily-imported
 * `WorkbenchPage` component reference (see router.tsx) — React reconciles
 * them as the same element at the same tree position, so navigating between
 * them re-runs this hook's `useParams` read without remounting the page,
 * its transcript, composer, or SSE subscription.
 */
export function useArtifactSelectionFromUrl(): ArtifactSelection | null {
  const params = useParams<{
    runId?: string;
    analysisId?: string;
    answerId?: string;
    observationId?: string;
    evidenceId?: string;
  }>();
  const [searchParams] = useSearchParams();
  if (params.runId) return { kind: 'run', entityId: params.runId };
  if (params.analysisId) return { kind: 'analysis', entityId: params.analysisId };
  if (params.answerId) return { kind: 'answer', entityId: params.answerId };
  if (params.observationId) {
    const fieldPath = searchParams.get('field_path');
    return { kind: 'observation', entityId: params.observationId, ...(fieldPath ? { fieldPath } : {}) };
  }
  if (params.evidenceId) return { kind: 'evidence', entityId: params.evidenceId };
  return null;
}
