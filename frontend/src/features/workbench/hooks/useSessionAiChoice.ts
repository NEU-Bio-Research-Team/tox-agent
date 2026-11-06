import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  getHealthReady,
  getSessionSettings,
  listModelConnections,
  listSupportedProviders,
  updateSessionSettings,
} from '../../../shared/api/endpoints';
import type { ModelConnection } from '../../../shared/api/types';
import { choiceForSession, sameChoice, type AiChoice } from '../../../shared/lib/aiProfile';
import { getLastAiChoice, setLastAiChoice } from '../../../shared/lib/preferences';

export interface SessionAiChoice {
  /** What the next run uses; `null` is the deployment's own runtime. */
  choice: AiChoice | null;
  /** Connections that passed their test — the only ones a run can use. */
  readyConnections: ModelConnection[];
  /** provider_id → its display name, for grouping in the picker. */
  providerNames: Record<string, string>;
  loading: boolean;
  /** The server has no AI of its own: a session without a connection cannot run. */
  userAiRequired: boolean;
  choose: (next: AiChoice) => Promise<void>;
  /** Awaited by the send path, so a pick is saved before the run reads it. */
  settled: () => Promise<void>;
}

export const MODEL_CONNECTIONS_KEY = ['model-connections'] as const;

/**
 * The session's AI: which connection, and which of its account's models.
 *
 * Picked in the chat (the composer's model picker), never in Settings —
 * Settings only adds credentials. The server reads the pick from session
 * settings when a run is created, so a pick is written before the next send;
 * `settled` lets the send path wait for that. A failed write never blocks a
 * send — the run itself reports an unusable profile.
 */
export function useSessionAiChoice(sessionId: string): SessionAiChoice {
  const queryClient = useQueryClient();
  const pending = useRef<Promise<void>>(Promise.resolve());
  const [choice, setChoice] = useState<AiChoice | null>(null);

  const connectionsQuery = useQuery({
    queryKey: MODEL_CONNECTIONS_KEY,
    queryFn: () => listModelConnections(),
    staleTime: 30_000,
  });
  const providersQuery = useQuery({
    queryKey: ['model-providers'],
    queryFn: () => listSupportedProviders(),
    staleTime: 10 * 60_000,
  });
  const healthQuery = useQuery({
    queryKey: ['health-ready'],
    queryFn: () => getHealthReady(),
    staleTime: 5 * 60_000,
  });
  const settingsQuery = useQuery({
    queryKey: ['session-settings', sessionId],
    queryFn: () => getSessionSettings(sessionId),
  });

  const readyConnections = useMemo(
    () => (connectionsQuery.data?.connections ?? []).filter((item) => item.status === 'ready'),
    [connectionsQuery.data],
  );
  const providerNames = useMemo(
    () => Object.fromEntries((providersQuery.data?.providers ?? []).map((item) => [item.provider_id, item.display_name])),
    [providersQuery.data],
  );
  const userAiRequired = healthQuery.data?.runtime?.ai_source === 'user_connections';

  const save = useCallback(
    (next: AiChoice | null) => {
      const saved = settingsQuery.data;
      pending.current = (async () => {
        try {
          const updated = await updateSessionSettings(sessionId, {
            predictor_bindings: saved?.predictor_bindings ?? {},
            ai_profile_id: next?.connectionId ?? null,
            ai_model_id: next?.modelId ?? null,
          });
          queryClient.setQueryData(['session-settings', sessionId], updated);
        } catch {
          // best effort — see above
        }
      })();
      return pending.current;
    },
    [queryClient, sessionId, settingsQuery.data],
  );

  // Once everything is known: keep the session's own pick if it still works,
  // otherwise start it on the user's last pick (aiProfile.choiceForSession).
  const initialised = useRef(false);
  useEffect(() => {
    if (initialised.current || !settingsQuery.data || !connectionsQuery.data || healthQuery.isLoading) return;
    initialised.current = true;
    const saved = settingsQuery.data.ai_profile_id
      ? { connectionId: settingsQuery.data.ai_profile_id, modelId: settingsQuery.data.ai_model_id ?? null }
      : null;
    const wanted = choiceForSession({
      saved,
      lastChoice: getLastAiChoice(),
      readyConnectionIds: readyConnections.map((item) => item.connection_id),
      userAiRequired,
    });
    setChoice(wanted);
    if (!sameChoice(wanted, saved)) void save(wanted);
  }, [settingsQuery.data, connectionsQuery.data, healthQuery.isLoading, readyConnections, userAiRequired, save]);

  const choose = useCallback(
    (next: AiChoice) => {
      setChoice(next);
      setLastAiChoice(next);
      return save(next);
    },
    [save],
  );

  return {
    choice,
    readyConnections,
    providerNames,
    loading: connectionsQuery.isLoading || settingsQuery.isLoading,
    userAiRequired,
    choose,
    settled: useCallback(() => pending.current, []),
  };
}
