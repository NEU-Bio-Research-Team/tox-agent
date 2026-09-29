import { useCallback, useEffect, useRef } from 'react';
import { getSessionSettings, listModelConnections, updateSessionSettings } from '../../../shared/api/endpoints';
import { profileForSession } from '../../../shared/lib/aiProfile';
import { getDefaultAiProfileId } from '../../../shared/lib/preferences';

/**
 * Pins the session to the AI profile chosen in Settings.
 *
 * The workbench header used to carry a per-session profile picker. Settings is
 * now the one place a profile is chosen, and the server reads the profile from
 * session settings when a run is created, so the choice has to be written
 * there before the first send. Returns a function the send path awaits; a
 * failed sync never blocks a send — the run reports an unusable profile itself.
 */
export function useSessionProfileSync(sessionId: string): () => Promise<void> {
  const pending = useRef<Promise<void>>(Promise.resolve());

  useEffect(() => {
    pending.current = (async () => {
      try {
        const [saved, profiles] = await Promise.all([getSessionSettings(sessionId), listModelConnections()]);
        const wanted = profileForSession({
          defaultProfileId: getDefaultAiProfileId(),
          currentProfileId: saved.ai_profile_id,
          readyProfileIds: profiles.connections.filter((item) => item.status === 'ready').map((item) => item.connection_id),
        });
        if (wanted !== saved.ai_profile_id) {
          await updateSessionSettings(sessionId, { ...saved, ai_profile_id: wanted });
        }
      } catch {
        // best effort — see above
      }
    })();
  }, [sessionId]);

  return useCallback(() => pending.current, []);
}
