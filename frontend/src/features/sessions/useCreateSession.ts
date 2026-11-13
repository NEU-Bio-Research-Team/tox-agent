import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router';
import { createSession } from '../../shared/api/endpoints';

/** Create an empty session and open it. Shared by the rail, the Sessions page
 * and the command palette so they cannot drift apart. */
export function useCreateSession() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => createSession({ preferred_language: 'vi' }),
    onSuccess: (session) => {
      void queryClient.invalidateQueries({ queryKey: ['sessions'] });
      navigate(`/s/${session.session_id}`);
    },
  });
}
