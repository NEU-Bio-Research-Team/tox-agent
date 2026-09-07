import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // The event stream is the push signal; a background refetch on top of
      // that would just be double bookkeeping for state that never goes stale
      // between events.
      staleTime: 30_000,
      refetchOnWindowFocus: true,
      retry: 1,
    },
  },
});
