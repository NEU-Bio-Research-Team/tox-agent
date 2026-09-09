import { useInfiniteQuery } from '@tanstack/react-query';
import { listSessions } from '../lib/api/endpoints';
import type { SessionListRow } from '../lib/api/types';

const PAGE_SIZE = 25;

/**
 * Shared across AppSidebar and SessionsPage so both read the same cache
 * entry (`['sessions']`) instead of the sidebar's list drifting out of sync
 * with the expanded history view — plan section 8.1: "/sessions là view lịch
 * sử mở rộng dùng cùng cache".
 */
export function useSessionsList() {
  const query = useInfiniteQuery({
    queryKey: ['sessions'],
    queryFn: ({ pageParam }) => listSessions({ limit: PAGE_SIZE, offset: pageParam }),
    initialPageParam: 0,
    getNextPageParam: (lastPage) => lastPage.next_offset ?? undefined,
  });

  const rows: SessionListRow[] = query.data?.pages.flatMap((page) => page.sessions) ?? [];

  return {
    rows,
    isLoading: query.isLoading,
    isError: query.isError,
    hasNextPage: query.hasNextPage,
    isFetchingNextPage: query.isFetchingNextPage,
    fetchNextPage: query.fetchNextPage,
    refetch: query.refetch,
  };
}
