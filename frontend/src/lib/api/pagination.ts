/**
 * Read a cursor-paginated, monotonically sequenced collection without
 * silently treating a first page as a complete history. Both the message and
 * durable-outbox endpoints use this shape. A non-advancing full page is a
 * server/proxy contract failure: continuing would spin forever and
 * presenting a partial transcript as complete would be worse than surfacing
 * the load error.
 */
export async function collectSequencedPages<T extends { sequence: number }>(
  readPage: (afterSequence: number, limit: number) => Promise<readonly T[]>,
  options: { pageSize: number; throughSequence?: number },
): Promise<T[]> {
  const { pageSize, throughSequence } = options;
  const items: T[] = [];
  let cursor = 0;

  while (throughSequence === undefined || cursor < throughSequence) {
    const page = await readPage(cursor, pageSize);
    // A later event may land while a bootstrap is paging. The caller's
    // session snapshot is the boundary; newer rows belong to the already
    // opened SSE stream, not this historical projection.
    const bounded = throughSequence === undefined
      ? page
      : page.filter((item) => item.sequence <= throughSequence);
    const last = bounded.at(-1);

    if (last && last.sequence <= cursor) {
      throw new Error('The paginated API returned a non-advancing sequence.');
    }
    if (bounded.length > 0) {
      items.push(...bounded);
      cursor = last!.sequence;
    }

    if (throughSequence !== undefined) {
      if (cursor >= throughSequence) return items;
      if (bounded.length === 0) {
        throw new Error('The paginated API could not reach the requested snapshot sequence.');
      }
      continue;
    }

    if (page.length < pageSize) return items;
    if (bounded.length === 0) {
      throw new Error('The paginated API returned an empty full page.');
    }
  }

  return items;
}
