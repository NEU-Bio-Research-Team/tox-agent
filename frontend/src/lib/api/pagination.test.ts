// @vitest-environment node
import { describe, expect, it, vi } from 'vitest';
import { collectSequencedPages } from './pagination';

describe('collectSequencedPages', () => {
  it('reads every page to a fixed bootstrap snapshot and leaves later rows for SSE', async () => {
    const readPage = vi
      .fn()
      .mockResolvedValueOnce([{ sequence: 1 }, { sequence: 2 }])
      .mockResolvedValueOnce([{ sequence: 3 }, { sequence: 4 }]);

    await expect(collectSequencedPages(readPage, { pageSize: 2, throughSequence: 3 })).resolves.toEqual([
      { sequence: 1 },
      { sequence: 2 },
      { sequence: 3 },
    ]);
    expect(readPage).toHaveBeenNthCalledWith(1, 0, 2);
    expect(readPage).toHaveBeenNthCalledWith(2, 2, 2);
  });

  it('reads messages until the first short page', async () => {
    const readPage = vi
      .fn()
      .mockResolvedValueOnce([{ sequence: 1 }, { sequence: 2 }])
      .mockResolvedValueOnce([{ sequence: 3 }]);

    await expect(collectSequencedPages(readPage, { pageSize: 2 })).resolves.toEqual([
      { sequence: 1 },
      { sequence: 2 },
      { sequence: 3 },
    ]);
  });

  it('fails closed instead of looping on a non-advancing full page', async () => {
    const readPage = vi.fn().mockResolvedValue([{ sequence: 1 }, { sequence: 1 }]);

    await expect(collectSequencedPages(readPage, { pageSize: 2 })).rejects.toThrow('non-advancing');
  });
});
