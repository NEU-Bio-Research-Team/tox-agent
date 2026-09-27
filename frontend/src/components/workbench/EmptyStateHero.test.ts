import { describe, expect, it } from 'vitest';

import { EMPTY_STATE_EXAMPLES } from './EmptyStateHero';
import { suggestMolecule } from '../../lib/smiles';

describe('empty-state examples', () => {
  it.each(EMPTY_STATE_EXAMPLES)('%s names exactly one molecule and keeps its question', (example) => {
    // Two candidates would block the send as ambiguous; a bare molecule would
    // drop the question. Either would make the example a bad first impression.
    const suggestion = suggestMolecule(example);
    expect(suggestion.candidates).toHaveLength(1);
    expect(suggestion.isBareMolecule).toBe(false);
  });
});
