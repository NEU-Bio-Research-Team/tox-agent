import { describe, expect, it } from 'vitest';
import { linkifyClaims, CLAIM_LINK_SCHEME } from './answerMarkdown';
import type { Claim } from './api/types';

function claim(overrides: Partial<Claim> & Pick<Claim, 'claim_id' | 'rendered_value'>): Claim {
  return {
    kind: 'numeric',
    text: '',
    transform: 'identity',
    citation_ids: [],
    ...overrides,
  };
}

describe('linkifyClaims', () => {
  it('links the exact rendered_value substring, once', () => {
    const c = claim({ claim_id: 'clm_1', rendered_value: '0.731' });
    const result = linkifyClaims('The probability is 0.731.', [c]);
    expect(result).toBe(`The probability is [0.731](${CLAIM_LINK_SCHEME}clm_1).`);
  });

  it('does not link a value that never appears in the text', () => {
    const c = claim({ claim_id: 'clm_1', rendered_value: '0.731' });
    const result = linkifyClaims('No numbers here.', [c]);
    expect(result).toBe('No numbers here.');
    expect(result).not.toContain(CLAIM_LINK_SCHEME);
  });

  it('links only the first occurrence of a repeated value', () => {
    const c = claim({ claim_id: 'clm_1', rendered_value: '0.731' });
    const result = linkifyClaims('0.731 and again 0.731.', [c]);
    expect(result).toBe(`[0.731](${CLAIM_LINK_SCHEME}clm_1) and again 0.731.`);
  });

  it('prefers the longer of two overlapping candidate values so a short one does not shadow it', () => {
    // "3.15%" must not be matched as "3.15" + a stray "%".
    const short = claim({ claim_id: 'clm_short', rendered_value: '3.15' });
    const long = claim({ claim_id: 'clm_long', rendered_value: '3.15%' });
    const result = linkifyClaims('The value is 3.15%.', [short, long]);
    expect(result).toBe(`The value is [3.15%](${CLAIM_LINK_SCHEME}clm_long).`);
  });

  it('unwraps a backtick-wrapped value and still produces a chip link, not literal brackets', () => {
    const c = claim({ claim_id: 'clm_1', rendered_value: 'non_blocker' });
    const result = linkifyClaims('The label is **`non_blocker`**.', [c]);
    expect(result).toBe(`The label is **[non_blocker](${CLAIM_LINK_SCHEME}clm_1)**.`);
  });

  it('links each distinct claim only once each, even if both values appear multiple times', () => {
    const a = claim({ claim_id: 'clm_a', rendered_value: '0.1' });
    const b = claim({ claim_id: 'clm_b', rendered_value: '0.2' });
    const result = linkifyClaims('0.1 0.2 0.1 0.2', [a, b]);
    expect(result).toBe(
      `[0.1](${CLAIM_LINK_SCHEME}clm_a) [0.2](${CLAIM_LINK_SCHEME}clm_b) 0.1 0.2`,
    );
  });

  it('ignores a claim with no rendered_value', () => {
    const c = claim({ claim_id: 'clm_1', rendered_value: undefined as unknown as string });
    const result = linkifyClaims('Some prose with no number.', [c]);
    expect(result).toBe('Some prose with no number.');
  });

  it('ignores a rendered_value shorter than 2 characters, to avoid linking every stray digit', () => {
    const c = claim({ claim_id: 'clm_1', rendered_value: '5' });
    const result = linkifyClaims('There are 5 assays.', [c]);
    expect(result).toBe('There are 5 assays.');
  });
});
