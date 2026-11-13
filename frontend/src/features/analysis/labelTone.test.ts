import { describe, expect, it } from 'vitest';
import { labelTone } from './labelTone';

describe('labelTone', () => {
  it('draws a positive class as a warning, never as danger', () => {
    for (const label of ['blocker', 'active', 'toxic', 'clinically_toxic']) {
      expect(labelTone(label)).toBe('warning');
    }
  });

  it('draws a negative class as neutral', () => {
    for (const label of ['non_blocker', 'non-blocker', 'inactive', 'not_toxic', 'negative']) {
      expect(labelTone(label)).toBe('neutral');
    }
  });

  it('never returns danger for any label', () => {
    for (const label of ['blocker', 'non_blocker', 'unknown', '']) {
      expect(labelTone(label)).not.toBe('danger');
    }
  });
});
