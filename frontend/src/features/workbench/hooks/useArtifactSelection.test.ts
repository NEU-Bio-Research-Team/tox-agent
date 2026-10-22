import { describe, expect, it } from 'vitest';
import { artifactPath } from './useArtifactSelection';

describe('artifactPath', () => {
  it('keeps a claim field path in an observation deep link', () => {
    expect(artifactPath('ses_1', {
      kind: 'observation',
      entityId: 'obs_1',
      fieldPath: 'predictions.herg.probability_blocker',
    })).toBe('/s/ses_1/observations/obs_1?field_path=predictions.herg.probability_blocker');
  });

  it('does not attach a field path to other artifact kinds', () => {
    expect(artifactPath('ses_1', {
      kind: 'evidence',
      entityId: 'evd_1',
      fieldPath: 'not-applicable',
    })).toBe('/s/ses_1/evidence/evd_1');
  });

  it('keeps the quoted passage in an evidence deep link', () => {
    expect(artifactPath('ses_1', {
      kind: 'evidence',
      entityId: 'evd_1',
      locator: 'table-wrap:1/row:3/cell:2',
    })).toBe('/s/ses_1/evidence/evd_1?locator=table-wrap%3A1%2Frow%3A3%2Fcell%3A2');
  });
});
