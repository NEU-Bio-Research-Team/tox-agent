import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import type { CaseRunChanges } from '../../../shared/api/endpoints';
import { ChangesBody } from './CaseChanges';

const SECTION = { added: [], changed: [], kept: [], removed: [] };
const SECOND: CaseRunChanges = {
  schema_version: 'case-changes-v1', case_id: 'scase_1', run_id: 'run_2', previous_run_id: 'run_1',
  from_revision: 6, to_revision: 13, subject: { subject_key: 'activity:ffff', kept: true },
  updates_by_actor: { user: 1, model: 4, server: 2 }, operations: [],
  context: { ...SECTION, added: [{ id: 'c1', key: 'measured_pIC50', value: '5.4', unit: 'pIC50', source: 'researcher', actor: 'user' }] },
  hypotheses: { ...SECTION, changed: [{ id: 'h1', statement: 'potent on EGFR', status: 'weakened',
    status_reason: 'the measurement is below the interval', actor: 'model',
    changes: [{ field: 'status', before: 'open', after: 'weakened' }] }] },
  evidence: { ...SECTION, added: [{ id: 'e1', claim: 'The lab measured pIC50 5.4', stance: 'contradicts', directness: 'direct' }] },
  uncertainties: { ...SECTION, changed: [{ id: 'u1', status: 'resolved', resolution: "the researcher's assay" }] },
  next_tests: SECTION,
  conclusion: { changed: true, before: {}, after: { can_say: [{ text: 'Measured potency is below the estimate', evidence_ids: ['e1'] }],
    cannot_say: [], status: 'proposed' }, released_kept: { can_say: [], cannot_say: ['Whether it is potent'] } },
  data_scope: { changed: false, before: {}, after: {} },
  run: { disposition: 'partial_facts' },
  artifacts: { compared: true, reason: null, added: [], changed: [], removed: [],
    kept: [{ kind: 'screening', source_id: 'scr_frozen_1', content_sha256: 'a'.repeat(64) }] },
};

describe('CaseChanges', () => {
  afterEach(cleanup);

  it('says what the turn changed, and that the frozen numbers did not', () => {
    render(<ChangesBody changes={SECOND} />);
    const changed = screen.getByRole('list', { name: 'Đã đổi' });
    expect(changed).toHaveTextContent('nhà nghiên cứu thêm bối cảnh: measured_pIC50 = 5.4 pIC50 (nguồn: researcher)');
    expect(changed).toHaveTextContent('h1: đang mở → bị yếu đi — the measurement is below the interval');
    expect(changed).toHaveTextContent("u1 đã giải quyết: the researcher's assay");
    expect(changed).toHaveTextContent('Kết luận được đề xuất (chưa phát hành)');
    expect(changed).toHaveTextContent('Kết luận đã phát hành trước đó vẫn được giữ: chưa thể nói: Whether it is potent');
    const kept = screen.getByRole('list', { name: 'Giữ nguyên' });
    expect(kept).toHaveTextContent('screening scr_frozen_1: cùng hash aaaaaaaaaaaa');
    expect(kept).toHaveTextContent('quyết định policy và trạng thái stage không đổi');
    expect(kept).toHaveTextContent('Đổi target hoặc ligand là một hồ sơ khác');
    expect(screen.getByText(/Revision 6 → 13/)).toHaveTextContent('1 của nhà nghiên cứu');
  });

  it('calls out a changed input hash instead of hiding it', () => {
    render(<ChangesBody changes={{ ...SECOND, artifacts: { ...SECOND.artifacts, kept: [],
      changed: [{ kind: 'screening', source_id: 'scr_frozen_1', before_sha256: 'a'.repeat(64), after_sha256: 'd'.repeat(64) }] } }} />);
    expect(screen.getByRole('list', { name: 'Đã đổi' })).toHaveTextContent('Đầu vào khác lượt trước: screening scr_frozen_1 đổi hash aaaaaaaaaaaa… → dddddddddddd…');
  });

  it('says the first turn has nothing to compare with', () => {
    render(<ChangesBody changes={{ ...SECOND, previous_run_id: null, from_revision: 0,
      artifacts: { ...SECOND.artifacts, compared: false, reason: 'the first run of the case', kept: [] } }} />);
    expect(screen.getByText(/Lượt đầu tiên của hồ sơ/)).toBeInTheDocument();
  });
});
