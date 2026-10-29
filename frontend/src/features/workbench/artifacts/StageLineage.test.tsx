import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import type { RunStageLineage } from '../../../shared/api/endpoints';
import { LineageBody } from './StageLineage';

const BASE: RunStageLineage = {
  schema_version: 'stage-lineage-v1', run_id: 'run_1', recorded: true, release_policy: 'strict-reviewed-v1',
  disposition: 'publish', review_status: 'repaired_and_passed', execution_outcome: 'completed',
  public_answer_id: 'ans_2', candidate_answer_id: 'ans_2',
  stages: [
    { index: 0, phase: 'initial', generation: 1, outcome: 'rejected', answer_id: null, hard_valid: false, is_public: false,
      violation_codes: ['claim_citation_missing'], claim_count: 2, limitation_codes: [], answer_markdown: 'raw draft text', created_at: null },
    { index: 1, phase: 'hard_correction', generation: 2, outcome: 'accepted', answer_id: 'ans_1', hard_valid: true, is_public: false,
      violation_codes: [], claim_count: 2, limitation_codes: ['evidence_scope_limited'], answer_markdown: 'This exact compound blocks hERG.', created_at: null },
    { index: 2, phase: 'semantic_repair', generation: 3, outcome: 'accepted', answer_id: 'ans_2', hard_valid: true, is_public: true,
      violation_codes: [], claim_count: 1, limitation_codes: [], answer_markdown: 'A study reports blockade in a series.', created_at: null },
  ],
  reviews: [
    { round: 1, phase: 'review_initial', status: 'completed', reason: null, answer_id: 'ans_1',
      counts: { supported: 1, not_supported: 1 }, blocking_claim_ids: ['clm_x'],
      verdicts: [{ claim_id: 'clm_x', key: 'overreach', text: 'This exact compound blocks hERG.', verdict: 'not_supported', reason: 'the source tested a series' }] },
    { round: 2, phase: 'review_final', status: 'completed', reason: null, answer_id: 'ans_2', counts: { supported: 1 },
      blocking_claim_ids: [], verdicts: [] },
  ],
  claims: [
    { key: 'lit', kind: 'scientific', candidate_text: 'A study reports blockade in a series.', final_text: 'A study reports blockade in a series.', change: 'kept', verdicts: [] },
    { key: 'overreach', kind: 'scientific', candidate_text: 'This exact compound blocks hERG.', final_text: null, change: 'removed',
      verdicts: [{ round: 1, verdict: 'not_supported', reason: 'the source tested a series' }] },
  ],
  repair: { removal_only: true, lost_limitation_codes: ['evidence_scope_limited'] },
  missing: { needed_inputs: [], unresolved_claim_ids: [], lost_limitation_codes: ['evidence_scope_limited'],
    confirmed_fact_count: 0, reason: null, failure_code: null },
};

describe('StageLineage', () => {
  afterEach(cleanup);

  it('lists the stages in order and marks the one actually released', () => {
    render(<LineageBody lineage={BASE} />);
    const stages = within(screen.getByRole('list', { name: 'Bản nháp theo thứ tự' })).getAllByRole('listitem');
    expect(stages).toHaveLength(3);
    expect(stages[0]).toHaveTextContent('Bản nháp đầu · bị validator từ chối');
    expect(stages[0]).toHaveTextContent('Lỗi cấu trúc: claim_citation_missing');
    expect(stages[1]).toHaveTextContent('không phải câu trả lời');
    expect(stages[2]).toHaveTextContent('bản đang phát hành');
  });

  it('labels a candidate draft as audit material when its text is opened', () => {
    render(<LineageBody lineage={BASE} />);
    const stages = within(screen.getByRole('list', { name: 'Bản nháp theo thứ tự' })).getAllByRole('listitem');
    fireEvent.click(within(stages[1]).getByRole('button', { name: 'Xem văn bản của giai đoạn này' }));
    expect(within(stages[1]).getByText(/Bản nháp, chỉ để audit/)).toBeInTheDocument();
    expect(within(stages[1]).getByText('This exact compound blocks hERG.')).toBeInTheDocument();
  });

  it('reads verdicts against their claims and shows what the repair removed', () => {
    render(<LineageBody lineage={BASE} />);
    expect(screen.getByText(/Review vòng 1 \(lần đầu\)/)).toHaveTextContent('1 được nguồn hỗ trợ, 1 không được hỗ trợ');
    const claims = within(screen.getByRole('list', { name: 'Lineage của claim' })).getAllByRole('listitem');
    expect(claims[1]).toHaveTextContent('overreach · bị xóa · vòng 1: không được hỗ trợ');
    expect(screen.getByText(/Bản sửa chỉ xóa claim/)).toBeInTheDocument();
    expect(screen.getByText(/Bản sửa làm mất limitation: evidence_scope_limited/)).toBeInTheDocument();
  });

  it('says what a withheld run did not give the reader', () => {
    render(<LineageBody lineage={{
      ...BASE, disposition: 'withhold', review_status: 'not_completed', execution_outcome: 'cancelled', public_answer_id: null,
      stages: BASE.stages.slice(0, 2).map((stage) => ({ ...stage, is_public: false })), reviews: [], claims: [], repair: null,
      missing: { needed_inputs: ['an exposure (free Cmax)'], unresolved_claim_ids: [], lost_limitation_codes: [],
        confirmed_fact_count: 1, reason: 'the run ended before an answer could be published', failure_code: 'cancelled' },
    }} />);
    expect(screen.queryByText('bản đang phát hành')).toBeNull();
    expect(screen.getByText(/Cần thêm đầu vào: an exposure \(free Cmax\)/)).toBeInTheDocument();
    expect(screen.getByText(/Run dừng vì: cancelled/)).toBeInTheDocument();
  });

  it('never reports a run before any release policy as reviewed', () => {
    render(<LineageBody lineage={{ ...BASE, recorded: false, release_policy: 'legacy', stages: [], reviews: [], claims: [] }} />);
    expect(screen.getByText(/không biết câu trả lời đã qua review hay chưa/)).toBeInTheDocument();
  });
});
