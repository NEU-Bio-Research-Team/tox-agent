import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import type { ScreeningDetail } from '../../../shared/api/endpoints';
import { ScreeningCard } from './ScreeningCard';

const REVIEW = {
  screening_id: 'scr_1',
  subject: {
    canonical_smiles: 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1',
    target: { target_id: 'EGFR', uniprot_id: 'P00533' },
    assay_context: 'IC50|wild_type',
  },
  subject_key: 'activity:abc',
  intended_target_role: 'intended_target',
  status: 'completed_with_toxicity_not_run',
  stages: {
    activity: { status: 'completed', failure_code: null },
    prioritisation: { status: 'completed', decision: 'review' },
    toxicity: {
      status: 'not_run',
      not_run_reason: 'review_required_awaiting_researcher',
      failure_code: null,
      authorised_by: null,
      frozen_analysis_available: false,
    },
  },
  activity_estimate: {
    estimate: { predicted_pic50: 7.1, prediction_interval: [6.8, 7.4], interval_level: 0.9 },
    applicability: { ligand: 'ok', target: 'ok', joint: 'ok' },
    model: { model_id: 'sair-descriptor-hgb-b0', admitted: false },
  },
  prioritisation_decision: {
    status: 'review',
    reason_codes: ['prediction_interval_crosses_target_threshold'],
    policy: { threshold_pic50: 7.0, threshold_source: 'programme_default' },
  },
  artifact_refs: [],
  required_limitations: [],
  mode: 'replay',
  serving: false,
  model_admitted: false,
  record_sha256: 'abc',
  attached_at: '2026-10-21T00:00:00Z',
} as unknown as ScreeningDetail;

describe('ScreeningCard', () => {
  afterEach(cleanup);

  it('keeps the estimate, the policy decision and the stage statuses apart', () => {
    render(<ScreeningCard screening={REVIEW} />);

    expect(screen.getByText('Ước lượng hoạt tính')).toBeInTheDocument();
    expect(screen.getByText('Quyết định ưu tiên (policy)')).toBeInTheDocument();
    expect(screen.getByText('Trạng thái các stage')).toBeInTheDocument();
    expect(screen.getByText('7.1')).toBeInTheDocument();
    expect(screen.getByText(/6.8–7.4/)).toBeInTheDocument();
    expect(screen.getByText('REVIEW')).toBeInTheDocument();
    expect(screen.getByText(/prediction_interval_crosses_target_threshold/)).toBeInTheDocument();
    // A policy decision is never presented as a measurement, and the model
    // behind the estimate says it is not admitted.
    expect(screen.getByText(/không phải một phép đo/)).toBeInTheDocument();
    expect(screen.getByText(/chưa được admit/)).toBeInTheDocument();
    expect(screen.getByText(/không phải hiệu lực điều trị/)).toBeInTheDocument();
  });

  it('labels a toxicity stage that did not run in words and denies it is a negative', () => {
    render(<ScreeningCard screening={REVIEW} />);

    expect(screen.getByText('chưa chạy')).toBeInTheDocument();
    expect(screen.getByText(/review_required_awaiting_researcher/)).toBeInTheDocument();
    expect(
      screen.getByText('Chưa chạy hoặc thất bại không phải kết quả âm tính.'),
    ).toBeInTheDocument();
  });

  it('says a failed activity stage produced no estimate, with its code', () => {
    const failed = {
      ...REVIEW,
      status: 'activity_failed',
      activity_estimate: null,
      prioritisation_decision: null,
      model_admitted: null,
      stages: {
        ...(REVIEW.stages as Record<string, unknown>),
        activity: { status: 'failed', failure_code: 'activity_model_unavailable' },
        prioritisation: { status: 'not_run', decision: null },
      },
    } as unknown as ScreeningDetail;

    render(<ScreeningCard screening={failed} />);

    expect(screen.getByText(/stage hoạt tính thất bại \(activity_model_unavailable\)/)).toBeInTheDocument();
    expect(
      screen.getByText(/Đây không phải kết quả âm tính hay bằng chứng không hoạt tính/),
    ).toBeInTheDocument();
    expect(screen.getByText('thất bại')).toBeInTheDocument();
    expect(screen.getByText('Chưa có quyết định policy nào cho screening này.')).toBeInTheDocument();
    expect(screen.queryByText('7.1')).not.toBeInTheDocument();
  });

  it('says a replayed screening is read from a frozen record', () => {
    render(<ScreeningCard screening={REVIEW} />);
    expect(screen.getByText(/đã đóng băng \(không chạy mô hình\)/)).toBeInTheDocument();
  });
});
