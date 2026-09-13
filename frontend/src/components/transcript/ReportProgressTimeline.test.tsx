import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { ReportProgress } from '../../lib/store/reportProgress';
import { ReportProgressTimeline } from './ReportProgressTimeline';

function progress(overrides: Partial<ReportProgress> = {}): ReportProgress {
  return {
    buildId: 'rpb_test',
    total: 4,
    completed: 2,
    currentStage: 'synthesizing',
    lastSequence: 9,
    retried: false,
    outcome: null,
    stages: [
      { stage: 'preparing_analysis', status: 'completed', attempt: 1, willRun: true },
      { stage: 'researching_evidence', status: 'skipped', attempt: 1, willRun: false },
      { stage: 'synthesizing', status: 'running', attempt: 1, willRun: true },
      { stage: 'rendering', status: 'pending', attempt: 0, willRun: true },
    ],
    ...overrides,
  };
}

describe('ReportProgressTimeline', () => {
  it('names the current stage and states each status in text, not colour alone', () => {
    render(<ReportProgressTimeline progress={progress()} />);

    expect(screen.getByRole('status')).toHaveTextContent('Viết phần diễn giải (2/4)');
    const steps = within(screen.getByRole('list', { name: 'Các bước tạo báo cáo' })).getAllByRole('listitem');
    expect(steps.map((step) => step.textContent)).toEqual([
      'Chuẩn bị phân tích — xong',
      'Tìm bằng chứng tài liệu — bỏ qua theo yêu cầu',
      'Viết phần diễn giải — đang chạy',
      'Xuất báo cáo — chưa bắt đầu',
    ]);
    expect(steps[2]).toHaveAttribute('aria-current', 'step');
  });

  it('shows a retry and a failed outcome without claiming a report exists', () => {
    render(
      <ReportProgressTimeline
        progress={progress({
          retried: true,
          outcome: 'failed',
          currentStage: null,
          stages: [{ stage: 'synthesizing', status: 'failed', attempt: 2, willRun: true }],
        })}
      />,
    );

    expect(screen.getByRole('status')).toHaveTextContent('Không tạo được báo cáo.');
    expect(screen.getByRole('status')).toHaveTextContent('có bước phải chạy lại');
    expect(screen.getByRole('listitem')).toHaveTextContent('Viết phần diễn giải — lỗi (lần thử 2)');
  });

  it('falls back to one line for the legacy single-stage event', () => {
    render(<ReportProgressTimeline progress={progress({ stages: [], currentStage: 'synthesizing' })} />);
    expect(screen.getByRole('status')).toHaveTextContent('Viết phần diễn giải…');
    expect(screen.queryByRole('list')).toBeNull();
  });
});
