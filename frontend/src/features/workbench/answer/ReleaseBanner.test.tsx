import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import type { RunResult } from '../../../shared/api/endpoints';
import { ReleaseBanner } from './ReleaseBanner';

function result(overrides: Partial<RunResult> = {}): RunResult {
  return {
    run_id: 'run_1',
    recorded: true,
    release_policy: 'strict-reviewed-v1',
    release_policy_sha256: 'abc',
    disposition: 'publish',
    review_status: 'passed',
    execution_outcome: 'completed',
    knowledge_outcome: 'answered_from_artifacts',
    public_answer_id: 'ans_1',
    candidate_answer_id: 'ans_1',
    detail: {},
    ...overrides,
  } as RunResult;
}

describe('ReleaseBanner', () => {
  afterEach(cleanup);

  it('says the claims were checked against their sources, not that they are right', () => {
    render(<ReleaseBanner result={result()} />);
    expect(screen.getByText('Đã kiểm tra hỗ trợ của nguồn')).toBeInTheDocument();
    expect(screen.getByText(/không có nghĩa kết luận đúng về khoa học/)).toBeInTheDocument();
  });

  it('distinguishes an answer that was repaired once and then passed', () => {
    render(<ReleaseBanner result={result({ review_status: 'repaired_and_passed' })} />);
    expect(screen.getByText('Đã sửa một lượt rồi qua kiểm tra hỗ trợ')).toBeInTheDocument();
    expect(screen.getByText(/bỏ hoặc thu hẹp claim/)).toBeInTheDocument();
  });

  it('tells the reader what was confirmed, what is open and what is needed when withheld', () => {
    render(
      <ReleaseBanner
        result={result({
          disposition: 'partial_facts',
          review_status: 'blocked',
          public_answer_id: null,
          detail: {
            confirmed_facts: [{ text: 'hERG probability', value: 0.81 }],
            unresolved_claim_ids: ['c2'],
            needed_inputs: ['free plasma Cmax ở liều dự kiến'],
          },
        })}
      />,
    );
    expect(screen.getByText('Chỉ công bố dữ kiện xác nhận được')).toBeInTheDocument();
    const detail = screen.getByText(/dữ kiện server xác nhận được vẫn được giữ/);
    expect(detail.textContent).toContain('1 claim không được nguồn hỗ trợ nên không hiển thị');
    expect(detail.textContent).toContain('review còn chặn claim');
    expect(detail.textContent).toContain('free plasma Cmax ở liều dự kiến');
  });

  it('does not call a reviewer that could not judge a pass', () => {
    render(
      <ReleaseBanner result={result({ disposition: 'withhold', review_status: 'cannot_judge' })} />,
    );
    expect(screen.getByText('Kết quả bị giữ lại, chưa công bố')).toBeInTheDocument();
    expect(screen.getByText(/review không thể phán định/)).toBeInTheDocument();
    expect(screen.queryByText(/Đã kiểm tra hỗ trợ/)).not.toBeInTheDocument();
  });

  it('calls a legacy run unknown rather than passed', () => {
    render(
      <ReleaseBanner
        result={{ run_id: 'run_0', recorded: false, release_policy: 'legacy' } as RunResult}
      />,
    );
    expect(screen.getByText('Chưa có thông tin kiểm chứng')).toBeInTheDocument();
    expect(screen.getByText(/không biết câu trả lời đã qua review hay chưa/)).toBeInTheDocument();
  });

  it('renders nothing when the run has no result', () => {
    const { container } = render(<ReleaseBanner result={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('says so when a deployment publishes without review by design', () => {
    render(
      <ReleaseBanner result={result({ review_status: 'not_reviewed_by_design' })} />,
    );
    expect(screen.getByText('Phát hành không qua review (chế độ đã khai)')).toBeInTheDocument();
    expect(screen.getByText(/hard-only đã khai trước/)).toBeInTheDocument();
  });
});
