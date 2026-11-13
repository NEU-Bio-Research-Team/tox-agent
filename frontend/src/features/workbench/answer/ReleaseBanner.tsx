import { Check, CircleAlert, CircleCheck, CircleHelp, Minus, Wrench } from 'lucide-react';
import type { RunResult } from '../../../shared/api/endpoints';
import { Callout, type Tone } from '../../../shared/ui/callout';
import { cn } from '../../../shared/ui/utils';

/**
 * Cách một câu trả lời đã được kiểm tra, nói bằng lời thường (#47, §10.3).
 *
 * "Khớp nguồn" không bao giờ được trình bày thành "đúng về khoa học": lời nói
 * đúng việc hệ thống đã làm. Một run đời cũ không có bản ghi kiểm tra thì ghi là
 * *chưa có thông tin kiểm chứng*, không mặc định là đã qua.
 *
 * Khi kết quả bị giữ lại, thẻ nói ba điều: dữ kiện nào xác nhận được, ý nào chưa
 * có nguồn hỗ trợ, và cần gì để đi tiếp. Ba bước phía dưới cho thấy câu trả lời
 * dừng ở đâu, để người đọc không phải biết tên các bước kỹ thuật.
 */
export function ReleaseBanner({ result }: { result: RunResult | null | undefined }) {
  if (!result) return null;
  const view = releaseView(result);
  return (
    <Callout tone={view.tone} icon={<view.Icon />} title={view.title} role="status" aria-label="Trạng thái phát hành" className="text-13">
      <p className="text-ink-secondary">{view.detail}</p>
      {view.steps.length > 0 ? <Steps steps={view.steps} /> : null}
    </Callout>
  );
}

type StepState = 'done' | 'issue' | 'todo' | 'skipped';
interface Step {
  label: string;
  state: StepState;
}

const STEP_STATE_TEXT: Record<StepState, string> = {
  done: 'xong',
  issue: 'chưa đạt',
  todo: 'chưa tới',
  skipped: 'bỏ qua',
};

/** Where the answer got to: written, checked against sources, published. */
function Steps({ steps }: { steps: Step[] }) {
  return (
    <ol aria-label="Các bước kiểm tra" className="mt-2.5 flex flex-wrap items-center gap-x-1 gap-y-1.5">
      {steps.map((step, index) => (
        <li key={step.label} className="flex items-center gap-1">
          {index > 0 ? <span className="mx-1 h-px w-4 bg-line-strong" aria-hidden="true" /> : null}
          <span
            className={cn(
              'flex size-[18px] items-center justify-center rounded-full border text-ink',
              step.state === 'done' && 'border-success bg-success text-canvas',
              step.state === 'issue' && 'border-warning bg-warning text-canvas',
              step.state === 'todo' && 'border-line-strong bg-transparent text-ink-tertiary',
              step.state === 'skipped' && 'border-dashed border-line-strong bg-transparent text-ink-tertiary',
            )}
            aria-hidden="true"
          >
            {step.state === 'done' ? <Check className="size-3" strokeWidth={3} /> : null}
            {step.state === 'issue' ? <span className="text-2xs font-bold leading-none">!</span> : null}
            {step.state === 'skipped' ? <Minus className="size-3" /> : null}
          </span>
          <span className={cn('text-xs', step.state === 'todo' || step.state === 'skipped' ? 'text-ink-tertiary' : 'font-medium text-ink')}>
            {step.label}
            <span className="sr-only"> ({STEP_STATE_TEXT[step.state]})</span>
          </span>
        </li>
      ))}
    </ol>
  );
}

interface ReleaseView {
  title: string;
  detail: string;
  tone: Tone;
  Icon: typeof CircleCheck;
  steps: Step[];
}

export function releaseView(result: RunResult): ReleaseView {
  if (!result.recorded) {
    return {
      title: 'Chưa có thông tin kiểm chứng',
      detail: 'Câu trả lời này được tạo trước khi hệ thống ghi nhận việc kiểm tra, nên chưa rõ nó đã được kiểm tra hay chưa.',
      tone: 'neutral',
      Icon: CircleHelp,
      steps: [],
    };
  }
  const reviewed = result.review_status;
  if (result.disposition === 'publish' && reviewed === 'repaired_and_passed') {
    return {
      title: 'Đã chỉnh lại một lần rồi mới khớp nguồn',
      detail:
        'Một ý không được nguồn hỗ trợ nên đã được bỏ hoặc thu hẹp. Phần còn lại khớp nguồn; điều đó chưa có nghĩa kết luận chắc chắn đúng về khoa học.',
      tone: 'info',
      Icon: Wrench,
      steps: [
        { label: 'Viết nháp', state: 'done' },
        { label: 'Chỉnh lại', state: 'done' },
        { label: 'Đối chiếu nguồn', state: 'done' },
        { label: 'Công bố', state: 'done' },
      ],
    };
  }
  if (result.disposition === 'publish') {
    const skipped = reviewed === 'not_reviewed_by_design';
    return {
      title: skipped ? 'Chưa đọc soát lại nội dung' : 'Số liệu đã được đối chiếu với nguồn',
      detail: skipped
        ? 'Bản triển khai này chỉ kiểm tra số liệu và định dạng, không có bước đọc soát lại ý nghĩa của câu trả lời.'
        : 'Mỗi con số trong câu trả lời khớp với kết quả mô hình hoặc bài báo mà nó dẫn. Khớp nguồn chưa có nghĩa kết luận chắc chắn đúng về khoa học.',
      tone: skipped ? 'neutral' : 'success',
      Icon: skipped ? CircleHelp : CircleCheck,
      steps: [
        { label: 'Viết nháp', state: 'done' },
        { label: 'Đối chiếu nguồn', state: skipped ? 'skipped' : 'done' },
        { label: 'Công bố', state: 'done' },
      ],
    };
  }
  const needed = neededInputs(result);
  const facts = listOf(result, 'confirmed_facts').length;
  const unresolved = listOf(result, 'unresolved_claim_ids').length;
  return {
    title: result.disposition === 'partial_facts' ? 'Chỉ công bố dữ kiện xác nhận được' : 'Kết quả bị giữ lại, chưa công bố',
    detail:
      `${facts > 0 ? `${facts} dữ kiện đã xác nhận vẫn được giữ lại; ` : ''}` +
      `${unresolved > 0 ? `${unresolved} ý chưa có nguồn hỗ trợ nên không hiển thị; ` : ''}` +
      `${reviewOutcomeVi(reviewed)}.` +
      (needed.length > 0 ? ` Cần thêm: ${needed.join('; ')}.` : ' Bản nháp chỉ xem được ở màn kiểm chứng.'),
    tone: 'warning',
    Icon: CircleAlert,
    steps: [
      { label: 'Viết nháp', state: 'done' },
      { label: 'Đối chiếu nguồn', state: 'issue' },
      { label: result.disposition === 'partial_facts' ? 'Công bố phần đã xác nhận' : 'Công bố', state: result.disposition === 'partial_facts' ? 'done' : 'todo' },
    ],
  };
}

function neededInputs(result: RunResult): string[] {
  return listOf(result, 'needed_inputs').map(String);
}

function listOf(result: RunResult, key: string): unknown[] {
  const value = ((result.detail ?? {}) as Record<string, unknown>)[key];
  return Array.isArray(value) ? value : [];
}

function reviewOutcomeVi(status: string | null | undefined): string {
  switch (status) {
    case 'blocked':
      return 'vẫn còn ý chưa được nguồn hỗ trợ';
    case 'cannot_judge':
      return 'chưa đủ căn cứ để đánh giá';
    case 'incomplete':
      return 'việc đối chiếu chưa hoàn tất';
    case 'timed_out':
      return 'việc đối chiếu quá thời gian';
    case 'not_applicable':
      return 'không có ý nào cần đối chiếu';
    default:
      return 'chưa rõ kết quả đối chiếu';
  }
}
