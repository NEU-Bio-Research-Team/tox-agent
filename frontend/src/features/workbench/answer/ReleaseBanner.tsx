import { AlertTriangle, CheckCircle2, HelpCircle, Wrench } from 'lucide-react';
import type { RunResult } from '../../../shared/api/endpoints';

/**
 * Trạng thái phát hành của một câu trả lời (#47, §10.3).
 *
 * "Đã kiểm tra nguồn hỗ trợ" không bao giờ được trình bày thành "đúng về khoa
 * học": nhãn nói đúng việc hệ thống đã làm. Một run đời cũ không có release
 * record thì ghi là *chưa có thông tin kiểm chứng*, không mặc định là đã pass.
 *
 * Khi kết quả bị giữ lại, banner nói ba điều: dữ kiện nào xác nhận được, phần
 * nào chưa kết luận, và cần gì để tiếp — chứ không chỉ một lỗi đỏ.
 */
export function ReleaseBanner({ result }: { result: RunResult | null | undefined }) {
  if (!result) return null;
  const view = releaseView(result);
  return (
    <div
      className="flex items-start gap-2 rounded-md border px-3 py-2 text-xs"
      style={{ borderColor: view.color, color: view.color }}
      role="status"
      aria-label="Trạng thái phát hành"
    >
      <view.Icon className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
      <div className="space-y-0.5" style={{ color: 'var(--text)' }}>
        <p className="font-semibold" style={{ color: view.color }}>
          {view.title}
        </p>
        <p style={{ color: 'var(--text-muted)' }}>{view.detail}</p>
      </div>
    </div>
  );
}

interface ReleaseView {
  title: string;
  detail: string;
  color: string;
  Icon: typeof CheckCircle2;
}

export function releaseView(result: RunResult): ReleaseView {
  if (!result.recorded) {
    return {
      title: 'Chưa có thông tin kiểm chứng',
      detail:
        'Run này chạy trước khi deployment khai báo release policy, nên không biết câu trả lời đã qua review hay chưa.',
      color: 'var(--text-muted)',
      Icon: HelpCircle,
    };
  }
  const reviewed = result.review_status;
  if (result.disposition === 'publish' && reviewed === 'repaired_and_passed') {
    return {
      title: 'Đã sửa một lượt rồi qua kiểm tra hỗ trợ',
      detail:
        'Một claim bị review chặn, bản sửa đã bỏ hoặc thu hẹp claim đó và qua được kiểm tra. Đây là kiểm tra nguồn hỗ trợ, không phải xác nhận đúng về khoa học.',
      color: 'var(--accent-blue)',
      Icon: Wrench,
    };
  }
  if (result.disposition === 'publish') {
    return {
      title: reviewed === 'not_reviewed_by_design' ? 'Phát hành không qua review (chế độ đã khai)' : 'Đã kiểm tra hỗ trợ của nguồn',
      detail:
        reviewed === 'not_reviewed_by_design'
          ? 'Deployment này chạy chế độ hard-only đã khai trước: chỉ validator cấu trúc, không có semantic review.'
          : 'Mỗi claim được đối chiếu với nguồn nó dẫn. Điều này không có nghĩa kết luận đúng về khoa học.',
      color: 'var(--accent-green)',
      Icon: CheckCircle2,
    };
  }
  const needed = neededInputs(result);
  const facts = listOf(result, 'confirmed_facts').length;
  const unresolved = listOf(result, 'unresolved_claim_ids').length;
  return {
    title:
      result.disposition === 'partial_facts'
        ? 'Chỉ công bố dữ kiện xác nhận được'
        : 'Kết quả bị giữ lại, chưa công bố',
    detail:
      `${facts > 0 ? `${facts} dữ kiện server xác nhận được vẫn được giữ; ` : ''}` +
      `${unresolved > 0 ? `${unresolved} claim không được nguồn hỗ trợ nên không hiển thị; ` : ''}` +
      `${reviewOutcomeVi(reviewed)}.` +
      (needed.length > 0 ? ` Cần tiếp: ${needed.join('; ')}.` : ' Bản nháp chỉ xem được trong audit.'),
    color: 'var(--accent-yellow)',
    Icon: AlertTriangle,
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
      return 'review còn chặn claim';
    case 'cannot_judge':
      return 'review không thể phán định';
    case 'incomplete':
      return 'review chưa hoàn tất';
    case 'timed_out':
      return 'review hết thời gian';
    case 'not_applicable':
      return 'không có claim nào cần review';
    default:
      return `trạng thái review: ${status ?? 'không rõ'}`;
  }
}
